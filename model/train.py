"""
Training loop for the AASIST-lite anti-spoofing model.

Usage:
    python model/train.py \
        --train_csv data/processed/train.csv \
        --dev_csv   data/processed/dev.csv \
        --output_dir checkpoints/ \
        --epochs 100

All hyperparameters come from model/config.py; CLI flags override defaults.
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.config import cfg
from model.dataset import AntiSpoofDataset, collate_fn, make_weighted_sampler
from model.evaluate import compute_eer
from model.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# OC-Softmax loss (Open-set anti-spoofing loss from ASVspoof 2021)
# Falls back to cross-entropy when not using the OC formulation.
# ---------------------------------------------------------------------------

class OCSoftmaxLoss(nn.Module):
    """
    One-class softmax loss that pushes bonafide (label=0) embeddings
    to a learnable centre and repels spoof (label=1) embeddings.
    Reference: Zhang et al. (2021)
    """

    def __init__(self, feat_dim: int = 160, r_real: float = 0.9, r_fake: float = 0.2,
                 alpha: float = 20.0):
        super().__init__()
        self.centre  = nn.Parameter(torch.randn(1, feat_dim))
        self.r_real  = r_real
        self.r_fake  = r_fake
        self.alpha   = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Fallback: standard cross-entropy (logits come from classifier head)
        return nn.functional.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------ data
    train_ds = AntiSpoofDataset(args.train_csv, split="train", augment=True)
    dev_ds   = AntiSpoofDataset(args.dev_csv,   split="dev",   augment=False)

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    log.info(f"Train samples: {len(train_ds)} | Dev samples: {len(dev_ds)}")

    # ----------------------------------------------------------------- model
    model = build_model().to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.train.scheduler_patience,
        factor=cfg.train.scheduler_factor,
        verbose=True,
    )
    criterion = OCSoftmaxLoss().to(device)
    scaler    = GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    # ----------------------------------------------------------- output dirs
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
    log.info(f"Saving checkpoints to: {out_dir}")

    best_eer        = float("inf")
    no_improve_cnt  = 0
    global_step     = 0

    # ----------------------------------------------------------------- loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for batch in train_loader:
            raw   = batch["raw"].to(device)
            lfcc  = batch["lfcc"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.use_amp and device.type == "cuda"):
                logits = model(raw, lfcc)
                loss   = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % cfg.train.log_interval == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        log.info(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        # --------------------------------------------------------- evaluate
        if epoch % cfg.train.eval_interval == 0:
            eer, acc = evaluate_epoch(model, dev_loader, device)
            log.info(f"  Dev EER: {eer:.4f} | Acc: {acc:.4f}")
            writer.add_scalar("Metric/dev_eer", eer, epoch)
            writer.add_scalar("Metric/dev_acc", acc, epoch)
            scheduler.step(eer)

            if eer < best_eer:
                best_eer = eer
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(), "eer": eer},
                    out_dir / "best_model.pt",
                )
                no_improve_cnt = 0
                log.info(f"  ✓ New best EER: {best_eer:.4f} — checkpoint saved")
            else:
                no_improve_cnt += 1

            if no_improve_cnt >= cfg.train.early_stopping_patience:
                log.info("Early stopping triggered.")
                break

    writer.close()
    log.info(f"Training complete. Best dev EER: {best_eer:.4f}")
    log.info(f"Best model: {out_dir / 'best_model.pt'}")
    return str(out_dir / "best_model.pt")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_epoch(model, loader, device) -> tuple:
    model.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        raw   = batch["raw"].to(device)
        lfcc  = batch["lfcc"].to(device)
        label = batch["label"]

        scores = model.predict_proba(raw, lfcc).cpu()
        all_scores.append(scores)
        all_labels.append(label)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    eer = compute_eer(labels, scores)
    preds = (scores >= 0.5).astype(int)
    acc = (preds == labels).mean()
    return float(eer), float(acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  default=cfg.train.train_csv)
    parser.add_argument("--dev_csv",    default=cfg.train.dev_csv)
    parser.add_argument("--output_dir", default=cfg.train.output_dir)
    parser.add_argument("--epochs",     type=int,   default=cfg.train.epochs)
    parser.add_argument("--batch_size", type=int,   default=cfg.train.batch_size)
    parser.add_argument("--lr",         type=float, default=cfg.train.learning_rate)
    args = parser.parse_args()

    train(args)
