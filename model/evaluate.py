"""
Evaluation metrics: EER (Equal Error Rate) and min t-DCF
(minimum normalised tandem Detection Cost Function) for ASVspoof 2019.

Reference: ASVspoof 2019 evaluation plan.
"""

import argparse
import logging

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from torch.utils.data import DataLoader

from model.config import cfg
from model.dataset import AntiSpoofDataset, collate_fn
from model.model import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate.
    labels: 0=real/bonafide, 1=fake/spoof
    scores: float in [0,1] — P(fake)
    """
    fpr, fnr = [], []
    thresholds = np.linspace(0, 1, 500)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fpr.append(fp / (fp + tn + 1e-8))
        fnr.append(fn / (fn + tp + 1e-8))

    fpr = np.array(fpr)
    fnr = np.array(fnr)

    # Interpolate EER
    try:
        eer = brentq(interp1d(thresholds, fpr - fnr), 0, 1)
    except ValueError:
        eer = (fpr + fnr).min() / 2

    return float(eer)


# ---------------------------------------------------------------------------
# min t-DCF (simplified; full implementation follows ASVspoof 2019 eval kit)
# ---------------------------------------------------------------------------

def compute_min_tdcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """
    Compute the minimum normalised tandem-DCF.
    Simplified version without ASV scores (uses CM scores only).
    """
    thresholds = np.unique(scores)
    tdcf_vals  = []
    for t in thresholds:
        preds  = (scores >= t).astype(int)
        p_miss = ((preds == 0) & (labels == 1)).sum() / (labels == 1).sum()
        p_fa   = ((preds == 1) & (labels == 0)).sum() / (labels == 0).sum()
        tdcf   = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        tdcf_vals.append(tdcf)

    # Normalise
    tdcf_default = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(min(tdcf_vals)) / tdcf_default


# ---------------------------------------------------------------------------
# Full evaluation script
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(checkpoint_path: str, eval_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"Loaded checkpoint: {checkpoint_path}")

    # Dataset
    ds = AntiSpoofDataset(eval_csv, split="eval", augment=False)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    all_scores, all_labels = [], []
    for batch in loader:
        raw   = batch["raw"].to(device)
        lfcc  = batch["lfcc"].to(device)
        label = batch["label"]
        scores = model.predict_proba(raw, lfcc).cpu()
        all_scores.append(scores)
        all_labels.append(label)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    eer     = compute_eer(labels, scores)
    min_dcf = compute_min_tdcf(labels, scores)
    preds   = (scores >= 0.5).astype(int)
    acc     = (preds == labels).mean()

    # Per-class
    real_acc = ((preds == 0) & (labels == 0)).sum() / (labels == 0).sum()
    fake_acc = ((preds == 1) & (labels == 1)).sum() / (labels == 1).sum()

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  EER         : {eer*100:.2f}%")
    print(f"  min t-DCF   : {min_dcf:.4f}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Real recall : {real_acc*100:.2f}%")
    print(f"  Fake recall : {fake_acc*100:.2f}%")
    print("=" * 50 + "\n")

    return {"eer": eer, "min_tdcf": min_dcf, "accuracy": acc}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv",    default=cfg.train.eval_csv)
    parser.add_argument("--checkpoint",  required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.eval_csv)
