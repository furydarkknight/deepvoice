"""
Export trained PyTorch model to ONNX for fast CPU inference in the backend.

Usage:
    python model/export.py \
        --checkpoint checkpoints/<run>/best_model.pt \
        --output     backend/app/model.onnx
"""

import argparse
import torch
from pathlib import Path

from model.config import cfg
from model.model import build_model


def export_onnx(checkpoint_path: str, output_path: str):
    device = torch.device("cpu")

    model = build_model().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Dummy inputs matching model forward signature
    raw  = torch.randn(1, cfg.feature.raw_chunk_length)
    lfcc = torch.randn(1, cfg.feature.n_lfcc, 256)          # approximate frame count

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (raw, lfcc),
        output_path,
        opset_version=14,
        input_names=["raw", "lfcc"],
        output_names=["logits"],
        dynamic_axes={
            "raw":    {0: "batch"},
            "lfcc":   {0: "batch", 2: "time"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
    )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    out  = sess.run(None, {"raw": raw.numpy(), "lfcc": lfcc.numpy()})
    print(f"✓ ONNX export verified. Output shape: {out[0].shape}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="backend/app/model.onnx")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)
