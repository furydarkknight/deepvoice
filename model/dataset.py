"""
PyTorch Dataset classes for ASVspoof 2019 LA and WaveFake.
Both datasets are unified under a single CSV format: path,label (0=real,1=fake).
"""

import random
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Literal

from model.features import preprocess


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------

class AntiSpoofDataset(Dataset):
    """
    Reads a CSV with columns: path, label (0=real, 1=fake).
    Supports both ASVspoof 2019 LA and WaveFake entries in the same CSV.
    """

    def __init__(
        self,
        csv_path: str,
        split: Literal["train", "dev", "eval"] = "train",
        augment: bool = False,
    ):
        df = pd.read_csv(csv_path)
        # Validate columns
        assert "path" in df.columns and "label" in df.columns, \
            "CSV must have 'path' and 'label' columns"
        self.df      = df.reset_index(drop=True)
        self.split   = split
        self.augment = augment and (split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        path  = row["path"]
        label = int(row["label"])

        features = preprocess(path, augment=self.augment)

        return {
            "raw":     features["raw"],
            "lfcc":    features["lfcc"],
            "log_mel": features["log_mel"],
            "label":   torch.tensor(label, dtype=torch.long),
            "path":    path,
        }


# ---------------------------------------------------------------------------
# Class-balanced sampler (handles heavy class imbalance in ASVspoof)
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: AntiSpoofDataset) -> WeightedRandomSampler:
    labels = dataset.df["label"].values
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    return {
        "raw":     torch.stack([b["raw"]     for b in batch]),
        "lfcc":    torch.stack([b["lfcc"]    for b in batch]),
        "log_mel": torch.stack([b["log_mel"] for b in batch]),
        "label":   torch.stack([b["label"]   for b in batch]),
        "path":    [b["path"]                for b in batch],
    }
