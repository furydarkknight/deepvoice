"""
Prepare ASVspoof 2019 LA dataset into train/dev/eval CSVs.

The ASVspoof 2019 LA protocol files contain lines like:
  LA_0001 LA_T_1138215 - - bonafide
  LA_0002 LA_T_1271820 - - spoof

Column meanings: speaker_id, file_id, -, -, label
"""

import argparse
import csv
import os
from pathlib import Path


SPLIT_MAP = {
    "train": "ASVspoof2019_LA_train",
    "dev":   "ASVspoof2019_LA_dev",
    "eval":  "ASVspoof2019_LA_eval",
}

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def prepare(data_dir: str, out_dir: str):
    data_dir  = Path(data_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_dir = data_dir / "ASVspoof2019_LA_cm_protocols"

    for split, audio_folder in SPLIT_MAP.items():
        proto_path = proto_dir / PROTOCOL_FILES[split]
        audio_dir  = data_dir  / audio_folder / "flac"

        if not proto_path.exists():
            print(f"[WARN] Protocol file not found: {proto_path} — skipping {split}")
            continue
        if not audio_dir.exists():
            print(f"[WARN] Audio directory not found: {audio_dir} — skipping {split}")
            continue

        rows = []
        with open(proto_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id  = parts[1]
                label_str = parts[4]
                label    = LABEL_MAP.get(label_str, -1)
                if label == -1:
                    continue

                audio_path = audio_dir / f"{file_id}.flac"
                if audio_path.exists():
                    rows.append({"path": str(audio_path), "label": label})

        out_csv = out_dir / f"{split}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)

        real_n = sum(1 for r in rows if r["label"] == 0)
        fake_n = sum(1 for r in rows if r["label"] == 1)
        print(f"[{split:5s}] {len(rows):6d} samples  "
              f"(real={real_n}, fake={fake_n})  → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASVspoof 2019 LA dataset")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory of ASVspoof 2019 LA")
    parser.add_argument("--out_dir",  default="data/processed",
                        help="Output directory for CSV files")
    args = parser.parse_args()
    prepare(args.data_dir, args.out_dir)
