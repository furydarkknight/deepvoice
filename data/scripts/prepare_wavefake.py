"""
Prepare WaveFake dataset into train/dev/eval CSVs.

WaveFake structure (Zenodo release):
  wavefake/
    real/                  ← LJSpeech original recordings
    ljspeech_full_band_melgan/
    ljspeech_hifiGAN/
    ljspeech_melgan/
    ljspeech_melgan_large/
    ljspeech_multi_band_melgan/
    ljspeech_parallel_wavegan/
    ljspeech_waveglow/

All .wav files in each generator folder are fake; all .wav in real/ are real.
The script creates an 80/10/10 train/dev/eval split stratified by label.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


GENERATOR_DIRS = [
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]


def collect_files(data_dir: Path, real_dir: Path) -> List[Tuple[str, int]]:
    rows = []

    # Real samples
    for ext in ("*.wav", "*.flac"):
        for p in sorted(real_dir.glob(ext)):
            rows.append((str(p), 0))

    # Fake samples from every generator
    for gen in GENERATOR_DIRS:
        gen_dir = data_dir / gen
        if not gen_dir.exists():
            print(f"[WARN] Generator dir not found: {gen_dir}")
            continue
        for ext in ("*.wav", "*.flac"):
            for p in sorted(gen_dir.glob(ext)):
                rows.append((str(p), 1))

    return rows


def split_data(rows, train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.seed(seed)
    real = [r for r in rows if r[1] == 0]
    fake = [r for r in rows if r[1] == 1]

    def _split(lst):
        random.shuffle(lst)
        n     = len(lst)
        n_tr  = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        return lst[:n_tr], lst[n_tr: n_tr + n_dev], lst[n_tr + n_dev:]

    tr_r, dv_r, ev_r = _split(real)
    tr_f, dv_f, ev_f = _split(fake)
    return (tr_r + tr_f), (dv_r + dv_f), (ev_r + ev_f)


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        for path, label in rows:
            writer.writerow({"path": path, "label": label})


def prepare(data_dir: str, real_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    real_dir = Path(real_dir)
    out_dir  = Path(out_dir)

    rows = collect_files(data_dir, real_dir)
    print(f"Total files collected: {len(rows)}")

    train_rows, dev_rows, eval_rows = split_data(rows)

    for split_name, split_rows in [("train", train_rows), ("dev", dev_rows), ("eval", eval_rows)]:
        # For WaveFake we append to existing CSVs from ASVspoof if present
        out_csv = out_dir / f"{split_name}.csv"
        mode    = "a" if out_csv.exists() else "w"
        with open(out_csv, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            if mode == "w":
                writer.writeheader()
            for path, label in split_rows:
                writer.writerow({"path": path, "label": label})
        r = sum(1 for _, l in split_rows if l == 0)
        k = sum(1 for _, l in split_rows if l == 1)
        print(f"[{split_name:5s}] {len(split_rows):6d} samples "
              f"(real={r}, fake={k})  → {out_csv}  (mode={mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WaveFake dataset")
    parser.add_argument("--data_dir", required=True,
                        help="Root WaveFake directory (contains generator subdirs)")
    parser.add_argument("--real_dir", required=True,
                        help="Path to real/ audio directory")
    parser.add_argument("--out_dir",  default="data/processed",
                        help="Output directory for CSV files")
    args = parser.parse_args()
    prepare(args.data_dir, args.real_dir, args.out_dir)
