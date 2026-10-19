"""
Jupyter notebook (as a plain Python script) demonstrating the full
ASVspoof 2019 + WaveFake training pipeline.
Run with: jupyter nbconvert --to notebook --execute training_demo.ipynb
Or just run cells in Jupyter directly.
"""

# %% [markdown]
# # VoiceShield Training Demo
# Full walk-through: data prep → training → evaluation → ONNX export

# %% Setup
import sys
sys.path.insert(0, '..')     # project root

# %% [markdown]
# ## 1. Prepare datasets
#
# Before running this cell, make sure you have downloaded:
# - ASVspoof 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336
# - WaveFake: https://zenodo.org/record/5642694

# %% Prepare ASVspoof
import subprocess
result = subprocess.run([
    'python', '../data/scripts/prepare_asvspoof.py',
    '--data_dir', '../data/raw/asvspoof2019',
    '--out_dir',  '../data/processed',
], capture_output=True, text=True)
print(result.stdout or result.stderr)

# %% Prepare WaveFake (appends to CSVs created above)
result = subprocess.run([
    'python', '../data/scripts/prepare_wavefake.py',
    '--data_dir', '../data/raw/wavefake',
    '--real_dir', '../data/raw/wavefake/real',
    '--out_dir',  '../data/processed',
], capture_output=True, text=True)
print(result.stdout or result.stderr)

# %% [markdown]
# ## 2. Inspect dataset

# %% Dataset stats
import pandas as pd

for split in ['train', 'dev', 'eval']:
    df = pd.read_csv(f'../data/processed/{split}.csv')
    real = (df.label == 0).sum()
    fake = (df.label == 1).sum()
    print(f'{split:5s}: {len(df):7,d} total  |  real={real:,}  fake={fake:,}')

# %% [markdown]
# ## 3. Train the model

# %% Train (short demo — use --epochs 100 for production)
result = subprocess.run([
    'python', '../model/train.py',
    '--train_csv',  '../data/processed/train.csv',
    '--dev_csv',    '../data/processed/dev.csv',
    '--output_dir', '../checkpoints',
    '--epochs',     '5',           # increase to 100 for real training
    '--batch_size', '16',
], capture_output=False, text=True)

# %% [markdown]
# ## 4. Evaluate

# %% Evaluate (replace checkpoint path with your actual path)
import glob, os

checkpoints = sorted(glob.glob('../checkpoints/**/best_model.pt', recursive=True))
if checkpoints:
    latest_ckpt = checkpoints[-1]
    print(f'Evaluating: {latest_ckpt}')
    result = subprocess.run([
        'python', '../model/evaluate.py',
        '--eval_csv',   '../data/processed/eval.csv',
        '--checkpoint', latest_ckpt,
    ], capture_output=False)
else:
    print('No checkpoint found — run training first')

# %% [markdown]
# ## 5. Export to ONNX

# %% ONNX export
if checkpoints:
    subprocess.run([
        'python', '../model/export.py',
        '--checkpoint', latest_ckpt,
        '--output',     '../backend/app/model.onnx',
    ])
    print('ONNX model exported to backend/app/model.onnx')
