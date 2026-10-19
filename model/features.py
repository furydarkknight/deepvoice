"""
Feature extraction: LFCC, MFCC, Log-Mel spectrogram, and raw waveform utils.
All features are computed on-the-fly in the Dataset __getitem__ for flexibility.
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch import Tensor

from model.config import cfg

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> Tensor:
    """Load audio, resample to target SR, convert to mono 1-D tensor."""
    wav, sr = torchaudio.load(path)
    if sr != cfg.feature.sample_rate:
        wav = T.Resample(orig_freq=sr, new_freq=cfg.feature.sample_rate)(wav)
    wav = wav.mean(dim=0)           # stereo → mono
    return wav                      # shape: (T,)


def _pad_or_trim(wav: Tensor, target_len: int) -> Tensor:
    """Repeat-pad or centre-trim a waveform to `target_len` samples."""
    if wav.shape[0] >= target_len:
        # Centre crop
        start = (wav.shape[0] - target_len) // 2
        return wav[start: start + target_len]
    # Repeat pad
    repeats = target_len // wav.shape[0] + 1
    wav = wav.repeat(repeats)
    return wav[:target_len]


# ---------------------------------------------------------------------------
# LFCC (Linear Frequency Cepstral Coefficients)
# ---------------------------------------------------------------------------

def compute_lfcc(wav: Tensor) -> Tensor:
    """
    Returns LFCC feature matrix of shape (n_lfcc, T_frames).
    Uses torchaudio.transforms.LFCC (available from torchaudio >= 0.12).
    """
    lfcc_transform = T.LFCC(
        sample_rate=cfg.feature.sample_rate,
        n_filter=128,
        f_min=cfg.feature.f_min,
        f_max=cfg.feature.f_max,
        n_lfcc=cfg.feature.n_lfcc,
        speckwargs={
            "n_fft": cfg.feature.n_fft,
            "win_length": cfg.feature.win_length,
            "hop_length": cfg.feature.hop_length,
        },
    )
    return lfcc_transform(wav.unsqueeze(0)).squeeze(0)      # (n_lfcc, T)


# ---------------------------------------------------------------------------
# Log-Mel spectrogram
# ---------------------------------------------------------------------------

def compute_log_mel(wav: Tensor) -> Tensor:
    """Returns log-mel spectrogram of shape (n_mels, T_frames)."""
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.feature.sample_rate,
        n_fft=cfg.feature.n_fft,
        win_length=cfg.feature.win_length,
        hop_length=cfg.feature.hop_length,
        n_mels=cfg.feature.n_mels,
        f_min=cfg.feature.f_min,
        f_max=cfg.feature.f_max,
    )
    mel = mel_transform(wav.unsqueeze(0)).squeeze(0)        # (n_mels, T)
    log_mel = torch.log(mel + 1e-8)
    # Normalise per frequency bin
    mean = log_mel.mean(dim=1, keepdim=True)
    std  = log_mel.std(dim=1, keepdim=True) + 1e-8
    return (log_mel - mean) / std


# ---------------------------------------------------------------------------
# SpecAugment (training only)
# ---------------------------------------------------------------------------

def spec_augment(spec: Tensor, cfg_train) -> Tensor:
    """Apply time and frequency masking to a spectrogram tensor."""
    freq_mask = T.FrequencyMasking(freq_mask_param=cfg_train.aug_freq_mask_max)
    time_mask = T.TimeMasking(time_mask_param=cfg_train.aug_time_mask_max)
    for _ in range(cfg_train.aug_num_freq_masks):
        spec = freq_mask(spec)
    for _ in range(cfg_train.aug_num_time_masks):
        spec = time_mask(spec)
    return spec


# ---------------------------------------------------------------------------
# Additive noise augmentation
# ---------------------------------------------------------------------------

def add_noise(wav: Tensor, snr_db: float) -> Tensor:
    """Add white Gaussian noise at a given SNR (dB)."""
    signal_power = wav.pow(2).mean()
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(wav) * noise_power.sqrt()
    return wav + noise


# ---------------------------------------------------------------------------
# Full pre-processing pipeline
# ---------------------------------------------------------------------------

def preprocess(path: str, augment: bool = False) -> dict:
    """
    Load an audio file and return a dict with:
      - 'raw'    : (T_raw,)      raw waveform
      - 'lfcc'   : (n_lfcc, T)   LFCC features
      - 'log_mel': (n_mels, T)   log-mel spectrogram
    """
    wav = _load_audio(path)
    wav = _pad_or_trim(wav, cfg.feature.raw_chunk_length)

    if augment and cfg.train.use_augmentation:
        import random
        snr = random.choice(cfg.train.aug_noise_snr_db)
        if random.random() < 0.5:
            wav = add_noise(wav, snr)

    lfcc    = compute_lfcc(wav)
    log_mel = compute_log_mel(wav)

    if augment:
        lfcc    = spec_augment(lfcc, cfg.train)
        log_mel = spec_augment(log_mel, cfg.train)

    return {
        "raw":     wav,
        "lfcc":    lfcc,
        "log_mel": log_mel,
    }
