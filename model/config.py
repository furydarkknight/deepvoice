"""
Central configuration for training, feature extraction, and model architecture.
All paths are relative to the project root.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FeatureConfig:
    # Audio
    sample_rate: int = 16000
    max_duration_sec: float = 4.0          # clips trimmed / padded to this
    n_mfcc: int = 40
    n_lfcc: int = 60
    n_fft: int = 512
    hop_length: int = 160                  # 10 ms @ 16 kHz
    win_length: int = 400                  # 25 ms
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 7600.0

    # Raw waveform branch
    use_raw_waveform: bool = True
    raw_chunk_length: int = 64600          # ~4 s @ 16 kHz


@dataclass
class ModelConfig:
    # AASIST-lite graph attention network
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 6
    dropout: float = 0.1

    # CNN front-end
    sinc_out_channels: int = 70
    sinc_kernel_size: int = 1024

    # Classifier head
    hidden_dim: int = 160
    num_classes: int = 2


@dataclass
class TrainConfig:
    # Data
    train_csv: str = "data/processed/train.csv"
    dev_csv: str = "data/processed/dev.csv"
    eval_csv: str = "data/processed/eval.csv"

    # Training
    epochs: int = 100
    batch_size: int = 24
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 15

    # Augmentation (SpecAugment + noise)
    use_augmentation: bool = True
    aug_noise_snr_db: List[float] = field(default_factory=lambda: [10, 20, 30])
    aug_time_mask_max: int = 30
    aug_freq_mask_max: int = 15
    aug_num_time_masks: int = 2
    aug_num_freq_masks: int = 2

    # Output
    output_dir: str = "checkpoints"
    log_interval: int = 50          # steps between log prints
    eval_interval: int = 1          # epochs between dev evaluation

    # Mixed precision
    use_amp: bool = True

    # Seed
    seed: int = 42


@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def max_len(self) -> int:
        return self.feature.raw_chunk_length


# Singleton
cfg = Config()
