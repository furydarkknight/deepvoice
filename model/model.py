"""
AASIST-lite model: a lightweight re-implementation of
"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"
(Jung et al., 2022) — adapted for CPU-friendly inference and ONNX export.

Architecture overview
─────────────────────
Raw waveform branch  →  SincConv  →  ResBlocks  →  GAT pooling
LFCC branch          →  Conv2D    →  ResBlocks  →  Temporal pooling
Both branches fused  →  Graph Attention  →  Classifier head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.config import cfg


# ---------------------------------------------------------------------------
# SincConv — learnable sinc-function filter bank (Ravanelli & Bengio 2018)
# ---------------------------------------------------------------------------

class SincConv(nn.Module):
    """1-D sinc convolution layer operating on raw waveform."""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000):
        super().__init__()
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate   = sample_rate

        # Initialise mel-spaced cutoff frequencies
        low_hz, high_hz = 30.0, sample_rate / 2 - 100
        mel = torch.linspace(
            self._hz_to_mel(low_hz), self._hz_to_mel(high_hz), out_channels + 1
        )
        hz = self._mel_to_hz(mel)
        self.low_hz_  = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))

        n = (self.kernel_size - 1) / 2.0
        self.register_buffer(
            "n_", (2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)
        )
        self.window_ = torch.hamming_window(self.kernel_size).unsqueeze(0)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: Tensor) -> Tensor:
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, T)
        low  = 50 + torch.abs(self.low_hz_)
        high = torch.clamp(low + torch.abs(self.band_hz_), min=50, max=self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low  = torch.matmul(low,  self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_.to(x.device)
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right  = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )
        band_pass = band_pass / (2 * band.view(-1, 1))
        filters   = band_pass.unsqueeze(1)                  # (out_ch, 1, kernel)

        return F.conv1d(x, filters, padding=self.kernel_size // 2)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Temporal Graph Attention (simplified)
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, N, F)
        h_w = self.W(h)                                     # (B, N, out_F)
        B, N, F = h_w.shape
        a_input = torch.cat(
            [h_w.unsqueeze(2).expand(-1, -1, N, -1),
             h_w.unsqueeze(1).expand(-1, N, -1, -1)],
            dim=-1
        )                                                   # (B, N, N, 2F)
        e = self.leaky(self.att(a_input).squeeze(-1))       # (B, N, N)
        alpha = F.softmax(e, dim=-1)
        return torch.bmm(alpha, h_w)                        # (B, N, out_F)


# ---------------------------------------------------------------------------
# Raw waveform encoder
# ---------------------------------------------------------------------------

class RawEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        sinc_out = cfg.model.sinc_out_channels
        self.sinc = SincConv(sinc_out, cfg.model.sinc_kernel_size)
        self.bn   = nn.BatchNorm1d(sinc_out)
        self.pool = nn.MaxPool1d(3)

        self.res_blocks = nn.Sequential(
            self._make_stage(sinc_out, 128, downsample=4),
            self._make_stage(128,      128, downsample=4),
            self._make_stage(128,      out_dim, downsample=4),
        )

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, downsample: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            ResBlock1D(out_ch),
            nn.MaxPool1d(downsample),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T)
        x = x.unsqueeze(1)                      # (B, 1, T)
        x = F.gelu(self.bn(self.sinc(x)))
        x = self.pool(x)
        x = self.res_blocks(x)                  # (B, out_dim, T')
        return x.permute(0, 2, 1)               # (B, T', out_dim)  — node features


# ---------------------------------------------------------------------------
# LFCC / spectral encoder
# ---------------------------------------------------------------------------

class SpectralEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResBlock2D(32),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResBlock2D(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))         # freq collapsed

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_lfcc, T)
        x = x.unsqueeze(1)                                  # (B, 1, F, T)
        x = self.front(x)                                   # (B, out_dim, F', T')
        x = self.pool(x).squeeze(2)                         # (B, out_dim, T')
        return x.permute(0, 2, 1)                           # (B, T', out_dim)


# ---------------------------------------------------------------------------
# Full AASIST-lite model
# ---------------------------------------------------------------------------

class AASISTLite(nn.Module):
    """
    Dual-branch anti-spoofing model.
    Branch 1: Raw waveform  →  SincConv encoder
    Branch 2: LFCC          →  2D-CNN encoder
    Fusion:   concatenate   →  GAT  →  MLP head
    """

    def __init__(self):
        super().__init__()
        enc_dim  = cfg.model.d_model * 2          # 128

        self.raw_encoder      = RawEncoder(out_dim=enc_dim)
        self.spectral_encoder = SpectralEncoder(out_dim=enc_dim)

        self.gat1 = GraphAttentionLayer(enc_dim * 2, enc_dim)
        self.gat2 = GraphAttentionLayer(enc_dim,     enc_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, cfg.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(cfg.model.hidden_dim, cfg.model.num_classes),
        )

    def forward(self, raw: Tensor, lfcc: Tensor) -> Tensor:
        """
        raw  : (B, T_raw)
        lfcc : (B, n_lfcc, T_frames)
        returns logits (B, 2)
        """
        raw_feat = self.raw_encoder(raw)            # (B, N1, D)
        spc_feat = self.spectral_encoder(lfcc)      # (B, N2, D)

        # Align temporal dimension by adaptive pooling
        n = min(raw_feat.shape[1], spc_feat.shape[1])
        raw_feat = raw_feat[:, :n, :]
        spc_feat = spc_feat[:, :n, :]

        x = torch.cat([raw_feat, spc_feat], dim=-1) # (B, N, 2D)
        x = F.gelu(self.gat1(x))                    # (B, N, D)
        x = F.gelu(self.gat2(x))                    # (B, N, D)

        x = x.mean(dim=1)                           # (B, D)  — graph readout
        return self.classifier(x)                   # (B, 2)

    @torch.no_grad()
    def predict_proba(self, raw: Tensor, lfcc: Tensor) -> Tensor:
        """Returns P(fake) for a batch; shape (B,)."""
        logits = self.forward(raw, lfcc)
        return F.softmax(logits, dim=-1)[:, 1]


def build_model() -> AASISTLite:
    return AASISTLite()
