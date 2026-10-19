"""
Inference wrapper — loads the ONNX model once at startup and exposes
a simple `analyze(audio_bytes, content_type)` function.

Falls back to a mock predictor if no model.onnx is found (for UI demo).
"""

import io
import logging
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# Lazy import — only needed when model exists
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (must match model/config.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 16000
CHUNK_SAMPLES    = 64600        # ~4 s
LFCC_N           = 60
HOP_LENGTH       = 160
WIN_LENGTH       = 400
N_FFT            = 512


# ---------------------------------------------------------------------------
# LFCC on numpy (mirrors model/features.py but without torch dependencies)
# ---------------------------------------------------------------------------

def _compute_lfcc_np(wav: np.ndarray) -> np.ndarray:
    """Compute LFCC features using torchaudio on a numpy waveform."""
    import torchaudio.transforms as T
    wav_t = torch.tensor(wav, dtype=torch.float32)
    transform = T.LFCC(
        sample_rate=SAMPLE_RATE,
        n_filter=128,
        f_min=20.0,
        f_max=7600.0,
        n_lfcc=LFCC_N,
        speckwargs={"n_fft": N_FFT, "win_length": WIN_LENGTH, "hop_length": HOP_LENGTH},
    )
    lfcc = transform(wav_t.unsqueeze(0)).squeeze(0).numpy()   # (60, T)
    return lfcc


def _pad_or_trim(wav: np.ndarray, length: int) -> np.ndarray:
    if len(wav) >= length:
        start = (len(wav) - length) // 2
        return wav[start: start + length]
    repeats = length // len(wav) + 1
    return np.tile(wav, repeats)[:length]


def _load_audio_bytes(audio_bytes: bytes, content_type: str) -> Tuple[np.ndarray, int]:
    """Decode audio bytes to mono float32 numpy array."""
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.mean(dim=0).numpy()           # mono
    return wav, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Session (ONNX or mock)
# ---------------------------------------------------------------------------

class _MockSession:
    """Returns random-ish scores when no model.onnx is available."""
    def run(self, _, inputs):
        batch = inputs["raw"].shape[0]
        logits = np.random.randn(batch, 2).astype(np.float32)
        return [logits]


def _load_session(model_path: str):
    if not os.path.exists(model_path):
        log.warning(f"model.onnx not found at {model_path}. Using MOCK predictor.")
        return _MockSession()
    if not ONNX_AVAILABLE:
        log.warning("onnxruntime not installed. Using MOCK predictor.")
        return _MockSession()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    log.info(f"Loaded ONNX model: {model_path}")
    return sess


_SESSION = None


def get_session(model_path: str = None):
    global _SESSION
    if _SESSION is None:
        path = model_path or os.environ.get("MODEL_PATH", "app/model.onnx")
        _SESSION = _load_session(path)
    return _SESSION


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _predict_chunk(session, raw_chunk: np.ndarray, lfcc_chunk: np.ndarray) -> float:
    """Return P(fake) for a single ~4s chunk."""
    raw_in  = raw_chunk[np.newaxis, :].astype(np.float32)   # (1, T)
    lfcc_in = lfcc_chunk[np.newaxis, :, :].astype(np.float32)  # (1, 60, T')
    logits  = session.run(None, {"raw": raw_in, "lfcc": lfcc_in})[0]
    probs   = _softmax(logits)
    return float(probs[0, 1])


def _risk_level(confidence: float) -> str:
    if confidence < 0.4:
        return "SAFE"
    if confidence < 0.7:
        return "SUSPICIOUS"
    return "HIGH"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> dict:
    """
    Main entry point: analyse uploaded audio bytes.
    Returns a dict matching AnalysisResponse schema.
    """
    t0 = time.time()
    session = get_session()

    wav, sr = _load_audio_bytes(audio_bytes, content_type)
    duration_sec = len(wav) / sr

    # Segment audio into CHUNK_SAMPLES windows with 50% overlap
    hop         = CHUNK_SAMPLES // 2
    segments    = []
    for start in range(0, max(len(wav), CHUNK_SAMPLES), hop):
        chunk = _pad_or_trim(wav[start:], CHUNK_SAMPLES)
        segments.append(chunk)
        if start + CHUNK_SAMPLES >= len(wav):
            break

    segment_scores = []
    for chunk in segments:
        lfcc = _compute_lfcc_np(chunk)
        score = _predict_chunk(session, chunk, lfcc)
        segment_scores.append(round(score, 4))

    # Aggregate: max-pool over segments (conservative — flag if any chunk suspicious)
    confidence = float(np.percentile(segment_scores, 75))   # 75th percentile
    is_fake    = confidence >= 0.5

    return {
        "is_fake":            is_fake,
        "confidence":         round(confidence, 4),
        "risk_level":         _risk_level(confidence),
        "segment_scores":     segment_scores,
        "processing_time_ms": round((time.time() - t0) * 1000, 1),
        "sample_rate":        sr,
        "duration_sec":       round(duration_sec, 2),
    }


def analyze_chunk(raw_bytes: bytes, chunk_id: int = 0) -> dict:
    """
    Analyse a single raw PCM chunk (16 kHz, 16-bit, mono).
    Used by the WebSocket live streaming endpoint.
    """
    session = get_session()

    wav = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = _pad_or_trim(wav, CHUNK_SAMPLES)
    lfcc = _compute_lfcc_np(wav)

    confidence = _predict_chunk(session, wav, lfcc)
    is_fake    = confidence >= 0.5

    return {
        "chunk_id":   chunk_id,
        "is_fake":    is_fake,
        "confidence": round(confidence, 4),
        "risk_level": _risk_level(confidence),
    }
