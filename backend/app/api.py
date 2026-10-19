"""
REST API endpoints for the DeepFake Voice Detector.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.inference import analyze_audio
from app.schemas import AnalysisResponse, HistoryEntry, RiskLevel

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# In-memory call history (replace with a real DB in production)
# ---------------------------------------------------------------------------
_history: List[HistoryEntry] = []

ALLOWED_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg",  "audio/webm",
    "audio/flac", "audio/x-flac",
    "application/octet-stream",      # fallback for some clients
}

MAX_FILE_BYTES = 60 * 1024 * 1024   # 60 MB


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(audio: UploadFile = File(...)):
    """
    Upload an audio file (WAV / MP3 / OGG / FLAC) for deepfake analysis.
    Returns prediction with confidence, risk level, and per-segment scores.
    """
    if audio.content_type not in ALLOWED_TYPES:
        # Lenient: allow if extension is audio
        ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in {"wav", "mp3", "ogg", "flac", "webm", "m4a"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type: {audio.content_type}",
            )

    raw = await audio.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 60 MB)")
    if len(raw) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too short")

    result = analyze_audio(raw, audio.content_type or "audio/wav")

    # Log to history
    entry = HistoryEntry(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        filename=audio.filename or "unknown",
        is_fake=result["is_fake"],
        confidence=result["confidence"],
        risk_level=RiskLevel(result["risk_level"]),
        duration_sec=result["duration_sec"],
    )
    _history.insert(0, entry)
    if len(_history) > 500:
        _history.pop()

    return AnalysisResponse(**result)


@router.get("/history", response_model=List[HistoryEntry])
async def get_history(limit: int = 50):
    """Return the last N analysed calls."""
    return _history[:limit]


@router.delete("/history")
async def clear_history():
    """Clear the call history."""
    _history.clear()
    return {"status": "ok"}


@router.get("/health")
async def health():
    return {"status": "ok", "model": "AASIST-lite ONNX"}
