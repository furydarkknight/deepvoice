"""
Pydantic schemas for request/response validation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class RiskLevel(str, Enum):
    SAFE       = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH       = "HIGH"


class AnalysisResponse(BaseModel):
    is_fake:            bool
    confidence:         float           # P(fake), 0-1
    risk_level:         RiskLevel
    segment_scores:     List[float]     # per-chunk fake probability
    processing_time_ms: float
    sample_rate:        int
    duration_sec:       float


class LiveChunkResponse(BaseModel):
    chunk_id:   int
    is_fake:    bool
    confidence: float
    risk_level: RiskLevel


class HistoryEntry(BaseModel):
    id:             str
    timestamp:      str
    filename:       str
    is_fake:        bool
    confidence:     float
    risk_level:     RiskLevel
    duration_sec:   float
