"""Pydantic Schemas for Prediction History."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class PredictionHistoryRecord(BaseModel):
    id: str
    timestamp: str
    input_data: Dict[str, Any]
    predicted_mental_health_score: float
    wellbeing_tier: str
    tier_color: str
    model_used: str


class HistoryListResponse(BaseModel):
    total: int
    storage_type: str
    records: List[PredictionHistoryRecord]
