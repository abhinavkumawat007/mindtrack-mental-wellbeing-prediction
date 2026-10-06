"""Pydantic Schemas for Prediction Request and Response."""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class StudentData(BaseModel):
    """Input payload representing a user's habits and demographic factors."""
    age: int = Field(..., ge=10, le=100, description="Age between 10 and 100")
    gender: Literal['Male', 'Female'] = Field(..., description="Gender identity")
    country: str = Field(..., min_length=1, max_length=100, description="Country of residence")
    academic_level: Literal['Undergraduate', 'Graduate', 'High School'] = Field(
        ..., description="Current educational stage"
    )
    most_used_platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
        'Twitter', 'YouTube', 'TikTok', 'LINE',
        'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ] = Field(..., description="Primary social media platform")
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News'] = Field(
        ..., description="Primary purpose of social media activity"
    )
    avg_daily_usage_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily screen/social media time in hours"
    )
    daily_unlocks: int = Field(
        ..., ge=0, le=1000, description="Number of phone unlocks per day"
    )
    study_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Daily focused study or academic hours"
    )
    physical_activity_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Daily exercise or movement in hours"
    )
    sleep_hours_per_night: float = Field(
        ..., ge=0.0, le=24.0, description="Nightly sleep duration in hours"
    )
    stress_level: Literal['Low', 'Medium', 'High', 'Very High'] = Field(
        ..., description="Subjective perceived stress level"
    )


class ContributingFactor(BaseModel):
    """Explainability factor with directional influence on the prediction."""
    factor: str
    impact: Literal['positive', 'negative', 'neutral']
    description: str
    weight_score: float = Field(..., description="Relative estimated score contribution")


class PredictionResponse(BaseModel):
    """Prediction outcome including score, interpretation, and explainability."""
    predicted_mental_health_score: float = Field(..., description="Score on a 0–10 scale")
    wellbeing_tier: str = Field(..., description="Category: High Wellbeing, Moderate, etc.")
    tier_color: str = Field(..., description="Color token for UI rendering")
    interpretation: str = Field(..., description="Contextual guidance based on predicted score")
    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list,
        description="Top lifestyle drivers that influenced this prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable lifestyle recommendations based on habit inputs"
    )
    model_used: str = Field(..., description="Name of the ML model that generated the prediction")
    prediction_id: Optional[str] = Field(None, description="Unique identifier stored in history")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of prediction")
    disclaimer: str = Field(..., description="Ethical/educational disclaimer")
