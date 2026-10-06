"""MindTrack Prediction Service.

Manages pipeline loading, thread-safe inference, score categorization,
explainability attachment, and history persistence.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import joblib

from backend.app.core.config import settings
from backend.app.schemas.prediction import StudentData, PredictionResponse
from backend.app.services.explainability import explain_prediction
from backend.app.services.storage import storage_service

# Ensure ml/src and backend/app/models are in sys.path so joblib can resolve FeatureEngineer
sys.path.insert(0, str(settings.PROJECT_ROOT / "ml" / "src"))
sys.path.insert(0, str(settings.APP_DIR / "models"))

class PredictorService:
    def __init__(self):
        self.pipeline = None
        self.metadata = {}
        self.model_name = "Extra Trees Regressor"
        self._load_pipeline()

    def _load_pipeline(self):
        """Loads the serialized scikit-learn pipeline and metadata."""
        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model pipeline not found at {settings.MODEL_PATH}. Please train the model first.")

        print(f"Loading MindTrack pipeline from {settings.MODEL_PATH}...")
        self.pipeline = joblib.load(settings.MODEL_PATH)

        if settings.METADATA_PATH.exists():
            with open(settings.METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
                self.model_name = self.metadata.get("model_name", "Extra Trees")
        print("MindTrack pipeline loaded successfully.")

    def _categorize_score(self, score: float) -> Tuple[str, str, str]:
        """Categorizes predicted score into empathetic, non-diagnostic wellbeing tiers."""
        if score >= 8.0:
            return (
                "Thriving / High Wellbeing",
                "#10b981",  # Emerald
                "Your lifestyle habits show strong balance with protective sleep, healthy screen-to-rest boundaries, and resilient routines."
            )
        elif score >= 6.5:
            return (
                "Balanced / Moderate Wellbeing",
                "#06b6d4",  # Cyan
                "Your lifestyle indicators are generally healthy with minor areas where digital boundaries or recovery sleep could offer noticeable boosts."
            )
        elif score >= 5.0:
            return (
                "Elevated Strain / Fatigue Signals",
                "#f59e0b",  # Amber
                "High screen time, phone checking, or moderate sleep deficits may be exerting noticeable downward pressure on your daily energy and focus."
            )
        else:
            return (
                "High Stress / Needs Self-Care Attention",
                "#ef4444",  # Coral / Red
                "Multiple high-strain indicators (elevated screen use, sleep deficit, high stress) are converging. Consider stepping back, resting, and prioritizing recovery."
            )

    def _generate_recommendations(self, data: StudentData, score: float) -> list:
        """Generates evidence-informed lifestyle recommendations tailored to user habits."""
        recs = []
        if data.sleep_hours_per_night < 7.0:
            recs.append(f"Target 7.5–8.5 hours of sleep nightly; increasing sleep from {data.sleep_hours_per_night:.1f}h acts as a strong cognitive recovery buffer.")
        elif data.sleep_hours_per_night >= 7.5:
            recs.append("Maintain your current restorative sleep routine—consistent recovery is your primary resilience buffer.")

        if data.avg_daily_usage_hours > 5.5:
            recs.append(f"Introduce digital boundaries (app timers, screen-free meals) to moderate daily exposure ({data.avg_daily_usage_hours:.1f}h).")

        if data.daily_unlocks > 170:
            recs.append(f"Batch device notifications into designated check-in intervals to curb attentional fragmentation ({data.daily_unlocks} unlocks/day).")

        if data.physical_activity_hours < 1.0:
            recs.append("Incorporate 30–45 minutes of daily physical movement or walking to support natural mood regulation.")

        if data.stress_level in ['High', 'Very High']:
            recs.append("Build structured restorative pauses into study sessions and consider connecting with campus wellbeing counseling.")
        elif not recs:
            recs.append("Your lifestyle indicators reflect strong overall balance. Continue prioritizing sleep consistency and intentional screen habits.")

        return recs[:4]

    def predict(self, data: StudentData) -> PredictionResponse:
        """Executes full inference workflow:

        Input Data -> DataFrame -> Trained Pipeline -> Categorization -> Explainability -> Storage
        """
        from datetime import datetime, timezone

        if self.pipeline is None:
            self._load_pipeline()

        # Format input into a single-row DataFrame matching training column names
        row_dict = {
            'Age': data.age,
            'Gender': data.gender,
            'Country': data.country,
            'Academic_Level': data.academic_level,
            'Most_Used_Platform': data.most_used_platform,
            'Purpose_Of_Use': data.purpose_of_use,
            'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
            'Daily_Unlocks': data.daily_unlocks,
            'Study_Hours': data.study_hours,
            'Physical_Activity_Hours': data.physical_activity_hours,
            'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
            'Stress_Level': data.stress_level
        }
        df_input = pd.DataFrame([row_dict])

        # Generate prediction
        raw_pred = float(self.pipeline.predict(df_input)[0])
        score = round(max(1.0, min(10.0, raw_pred)), 2)

        tier, color, interpretation = self._categorize_score(score)

        # Generate local explainability factors and recommendations
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        contributing_factors = explain_prediction(input_dict, score)
        recommendations = self._generate_recommendations(data, score)

        # Persist prediction to history
        rec_id = storage_service.save_prediction(
            input_data=input_dict,
            predicted_score=score,
            wellbeing_tier=tier,
            tier_color=color,
            model_used=self.model_name
        )

        now_utc = datetime.now(timezone.utc).isoformat()

        return PredictionResponse(
            predicted_mental_health_score=score,
            wellbeing_tier=tier,
            tier_color=color,
            interpretation=interpretation,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            model_used=self.model_name,
            prediction_id=rec_id,
            timestamp=now_utc,
            disclaimer=settings.DISCLAIMER
        )

predictor_service = PredictorService()
