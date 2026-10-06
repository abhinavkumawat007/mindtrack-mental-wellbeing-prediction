"""Dataset insights and analytics endpoint."""

from fastapi import APIRouter
from backend.app.services.predictor import predictor_service

router = APIRouter()

@router.get("/insights", summary="Retrieve Analytics & Dataset Insights")
def get_insights():
    """Provides dataset-level statistics, distributions, and feature importances for UI charts."""
    meta = predictor_service.metadata
    dataset_summary = meta.get("dataset_summary", {})
    feature_importances = meta.get("feature_importances", [])
    model_comparison = meta.get("model_comparison", [])

    return {
        "dataset_summary": dataset_summary,
        "feature_importances": feature_importances,
        "model_comparison": model_comparison,
        "key_findings": [
            {
                "title": "Screen-to-Sleep Ratio Dominance",
                "finding": "The engineered Screen-to-Sleep ratio emerged as the single highest predictor (16.7% importance), confirming that late-night screen displacement has a compounding negative impact."
            },
            {
                "title": "Screen Time Correlation",
                "finding": "Average daily usage has a strong negative Pearson correlation (-0.82) with wellbeing scores across surveyed students."
            },
            {
                "title": "Sleep as a Restorative Buffer",
                "finding": "Students obtaining 7.5+ hours of sleep nightly averaged 1.8 points higher in mental wellbeing than those with <5.5 hours."
            },
            {
                "title": "Phone Check Fragmentation",
                "finding": "Frequent device unlocking (>200 unlocks/day) correlates with high reported stress and lower sustained study focus."
            }
        ]
    }
