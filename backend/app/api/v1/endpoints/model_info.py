"""Model information and diagnostics endpoint."""

from fastapi import APIRouter, HTTPException
from backend.app.core.config import settings
from backend.app.schemas.model_info import ModelInfoResponse
from backend.app.services.predictor import predictor_service

router = APIRouter()

@router.get("/model-info", response_model=ModelInfoResponse, summary="Retrieve Trained Model Diagnostics & Metrics")
def get_model_info():
    """Returns champion model metadata, evaluation metrics, feature importances,

    and cross-validation comparison results.
    """
    meta = predictor_service.metadata
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    return ModelInfoResponse(
        project_name=meta.get("project_name", settings.PROJECT_NAME),
        version=settings.VERSION,
        trained_at=meta.get("trained_at", ""),
        model_name=meta.get("model_name", "Extra Trees"),
        selection_rationale=meta.get("selection_rationale", ""),
        metrics=meta.get("metrics", {}),
        train_samples=meta.get("train_samples", 0),
        test_samples=meta.get("test_samples", 0),
        total_samples=meta.get("total_samples", 0),
        feature_importances=meta.get("feature_importances", []),
        model_comparison=meta.get("model_comparison", []),
        dataset_summary=meta.get("dataset_summary", {})
    )
