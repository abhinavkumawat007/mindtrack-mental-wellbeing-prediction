"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime, timezone
from backend.app.core.config import settings
from backend.app.services.predictor import predictor_service
from backend.app.services.storage import storage_service

router = APIRouter()

@router.get("/health", summary="Service Health & Diagnostic Status")
def health_check():
    """Returns API health status, loaded model name, storage backend, and version."""
    model_loaded = predictor_service.pipeline is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": model_loaded,
        "model_name": predictor_service.model_name,
        "storage_backend": storage_service.storage_type,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
