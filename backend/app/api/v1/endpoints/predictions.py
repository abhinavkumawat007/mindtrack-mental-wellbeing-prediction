"""Prediction history management endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from backend.app.schemas.history import HistoryListResponse
from backend.app.services.storage import storage_service

router = APIRouter()

@router.get("/predictions", response_model=HistoryListResponse, summary="Retrieve Recent Prediction History")
def get_prediction_history(limit: int = Query(default=30, ge=1, le=100)):
    """Returns stored historical predictions with input values, scores, and timestamps."""
    records = storage_service.get_recent_predictions(limit=limit)
    return HistoryListResponse(
        total=len(records),
        storage_type=storage_service.storage_type,
        records=records
    )

@router.delete("/predictions/{prediction_id}", summary="Delete a Prediction from History")
def delete_prediction(prediction_id: str):
    """Deletes a specific prediction entry by ID."""
    success = storage_service.delete_prediction(prediction_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID '{prediction_id}' not found."
        )
    return {"status": "success", "message": f"Prediction '{prediction_id}' deleted."}

@router.delete("/predictions", summary="Clear All Prediction History")
def clear_all_predictions():
    """Removes all stored predictions."""
    count = storage_service.clear_all()
    return {"status": "success", "deleted_count": count}
