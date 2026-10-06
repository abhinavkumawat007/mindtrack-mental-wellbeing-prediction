"""Prediction endpoint."""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.prediction import StudentData, PredictionResponse
from backend.app.services.predictor import predictor_service

router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Mental Health Score & Influential Factors"
)
def predict_mental_health_score(payload: StudentData):
    """Generates an estimated mental health score (0–10) with local explainability factors

    based on self-reported demographic, academic, and digital lifestyle habits.
    """
    try:
        response = predictor_service.predict(payload)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline encountered an error: {str(e)}"
        )
