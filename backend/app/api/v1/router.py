"""API v1 central router aggregating all domain endpoints."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, predict, model_info, insights, predictions

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(predict.router, tags=["Prediction"])
api_router.include_router(model_info.router, tags=["Model Info"])
api_router.include_router(insights.router, tags=["Analytics & Insights"])
api_router.include_router(predictions.router, tags=["Prediction History"])
