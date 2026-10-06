"""MindTrack — Mental Health Score Prediction & Insights API.

Production FastAPI Application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.app.core.config import settings
from backend.app.api.v1.router import api_router
from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.predict import router as predict_router
from backend.app.api.v1.endpoints.model_info import router as model_info_router
from backend.app.api.v1.endpoints.predictions import router as predictions_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "MindTrack is an end-to-end Machine Learning web service that predicts an individual's "
        "wellbeing score based on digital habits, sleep, movement, and stress patterns.\n\n"
        f"**Responsible Use Notice:** {settings.DISCLAIMER}"
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root level aliases for direct backward compatibility & convenience
app.include_router(health_router, prefix="")
app.include_router(predict_router, prefix="")
app.include_router(model_info_router, prefix="")
app.include_router(predictions_router, prefix="")

# API v1 prefix routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
        "health_check": "/health",
        "disclaimer": settings.DISCLAIMER
    }

# Mount static frontend if available
frontend_dir = settings.PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
