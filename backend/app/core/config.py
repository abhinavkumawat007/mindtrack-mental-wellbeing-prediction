"""MindTrack Application Configuration."""

import os
from pathlib import Path
from typing import List

# Base paths
APP_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

class Settings:
    PROJECT_NAME: str = "MindTrack — Mental Health Score Prediction & Insights"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    APP_DIR: Path = APP_DIR
    BACKEND_DIR: Path = BACKEND_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT
    
    # Model artifact paths
    MODEL_PATH: Path = APP_DIR / "models" / "mindtrack_pipeline.joblib"
    METADATA_PATH: Path = APP_DIR / "models" / "model_metadata.json"
    COMPARISON_PATH: Path = APP_DIR / "models" / "model_comparison.json"
    
    # Fallback to ml/artifacts if not in app/models
    if not MODEL_PATH.exists():
        ALT_MODEL = PROJECT_ROOT / "ml" / "artifacts" / "mindtrack_pipeline.joblib"
        if ALT_MODEL.exists():
            MODEL_PATH = ALT_MODEL
    if not METADATA_PATH.exists():
        ALT_META = PROJECT_ROOT / "ml" / "artifacts" / "model_metadata.json"
        if ALT_META.exists():
            METADATA_PATH = ALT_META

    # Database Settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "mindtrack_db")
    SQLITE_DB_PATH: Path = BACKEND_DIR / "data" / "predictions.db"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Responsible Use Notice
    DISCLAIMER: str = (
        "MindTrack is an educational and informational tool powered by machine learning. "
        "It is NOT a medical diagnostic system or a clinical assessment tool. "
        "Scores represent statistical estimates based on self-reported lifestyle habits. "
        "If you are experiencing severe distress, please consult a qualified healthcare provider."
    )

settings = Settings()
