"""Pydantic Schemas for Model Information and Diagnostics."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_rmse_mean: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelComparisonItem(BaseModel):
    model_name: str
    cv_mae: float
    cv_rmse: float
    cv_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    selected: bool


class ModelInfoResponse(BaseModel):
    project_name: str
    version: str
    trained_at: str
    model_name: str
    selection_rationale: str
    metrics: ModelMetrics
    train_samples: int
    test_samples: int
    total_samples: int
    feature_importances: List[FeatureImportanceItem]
    model_comparison: List[ModelComparisonItem]
    dataset_summary: Dict[str, Any]
