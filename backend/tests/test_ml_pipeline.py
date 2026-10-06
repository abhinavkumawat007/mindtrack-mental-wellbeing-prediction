"""Machine Learning Pipeline, Preprocessing, and Robustness Tests."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Register 'transformers' in sys.modules so joblib can unpickle the
# pipeline's custom FeatureEngineer step which was serialized under module 'transformers'
from ml.src import transformers
sys.modules.setdefault("transformers", transformers)

from ml.src.transformers import (
    FeatureEngineer,
    TOP_COUNTRIES,
    STRESS_LEVEL_ORDER,
    ALL_NUMERIC_COLS,
    CATEGORICAL_COLS
)


@pytest.fixture(scope="module")
def pipeline():
    model_path = PROJECT_ROOT / "ml" / "artifacts" / "mindtrack_pipeline.joblib"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "backend" / "app" / "models" / "mindtrack_pipeline.joblib"
    assert model_path.exists(), f"Trained model artifact not found at {model_path}"
    pipe = joblib.load(model_path)
    return pipe


@pytest.fixture
def sample_input_df():
    return pd.DataFrame([{
        "Age": 21,
        "Gender": "Female",
        "Country": "Canada",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Entertainment",
        "Avg_Daily_Usage_Hours": 3.5,
        "Daily_Unlocks": 110,
        "Study_Hours": 4.5,
        "Physical_Activity_Hours": 2.0,
        "Sleep_Hours_Per_Night": 7.8,
        "Stress_Level": "Medium"
    }])


def test_model_artifact_loading(pipeline):
    """Test that the trained pipeline loads as a valid Scikit-Learn Pipeline."""
    assert isinstance(pipeline, Pipeline)
    assert "engineer" in pipeline.named_steps
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps


def test_feature_engineering_transformer(sample_input_df):
    """Test domain-specific feature engineering transformer logic."""
    fe = FeatureEngineer()
    df_out = fe.transform(sample_input_df)

    assert "Screen_to_Sleep_Ratio" in df_out.columns
    assert "Active_to_Sedentary_Ratio" in df_out.columns
    assert "Unlocks_Per_Usage_Hour" in df_out.columns
    assert "Grouped_Country" in df_out.columns

    # Verify Screen-to-Sleep calculation: 3.5 / (7.8 + 0.1) = 0.443
    expected_ratio = round(3.5 / 7.9, 4)
    assert abs(df_out["Screen_to_Sleep_Ratio"].iloc[0] - expected_ratio) < 1e-3

    # Country mapping check
    assert df_out["Grouped_Country"].iloc[0] == "Canada"


def test_feature_engineer_rare_country_grouping():
    """Test that rare/unseen countries are mapped to 'Other' without failing."""
    df_rare = pd.DataFrame([{
        "Age": 22,
        "Gender": "Male",
        "Country": "Iceland",  # Not in top 10
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Entertainment",
        "Avg_Daily_Usage_Hours": 4.0,
        "Daily_Unlocks": 100,
        "Study_Hours": 3.0,
        "Physical_Activity_Hours": 1.0,
        "Sleep_Hours_Per_Night": 7.0,
        "Stress_Level": "Low"
    }])
    fe = FeatureEngineer()
    df_out = fe.transform(df_rare)
    assert df_out["Grouped_Country"].iloc[0] == "Other"


def test_feature_engineer_negative_activity_clipping():
    """Test that negative physical activity is clipped to 0.0."""
    df_invalid = pd.DataFrame([{
        "Physical_Activity_Hours": -3.5,
        "Avg_Daily_Usage_Hours": 4.0,
        "Sleep_Hours_Per_Night": 7.0,
        "Study_Hours": 3.0,
        "Daily_Unlocks": 100
    }])
    fe = FeatureEngineer()
    df_out = fe.transform(df_invalid)
    assert df_out["Physical_Activity_Hours"].iloc[0] == 0.0


def test_pipeline_prediction_numeric_and_bounded(pipeline, sample_input_df):
    """Test that pipeline prediction is a float within the 0-10 score range."""
    pred = pipeline.predict(sample_input_df)
    assert len(pred) == 1
    score = float(pred[0])
    assert not np.isnan(score)
    assert 1.0 <= score <= 10.0


def test_pipeline_handles_missing_values(pipeline, sample_input_df):
    """Test that pipeline handles missing values using learned median/mode imputers without crashing."""
    df_with_na = sample_input_df.copy()
    df_with_na.loc[0, "Avg_Daily_Usage_Hours"] = np.nan
    df_with_na.loc[0, "Study_Hours"] = np.nan
    df_with_na.loc[0, "Stress_Level"] = np.nan

    pred = pipeline.predict(df_with_na)
    assert len(pred) == 1
    score = float(pred[0])
    assert not np.isnan(score)
    assert 1.0 <= score <= 10.0


def test_directional_prediction_validity(pipeline):
    """Test that a healthy profile predicts a significantly higher wellbeing score than a high-strain profile."""
    healthy_profile = pd.DataFrame([{
        "Age": 21,
        "Gender": "Female",
        "Country": "Canada",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Education",
        "Avg_Daily_Usage_Hours": 2.0,
        "Daily_Unlocks": 60,
        "Study_Hours": 5.0,
        "Physical_Activity_Hours": 2.5,
        "Sleep_Hours_Per_Night": 8.5,
        "Stress_Level": "Low"
    }])

    strained_profile = pd.DataFrame([{
        "Age": 21,
        "Gender": "Female",
        "Country": "Canada",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Entertainment",
        "Avg_Daily_Usage_Hours": 9.0,
        "Daily_Unlocks": 280,
        "Study_Hours": 1.0,
        "Physical_Activity_Hours": 0.2,
        "Sleep_Hours_Per_Night": 4.5,
        "Stress_Level": "Very High"
    }])

    score_healthy = float(pipeline.predict(healthy_profile)[0])
    score_strained = float(pipeline.predict(strained_profile)[0])

    assert score_healthy > score_strained
    assert (score_healthy - score_strained) >= 2.0
