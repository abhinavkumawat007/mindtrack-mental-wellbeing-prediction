"""MindTrack Model Evaluation & Visualization Suite.

Loads the trained production pipeline and held-out test data,
calculates official regression metrics (MAE, MSE, RMSE, R²),
and generates publication-quality evaluation figures.

Usage:
    python ml/src/evaluate.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure ml/src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))
from transformers import FeatureEngineer, CATEGORICAL_COLS


def evaluate_champion_model():
    print("=" * 70)
    print("🧠 MindTrack — Production Model Evaluation")
    print("=" * 70)

    data_path = PROJECT_ROOT / "data" / "Student Social Media And Mental Health Impact.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"

    model_path = PROJECT_ROOT / "ml" / "artifacts" / "mindtrack_pipeline.joblib"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "backend" / "app" / "models" / "mindtrack_pipeline.joblib"

    print(f"Loading dataset: {data_path}")
    raw_df = pd.read_csv(data_path)
    df = raw_df.drop_duplicates().copy()
    print(f"Cleaned sample size: {len(df)}")

    target_col = "Mental_Health_Score"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Exactly match 80/20 train/test split with seed 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Loading pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    # Predict on test set
    preds = pipeline.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("\n" + "-" * 45)
    print("  HELD-OUT TEST SET EVALUATION METRICS")
    print("-" * 45)
    print(f"  R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE):               {mae:.4f}")
    print(f"  Mean Squared Error (MSE):                {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE):          {rmse:.4f}")
    print("-" * 45)

    # Generate or refresh plots
    plots_dir = PROJECT_ROOT / "ml" / "artifacts" / "plots"
    frontend_plots_dir = PROJECT_ROOT / "frontend" / "assets" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frontend_plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Actual vs Predicted
        plt.figure(figsize=(6.5, 6), dpi=200)
        plt.scatter(y_test, preds, alpha=0.45, color="#06b6d4", edgecolors="none", s=28)
        min_v = min(y_test.min(), preds.min()) - 0.2
        max_v = max(y_test.max(), preds.max()) + 0.2
        plt.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect Fit (Identity)")
        plt.title(f"Actual vs. Predicted Wellbeing Scores\nTest R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}", fontsize=11, fontweight="bold")
        plt.xlabel("Ground Truth Mental Health Score", fontsize=10)
        plt.ylabel("Predicted Mental Health Score", fontsize=10)
        plt.xlim(min_v, max_v)
        plt.ylim(min_v, max_v)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "actual_vs_predicted.png")
        plt.savefig(frontend_plots_dir / "actual_vs_predicted.png")
        plt.close()

        # Plot 2: Residuals Distribution
        residuals = y_test - preds
        plt.figure(figsize=(8, 4.8), dpi=200)
        plt.hist(residuals, bins=35, color="#8b5cf6", edgecolor="white", alpha=0.85, density=True)
        plt.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero Error")
        plt.title(f"Prediction Residual Distribution (Mean = {residuals.mean():.3f}, Std = {residuals.std():.3f})", fontsize=11, fontweight="bold")
        plt.xlabel("Residual Error (Actual - Predicted)", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "residuals_distribution.png")
        plt.savefig(frontend_plots_dir / "residuals_distribution.png")
        plt.close()

        print(f"Evaluation figures refreshed in: {plots_dir}")
    except Exception as e:
        print(f"Warning generating plots: {e}")

    print("=" * 70)
    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


if __name__ == "__main__":
    evaluate_champion_model()
