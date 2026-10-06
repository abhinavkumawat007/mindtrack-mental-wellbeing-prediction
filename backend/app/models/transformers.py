"""MindTrack Feature Engineering & Preprocessing Transformers.

Reusable across training, evaluation, and production FastAPI serving.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TOP_COUNTRIES = [
    'Other', 'India', 'USA', 'Canada', 'Australia',
    'UK', 'Germany', 'Mexico', 'Turkey', 'France'
]

STRESS_LEVEL_ORDER = ['Low', 'Medium', 'High', 'Very High']

CATEGORICAL_COLS = [
    'Gender', 'Academic_Level', 'Most_Used_Platform',
    'Purpose_Of_Use', 'Grouped_Country'
]

ORDINAL_COLS = ['Stress_Level']

NUMERIC_BASE_COLS = [
    'Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
    'Study_Hours', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night'
]

ENGINEERED_COLS = [
    'Screen_to_Sleep_Ratio',
    'Active_to_Sedentary_Ratio',
    'Unlocks_Per_Usage_Hour'
]

ALL_NUMERIC_COLS = NUMERIC_BASE_COLS + ENGINEERED_COLS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer that implements domain-specific feature engineering,

    outlier clipping, and categorical standardization.
    """

    def __init__(self, top_countries: List[str] = None):
        self.top_countries = top_countries or TOP_COUNTRIES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a clean DataFrame copy
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # If array or dict
            df = pd.DataFrame(X)

        # 1. Physical activity domain clipping (cannot be negative)
        if 'Physical_Activity_Hours' in df.columns:
            df['Physical_Activity_Hours'] = pd.to_numeric(df['Physical_Activity_Hours'], errors='coerce').clip(lower=0.0)

        # 2. Grouped Country logic
        if 'Country' in df.columns:
            df['Grouped_Country'] = df['Country'].apply(
                lambda c: c if str(c).strip() in self.top_countries else 'Other'
            )
        elif 'Grouped_Country' not in df.columns:
            df['Grouped_Country'] = 'Other'

        # 3. Domain Engineered Features
        usage = pd.to_numeric(df['Avg_Daily_Usage_Hours'], errors='coerce') if 'Avg_Daily_Usage_Hours' in df.columns else 0.0
        sleep = pd.to_numeric(df['Sleep_Hours_Per_Night'], errors='coerce') if 'Sleep_Hours_Per_Night' in df.columns else 7.0
        study = pd.to_numeric(df['Study_Hours'], errors='coerce') if 'Study_Hours' in df.columns else 4.0
        activity = pd.to_numeric(df['Physical_Activity_Hours'], errors='coerce') if 'Physical_Activity_Hours' in df.columns else 1.0
        unlocks = pd.to_numeric(df['Daily_Unlocks'], errors='coerce') if 'Daily_Unlocks' in df.columns else 100.0

        # Screen-to-Sleep Ratio: high ratio indicates digital overload competing with recovery
        df['Screen_to_Sleep_Ratio'] = np.round(usage / (sleep + 0.1), 4)

        # Active-to-Sedentary Ratio: (study + physical) vs screen time
        df['Active_to_Sedentary_Ratio'] = np.round((study + activity) / (usage + 0.5), 4)

        # Unlocks per usage hour: frequency/fragmentation of phone checking
        df['Unlocks_Per_Usage_Hour'] = np.round(unlocks / (usage + 0.2), 2)

        return df

    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation for Scikit-learn Pipeline compliance."""
        if input_features is None:
            return None
        names = list(input_features)
        if 'Country' in names and 'Grouped_Country' not in names:
            names.append('Grouped_Country')
        for col in ENGINEERED_COLS:
            if col not in names:
                names.append(col)
        return np.array(names, dtype=object)

