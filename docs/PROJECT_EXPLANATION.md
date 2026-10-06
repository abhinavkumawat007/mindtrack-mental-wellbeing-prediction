# 🎓 MindTrack — Project Technical Deep-Dive & Interview Guide

This document provides a concise, structured technical walkthrough of the **MindTrack** project designed for portfolio defense, engineering interviews, and technical presentations.

---

## 1. Problem Statement & Motivation

### What real-world problem does this project address?
Modern students and young professionals face unprecedented levels of digital saturation—averaging over 5 hours of social media usage and more than 170 device unlocks daily. While smartphones quantify raw screen time, raw metrics fail to convey overall lifestyle health.

Crucially, lifestyle indicators interact non-linearly:
* 5 hours of screen time with 8.5 hours of restorative sleep has a fundamentally different physiological impact than 5 hours of screen time combined with 4.5 hours of sleep deprivation.
* Frequent device unlocks (attentional fragmentation) induce cognitive strain independent of total hours spent.

### Formulation
MindTrack frames this as a **supervised continuous regression task**: mapping a 12-dimensional vector of digital habits, academic engagement, and lifestyle indicators to an interpretable **Mental Health / Wellbeing Score** ($1.0 - 10.0$).

---

## 2. Dataset Overview

* **Source & Size:** 4,998 unique student survey records (after removing 2 exact duplicates from 5,000 raw samples).
* **Target Variable:** `Mental_Health_Score` (Continuous float, range: $3.6 - 9.4$, mean: $6.23$, standard deviation: $1.28$).
* **Input Features:**
  * Numerical: `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`.
  * Categorical: `Gender`, `Country`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`.
  * Ordinal: `Stress_Level` (`Low`, `Medium`, `High`, `Very High`).

---

## 3. Data Cleaning & Domain Safeguards

1. **Deduplication:** Dropped 2 exact duplicates across all attributes.
2. **Domain Outlier Correction:** In the raw dataset, `Physical_Activity_Hours` contained negative values (e.g. $-0.4$ hours). Physical activity cannot be negative, so we applied lower-bound clipping at $0.0$.
3. **Cardinality Management:** Over 30 rare countries appeared with very low frequencies. To prevent sparse one-hot encoding columns, countries outside the top 10 were mapped to `'Other'`.

---

## 4. Feature Engineering (`FeatureEngineer`)

Domain-specific feature combinations were constructed within a custom Scikit-Learn transformer (`FeatureEngineer` inheriting from `BaseEstimator` and `TransformerMixin`):

1. **Screen-to-Sleep Ratio:**
   $$\text{Screen\_to\_Sleep\_Ratio} = \frac{\text{Avg\_Daily\_Usage\_Hours}}{\text{Sleep\_Hours\_Per\_Night} + 0.1}$$
   *Rationale:* Directly models digital consumption competing against neurological recovery. Higher ratios strongly suppress predicted wellbeing.
2. **Active-to-Sedentary Ratio:**
   $$\text{Active\_to\_Sedentary\_Ratio} = \frac{\text{Study\_Hours} + \text{Physical\_Activity\_Hours}}{\text{Avg\_Daily\_Usage\_Hours} + 0.5}$$
   *Rationale:* Measures constructive lifestyle investment (movement and focused study) relative to passive screen consumption.
3. **Unlocks Per Usage Hour:**
   $$\text{Unlocks\_Per\_Usage\_Hour} = \frac{\text{Daily\_Unlocks}}{\text{Avg\_Daily\_Usage\_Hours} + 0.2}$$
   *Rationale:* Differentiates prolonged focus (watching a 2-hour lecture) from severe attention fragmentation (checking notifications every 2 minutes).

---

## 5. Machine Learning Pipeline & Zero Data Leakage

### Pipeline Architecture
A single, self-contained `sklearn.pipeline.Pipeline` was built:
1. **Step 1 (`engineer`):** Custom `FeatureEngineer` creating the 3 interaction features and standardizing countries.
2. **Step 2 (`preprocessor`):** `ColumnTransformer` with three branches:
   * **Numerical Branch:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
   * **Ordinal Branch:** `OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']])`
   * **Nominal Categorical Branch:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
3. **Step 3 (`regressor`):** `ExtraTreesRegressor`

### Why is this leak-free?
- Imputers, scalers, and encoders calculate statistics (medians, means, standard deviations, and category sets) **strictly on training folds**.
- When evaluating with cross-validation or testing, the validation/test fold is transformed using the parameters learned during training—never seeing test data during fitting.

---

## 6. Model Selection & Benchmarking

We benchmarked 5 candidate regression algorithms using **5-Fold Cross-Validation** on an 80% training split (3,998 samples) and evaluated on a held-out 20% test split (1,000 samples):

| Model | 5-Fold CV R² | 5-Fold CV RMSE | Test Set R² | Test Set RMSE | Test Set MAE | Selection |
|---|---|---|---|---|---|---|
| **Extra Trees Regressor** | **0.8372** | **0.5100** | **0.8552** | **0.5085** | **0.3875** | **🏆 Selected** |
| Random Forest Regressor | 0.8280 | 0.5240 | 0.8441 | 0.5276 | 0.4056 | Runner-Up |
| Gradient Boosting Regressor | 0.8024 | 0.5616 | 0.8216 | 0.5643 | 0.4401 | Candidate |
| Ridge Regression (L2) | 0.7293 | 0.6571 | 0.7454 | 0.6741 | 0.5328 | Linear Baseline |
| Linear Regression | 0.7293 | 0.6572 | 0.7453 | 0.6743 | 0.5330 | Baseline |

### Why Extra Trees (Extremely Randomized Trees)?
1. **Strongest Generalization:** Outperformed Random Forest and Gradient Boosting across all cross-validation folds ($R^2 = 0.8552$, $\text{RMSE} = 0.5085$).
2. **Variance Reduction:** Unlike standard Random Forests that optimize split thresholds for each feature candidate, Extra Trees draws cut-point thresholds at random and selects the best among them, reducing model variance on tabular lifestyle data.
3. **Compact Production Footprint:** Hyperparameters (`n_estimators=120`, `max_depth=14`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features='sqrt'`) compressed the model artifact from 23 MB down to **3.1 MB**, enabling near-instantaneous cloud loading.

---

## 7. Model Explainability Engine

A core design requirement was that MindTrack must never function as an opaque black box. We implemented dual-level explainability:

### 1. Global Feature Importance
Derived from ensemble tree split gains:
* `Screen_to_Sleep_Ratio` (16.69%) and `Avg_Daily_Usage_Hours` (15.37%) are the primary negative drivers.
* `Stress_Level` (14.62%) and `Sleep_Hours_Per_Night` (11.57%) are the strongest psychological and physical modulators.

### 2. Local Factor Attribution
For each individual prediction, `backend/app/services/explainability.py` computes directional factor impacts relative to cohort baseline means ($\mu_{\text{usage}} = 5.08\text{h}$, $\mu_{\text{sleep}} = 6.63\text{h}$, etc.):
* **Positive Protective Buffers:** Highlights beneficial routines (e.g., *"Sufficient sleep of 8.0h provides an active recovery buffer"*).
* **Negative Strain Drivers:** Highlights high-risk behaviors (e.g., *"11.0h screen use is 5.9h above peer average"*).

---

## 8. FastAPI Backend Architecture

```text
backend/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── health.py        # Health probe, model status, storage backend
│   │   │   ├── predict.py       # Inference endpoint (/api/v1/predict)
│   │   │   ├── model_info.py    # Model metadata, comparison metrics, feature importance
│   │   │   └── predictions.py   # History CRUD (/api/v1/predictions)
│   │   └── router.py            # API v1 aggregator
│   ├── core/
│   │   └── config.py            # Environment settings, artifact paths, disclaimer
│   ├── models/
│   │   ├── mindtrack_pipeline.joblib  # Serialized Scikit-Learn pipeline
│   │   ├── transformers.py            # Custom FeatureEngineer class
│   │   └── model_metadata.json        # Official evaluation metrics & schema
│   ├── schemas/
│   │   └── prediction.py        # Pydantic v2 validation contracts
│   ├── services/
│   │   ├── predictor.py         # Singleton model manager, thread-safe inference
│   │   ├── explainability.py    # Local factor attribution logic
│   │   └── storage.py           # Dual SQLite / MongoDB persistence service
│   └── main.py                  # FastAPI application entrypoint & static mount
└── tests/                       # Pytest automated test suite (20 tests)
```

### Key Engineering Decisions
* **Singleton Model Lifecycle:** The 3.1 MB model pipeline is loaded once into memory during application startup (`PredictorService`), avoiding per-request deserialization latency.
* **Pydantic v2 Input Validation:** Strict bounds (`10 <= age <= 100`, `0.0 <= hours <= 24.0`, literal stress categories) prevent malformed or adversarial inputs from reaching the pipeline.
* **Graceful Fallbacks:** The pipeline features automatic imputers so missing or null inputs are handled safely without crashing.

---

## 9. Storage & Dual Persistence Strategy

1. **Default Mode (SQLite):** Out-of-the-box local execution requires zero database installation. SQLite initializes automatically at `backend/data/predictions.db` using WAL (Write-Ahead Logging) mode.
2. **Production Mode (MongoDB):** If the `MONGODB_URI` environment variable is configured, `StorageService` connects to MongoDB and persists predictions with automatic fallback to SQLite if MongoDB becomes unreachable.

---

## 10. Frontend Dashboard

* **Technology:** Vanilla HTML5, modern CSS3 (Glassmorphism, CSS Grid, Custom Properties), and Vanilla JavaScript (ES6+). Zero heavy frameworks required.
* **Components:**
  * **Interactive Predictor:** Card layout with responsive inputs, live feature-engineering ratio preview, and 4 quick-pick student presets.
  * **Dynamic SVG Score Ring:** Color-coded animated gauge displaying the continuous 0–10 score and empathetic wellbeing tier.
  * **Analytics Visualizations:** Chart.js integration rendering population feature importances, stress-level score averages, and cohort distributions.
  * **Diagnostics Tab:** Real-time benchmark table displaying cross-validation vs. test set performance across all 5 models.
  * **History Manager:** Local record history enabling users to review previous check-ins.

---

## 11. Automated Testing Suite

Tested with `pytest` across 20 comprehensive automated test cases:
* **API Endpoints:** `/health`, `/api/v1/predict`, `/api/v1/model-info`, `/api/v1/insights`, `/api/v1/predictions`.
* **Input Validation:** Verification that invalid ages, out-of-range usage hours, and unmapped stress levels return HTTP 422 with structured validation errors.
* **Pipeline Integrity:**
  * Model artifact successfully unpickles as a Scikit-Learn `Pipeline`.
  * Negative physical activity is clipped to 0.0.
  * Unseen/rare countries map to `'Other'`.
  * Directional sanity check: verifies that a healthy student profile predicts significantly higher wellbeing than an overloaded, sleep-deprived profile ($\Delta \ge 2.0$ points).

---

## 12. Known Limitations & Responsible-Use Ethics

### Limitations
1. **Self-Reported Survey Data:** Inputs are based on self-reported estimates rather than automated hardware telemetry (e.g., Apple Screen Time APIs).
2. **Student Demographic Focus:** The dataset captures students aged 18–25; generalization to older working professionals should be interpreted with caution.
3. **Statistical Correlation vs. Causation:** High screen time correlates with lower wellbeing, but may act as a coping symptom of existing stress rather than the sole root cause.

### Ethical Safeguards & Medical Disclaimer
MindTrack explicitly positions itself as an **educational, statistical analytics tool—NOT a clinical diagnostic instrument**. It includes non-diagnostic disclaimers and emergency helpline resources (US 988, UK 111, India 14416) across both UI and API contracts.

---

## 13. Common Technical Interview Questions

### Q: Why did you use an ensemble tree model instead of Deep Learning?
> *"With tabular survey data of 5,000 samples and heterogeneous features (continuous, ordinal, categorical), tree ensembles like Extra Trees and Random Forests consistently outperform Deep Neural Networks. They require less tuning, do not suffer from vanishing gradients, capture high-order feature interactions natively, and produce compact, fast artifacts (3.1 MB) with immediate inference latencies (< 15ms)."*

### Q: How did you ensure there was no data leakage in your ML pipeline?
> *"We bundled all feature engineering, scaling, and categorical encoding into a single scikit-learn `Pipeline` and `ColumnTransformer`. Cross-validation fits the pipeline strictly on the training folds. As a result, medians, standard deviations, and one-hot categories are learned purely from training data and never leak from validation or test splits."*

### Q: Why did you implement dual persistence (SQLite + MongoDB)?
> *"To ensure seamless developer experience and easy local deployment. Reviewers and recruiters can clone the repo and run the app instantly without configuring a database daemon. In production, setting `MONGODB_URI` seamlessly switches the application to distributed document storage."*
