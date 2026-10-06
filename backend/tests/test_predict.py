"""Test Prediction Endpoint and Validation."""

VALID_PAYLOAD = {
    "age": 21,
    "gender": "Female",
    "country": "Canada",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment",
    "avg_daily_usage_hours": 4.5,
    "daily_unlocks": 140,
    "study_hours": 4.0,
    "physical_activity_hours": 1.5,
    "sleep_hours_per_night": 7.0,
    "stress_level": "Medium"
}

def test_predict_success(client):
    response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mental_health_score" in data
    score = data["predicted_mental_health_score"]
    assert 1.0 <= score <= 10.0
    assert "wellbeing_tier" in data
    assert "tier_color" in data
    assert "interpretation" in data
    assert "contributing_factors" in data
    assert isinstance(data["contributing_factors"], list)
    assert len(data["contributing_factors"]) > 0
    assert data["model_used"] != ""
    assert "prediction_id" in data
    assert "disclaimer" in data

def test_predict_backward_compatible_endpoint(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mental_health_score" in data

def test_predict_high_strain_profile(client):
    payload = VALID_PAYLOAD.copy()
    payload.update({
        "avg_daily_usage_hours": 8.5,
        "daily_unlocks": 240,
        "sleep_hours_per_night": 4.5,
        "stress_level": "Very High",
        "study_hours": 1.0,
        "physical_activity_hours": 0.2
    })
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Score should be lower than baseline
    assert data["predicted_mental_health_score"] < 6.0
    # Negative factors should be identified
    negative_factors = [f for f in data["contributing_factors"] if f["impact"] == "negative"]
    assert len(negative_factors) > 0

def test_predict_invalid_age(client):
    payload = VALID_PAYLOAD.copy()
    payload["age"] = 5  # Too young (< 10)
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_hours(client):
    payload = VALID_PAYLOAD.copy()
    payload["avg_daily_usage_hours"] = -1.0  # Negative hours
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_stress_level(client):
    payload = VALID_PAYLOAD.copy()
    payload["stress_level"] = "Extreme"  # Not in allowed literals
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_required_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["sleep_hours_per_night"]
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
