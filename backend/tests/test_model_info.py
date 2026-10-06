"""Test Model Info and Insights endpoints."""

def test_model_info_endpoint(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data
    assert "r2" in data["metrics"]
    assert data["metrics"]["r2"] > 0.8
    assert "feature_importances" in data
    assert len(data["feature_importances"]) > 0
    assert "model_comparison" in data
    assert len(data["model_comparison"]) >= 4

def test_insights_endpoint(client):
    response = client.get("/api/v1/insights")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_summary" in data
    assert "feature_importances" in data
    assert "key_findings" in data
    assert len(data["key_findings"]) >= 3
