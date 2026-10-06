"""Test Prediction History and Deletion."""

from backend.tests.test_predict import VALID_PAYLOAD

def test_history_workflow(client):
    # 1. Make 2 predictions
    res1 = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert res1.status_code == 200
    id1 = res1.json()["prediction_id"]

    payload2 = VALID_PAYLOAD.copy()
    payload2["sleep_hours_per_night"] = 8.5
    res2 = client.post("/api/v1/predict", json=payload2)
    assert res2.status_code == 200
    id2 = res2.json()["prediction_id"]

    # 2. Get history
    hist_res = client.get("/api/v1/predictions")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total"] >= 2
    record_ids = [r["id"] for r in hist_data["records"]]
    assert id1 in record_ids
    assert id2 in record_ids

    # 3. Delete a specific prediction
    del_res = client.delete(f"/api/v1/predictions/{id1}")
    assert del_res.status_code == 200

    # 4. Confirm it's gone
    hist_after = client.get("/api/v1/predictions").json()
    record_ids_after = [r["id"] for r in hist_after["records"]]
    assert id1 not in record_ids_after
    assert id2 in record_ids_after

    # 5. Delete non-existent
    del_fake = client.delete("/api/v1/predictions/nonexistent-id-xyz")
    assert del_fake.status_code == 404
