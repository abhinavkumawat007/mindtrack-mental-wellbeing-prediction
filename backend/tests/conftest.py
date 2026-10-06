"""Pytest fixtures for MindTrack test suite."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.storage import storage_service

@pytest.fixture(scope="session")
def client():
    """Returns a TestClient instance for API calls."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(autouse=True)
def clean_test_db():
    """Cleans test storage before running tests."""
    storage_service.clear_all()
    yield
