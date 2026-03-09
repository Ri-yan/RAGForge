"""Integration tests for the API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_ingestion_service, get_query_service
from src.main import app


@pytest.fixture
def mock_query_service():
    service = MagicMock()
    service.ask.return_value = {
        "question": "test?",
        "answer": "test answer",
        "sources": [{"content": "source text", "metadata": {"source": "test.txt"}}],
    }
    return service


@pytest.fixture
def client(mock_query_service):
    app.dependency_overrides[get_query_service] = lambda: mock_query_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_query_endpoint(client):
    response = client.post("/api/v1/query/", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "test answer"


def test_query_empty_question(client):
    response = client.post("/api/v1/query/", json={"question": ""})
    assert response.status_code == 422
