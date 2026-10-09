from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Legal Metrology Vision API"
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "keys_configured" in data


def test_analyze_without_keys_or_file():
    # Empty request should fail validation with 422
    response = client.post("/api/v1/analyze")
    assert response.status_code == 422
