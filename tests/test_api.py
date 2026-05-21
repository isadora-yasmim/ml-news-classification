from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    payload = {
        "headline": "New technology impacts companies",
        "short_description": "Artificial intelligence tools are changing productivity.",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_category" in data
    assert data["predicted_category"] is not None
    assert "confidence" in data


def test_predict_endpoint_with_empty_headline() -> None:
    payload = {
        "headline": "",
        "short_description": "Some description",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_with_empty_short_description() -> None:
    payload = {
        "headline": "Some headline",
        "short_description": "",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_with_very_short_text() -> None:
    payload = {
        "headline": "AI",
        "short_description": "Tech",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_category" in data
    assert data["predicted_category"] is not None