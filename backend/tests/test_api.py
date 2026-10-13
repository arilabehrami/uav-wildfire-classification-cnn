"""Rrugët HTTP, pa model dhe pa TensorFlow."""

from __future__ import annotations

import pytest

from tests.conftest import JPEG_MAGIC, VALID_IMAGE


def _upload(payload: bytes, name: str = "frame.jpg", mime: str = "image/jpeg"):
    return {"file": (name, payload, mime)}


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_loading_when_service_absent(app):
    from fastapi.testclient import TestClient

    app.state.classification_service = None
    body = TestClient(app).get("/health").json()
    assert body == {"status": "loading", "model_loaded": False}


def test_classify_returns_full_contract(client):
    response = client.post("/api/classify", files=_upload(VALID_IMAGE))
    assert response.status_code == 200

    body = response.json()
    assert body["label"] == "Fire"
    assert body["zone"] == "alarm"
    assert body["fire_score"] == pytest.approx(0.99)
    assert body["gates"] == {"low": 0.5244, "high": 0.9713}
    assert body["calibrated"] is False
    assert "model_sha256" in body and "model_version" in body


def test_classify_rejects_empty_file(client):
    response = client.post("/api/classify", files=_upload(b""))
    assert response.status_code == 400
    assert "bosh" in response.json()["detail"].lower()


def test_classify_rejects_non_image(client):
    response = client.post("/api/classify", files=_upload(b"kjo nuk eshte foto"))
    assert response.status_code == 400


def test_classify_rejects_oversize_without_reading_it_all(client, service, settings):
    too_big = JPEG_MAGIC + b"\x00" * settings.max_file_bytes
    response = client.post("/api/classify", files=_upload(too_big))

    assert response.status_code == 413
    assert "MB" in response.json()["detail"]
    # Ngarkesa nuk arriti kurre te modeli.
    assert service.calls == []


def test_classify_accepts_file_exactly_at_the_limit(client, service, settings):
    exact = JPEG_MAGIC + b"\x00" * (settings.max_file_bytes - len(JPEG_MAGIC))
    response = client.post("/api/classify", files=_upload(exact))

    assert response.status_code == 200
    assert service.calls == [settings.max_file_bytes]


def test_classify_returns_503_when_model_not_loaded(app):
    from fastapi.testclient import TestClient

    app.state.classification_service = None
    response = TestClient(app).post("/api/classify", files=_upload(VALID_IMAGE))
    assert response.status_code == 503


def test_classify_requires_a_file(client):
    assert client.post("/api/classify").status_code == 422


def test_cors_allows_the_configured_origin(client, settings):
    response = client.options(
        "/api/classify",
        headers={
            "Origin": settings.cors_origins[0],
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == settings.cors_origins[0]
