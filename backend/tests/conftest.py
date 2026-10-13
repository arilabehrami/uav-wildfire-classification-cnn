"""Infrastruktura e testeve.

Shumica e testeve nuk kerkojne TensorFlow: aplikacioni ndertohet me
'use_lifespan=False' dhe ne 'app.state.classification_service' vendoset nje
zevendesues qe respekton te njejten kontrate dalje si klasifikuesi i vertete.
Vetem test_preprocessing.py kerkon TF, dhe anashkalohet nese nuk eshte i
instaluar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml.zones import label_for, zone_for  # noqa: E402

# Porta te fiksuara per zevendesuesin. Sjellja e API-se nuk varet nga keto
# vlera; portat reale verifikohen ne test_contract.py.
LOW = 0.5244
HIGH = 0.9713

JPEG_MAGIC = b"\xff\xd8\xff"
VALID_IMAGE = JPEG_MAGIC + b"\x00" * 64


class StubService:
    """Sherbim i rremë: po ai kontrat dalje, pa model dhe pa TensorFlow."""

    def __init__(self, score: float = 0.99):
        self.score = score
        self.calls: list[int] = []

    def classify(self, image_bytes: bytes) -> dict:
        self.calls.append(len(image_bytes))

        if not image_bytes:
            raise ValueError("Imazhi është bosh.")
        if not image_bytes.startswith(JPEG_MAGIC):
            raise ValueError("Fajlli nuk mund të lexohet si imazh.")

        zone = zone_for(self.score, LOW, HIGH)
        return {
            "label": label_for(zone),
            "zone": zone,
            "fire_score": self.score,
            "gates": {"low": LOW, "high": HIGH},
            "model_version": "stub@000000000000",
            "model_sha256": "0" * 64,
            "calibrated": False,
        }


@pytest.fixture
def settings():
    # 1 MB e ben testin e tejkalimit te kufirit te lire ne memorie.
    return load_settings({"FIRE_MAX_UPLOAD_MB": "1", "FIRE_LOG_LEVEL": "WARNING"})


@pytest.fixture
def app(settings):
    return create_app(settings, use_lifespan=False)


@pytest.fixture
def service(app):
    stub = StubService()
    app.state.classification_service = stub
    return stub


@pytest.fixture
def client(app, service):
    from fastapi.testclient import TestClient

    # Pa 'with': lifespan-i nuk ekzekutohet, pra TF nuk importohet kurre.
    return TestClient(app)
