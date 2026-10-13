"""Klasifikuesi ne sherbim.

Modeli, kontrata e hyrjes dhe pragjet lexohen nga inference.json i ekzekutimit;
asnje prej tyre nuk shkruhet ne kod. Keshtu nje ndryshim modeli nuk le pas nje
prag te vjeteruar, dhe fajlli qe citohet ne punim eshte fajlli qe sherben.

Logjika e zonave jeton ne app/ml/zones.py, pa TensorFlow, qe te testohet
ne njesi pa ngarkuar modelin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tensorflow as tf

from ..config import DEFAULT_RUN
from .preprocessing import prepare_batch
from .zones import label_for, validate_gates, zone_for

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FireClassifier:
    def __init__(self, run: str = DEFAULT_RUN, verify_hash: bool = True):
        self.run = run
        self.run_dir = ARTIFACTS / run
        contract_path = self.run_dir / "inference.json"

        if not contract_path.is_file():
            raise FileNotFoundError(
                f"Mungon kontrata e inferencës: {contract_path}"
            )

        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        gates = self.contract.get("gates")
        if not gates or "low" not in gates or "high" not in gates:
            raise ValueError(f"{contract_path} nuk përmban 'gates' me low dhe high.")

        self.low, self.high = validate_gates(gates["low"], gates["high"])

        height, width, channels = self.contract["input_size"]
        if channels != 3:
            raise ValueError("Kontrata pret tre kanale (RGB).")
        self.input_size = (int(height), int(width))

        model_path = self.run_dir / self.contract["model"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Mungon modeli: {model_path}")

        expected = self.contract.get("model_sha256")
        self.model_sha256 = _sha256(model_path)
        if verify_hash and expected and self.model_sha256 != expected:
            raise ValueError(
                f"SHA256 i {model_path.name} nuk përputhet me inference.json. "
                f"Pritej {expected}, u gjet {self.model_sha256}."
            )

        self.model = tf.keras.models.load_model(model_path, compile=False)

        if tuple(self.model.input_shape[1:]) != (*self.input_size, 3):
            raise ValueError(
                f"Modeli pret {self.model.input_shape[1:]}, kontrata "
                f"{(*self.input_size, 3)}."
            )
        if self.model.output_shape[-1] != 1:
            raise ValueError("Modeli duhet të japë një dalje sigmoid.")

        self.model_version = f"{run}@{self.model_sha256[:12]}"

    def zone_for(self, fire_score: float) -> str:
        return zone_for(fire_score, self.low, self.high)

    def classify(self, image_bytes: bytes) -> dict:
        batch = prepare_batch(image_bytes, self.input_size)
        fire_score = float(self.model(batch, training=False)[0, 0])
        zone = self.zone_for(fire_score)

        return {
            "label": label_for(zone),
            "zone": zone,
            "fire_score": fire_score,
            "gates": {"low": self.low, "high": self.high},
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "calibrated": bool(self.contract.get("calibrated", False)),
        }
