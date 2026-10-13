"""Kontrata e inferences dhe pershtatja e frontend-it me te.

Dy gjera ruhen ketu. E para: çdo inference.json i ekzekutimeve te ruajtura
permban fushat qe klasifikuesi i kerkon, me vlera te perdorshme. E dyta:
vlerat e rezerves ne frontend/src/config/model.js nuk largohen pa u vene re
nga ekzekutimi te cilin pretendojne te pasqyrojne.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import DEFAULT_RUN
from app.ml.zones import validate_gates

BACKEND = Path(__file__).resolve().parents[1]
ARTIFACTS = BACKEND / "artifacts"
MODEL_JS = BACKEND.parent / "frontend" / "src" / "config" / "model.js"

CONTRACTS = sorted(ARTIFACTS.glob("*/inference.json"))


def _number(source: str, pattern: str) -> float:
    match = re.search(pattern, source)
    assert match, f"Nuk u gjet {pattern!r} ne {MODEL_JS.name}."
    return float(match.group(1))


def _string(source: str, pattern: str) -> str:
    match = re.search(pattern, source)
    assert match, f"Nuk u gjet {pattern!r} ne {MODEL_JS.name}."
    return match.group(1)


def test_at_least_one_contract_exists():
    assert CONTRACTS, f"Asnje inference.json nen {ARTIFACTS}."


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.parent.name)
def test_contract_has_usable_fields(path: Path):
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert contract["model"].endswith(".keras")
    assert re.fullmatch(r"[0-9a-f]{64}", contract["model_sha256"])
    assert contract["labels"] == {"No_Fire": 0, "Fire": 1}
    assert contract["input_range"] == [0, 1]
    assert contract["color"] == "RGB"

    height, width, channels = contract["input_size"]
    assert channels == 3
    assert height > 0 and width > 0

    gates = contract.get("gates")
    if gates is not None:
        validate_gates(gates["low"], gates["high"])


def test_default_run_is_servable():
    """Ekzekutimi i parazgjedhur duhet te kete porta; pa ato API-ja nuk ngrihet."""
    contract = json.loads(
        (ARTIFACTS / DEFAULT_RUN / "inference.json").read_text(encoding="utf-8")
    )
    gates = contract.get("gates")
    assert gates, f"{DEFAULT_RUN} duhet te kete 'gates' me low dhe high."
    validate_gates(gates["low"], gates["high"])


@pytest.mark.skipif(not MODEL_JS.is_file(), reason="frontend/src/config/model.js mungon")
def test_frontend_fallback_matches_the_run_it_claims():
    source = MODEL_JS.read_text(encoding="utf-8")
    run = _string(source, r'run:\s*"([^"]+)"')

    contract_path = ARTIFACTS / run / "inference.json"
    assert contract_path.is_file(), f"model.js citon '{run}', qe nuk ekziston."
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert _string(source, r'sha256:\s*"([0-9a-f]{64})"') == contract["model_sha256"]
    assert _string(source, r'file:\s*"([^"]+)"') == contract["model"]

    gates = contract["gates"]
    assert _number(source, r"low:\s*([0-9.]+)") == pytest.approx(gates["low"])
    assert _number(source, r"high:\s*([0-9.]+)") == pytest.approx(gates["high"])
