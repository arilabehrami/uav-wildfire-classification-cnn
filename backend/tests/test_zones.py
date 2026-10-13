"""Kufijte e zonave.

Kontrata: score < low -> clear; low <= score < high -> review; score >= high
-> alarm. Testet e provojne pikerisht ne kufij, sepse ketu nje '<' kundrejt
'<=' e ndryshon vendimin per nje kuader real.
"""

from __future__ import annotations

import math

import pytest

from app.ml.zones import ALARM, CLEAR, REVIEW, label_for, validate_gates, zone_for

LOW = 0.5244
HIGH = 0.9713
EPS = 1e-9


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, CLEAR),
        (LOW - EPS, CLEAR),
        (LOW, REVIEW),
        ((LOW + HIGH) / 2, REVIEW),
        (HIGH - EPS, REVIEW),
        (HIGH, ALARM),
        (1.0, ALARM),
    ],
)
def test_zone_for_at_boundaries(score, expected):
    assert zone_for(score, LOW, HIGH) == expected


def test_low_is_inclusive_and_high_is_inclusive():
    # 'low' hyn ne review, 'high' hyn ne alarm. Asnje vlere nuk bie jashte.
    assert zone_for(LOW, LOW, HIGH) == REVIEW
    assert zone_for(HIGH, LOW, HIGH) == ALARM


def test_degenerate_gates_leave_no_review_zone():
    # Kur low == high, zona e shqyrtimit zhduket dhe 'high' ka perparesi.
    assert zone_for(0.5, 0.5, 0.5) == ALARM
    assert zone_for(0.5 - EPS, 0.5, 0.5) == CLEAR


@pytest.mark.parametrize("zone, expected", [(CLEAR, "No_Fire"), (REVIEW, "Fire"), (ALARM, "Fire")])
def test_label_for(zone, expected):
    assert label_for(zone) == expected


def test_label_for_rejects_unknown_zone():
    with pytest.raises(ValueError):
        label_for("maybe")


def test_validate_gates_accepts_valid_pair():
    assert validate_gates(0.2, 0.8) == (0.2, 0.8)
    assert validate_gates("0.3", "0.9") == (0.3, 0.9)


@pytest.mark.parametrize("low, high", [(0.8, 0.2), (-0.1, 0.5), (0.5, 1.1), (math.nan, 0.5)])
def test_validate_gates_rejects_invalid_pairs(low, high):
    with pytest.raises(ValueError):
        validate_gates(low, high)
