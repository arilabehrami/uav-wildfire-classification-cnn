"""Logjika e zonave te vendimit, pa varesi nga TensorFlow.

E ndare nga klasifikuesi me qellim qe kufijte e portave te testohen ne njesi
pa ngarkuar modelin, dhe qe i njejti rregull te mos rishkruhet ne frontend.

Kufijte, ne mod eksplicit:
    score <  low           -> clear   (kuadri pastrohet pa njeri)
    low  <= score <  high  -> review  (duhet ta shohe njeriu)
    score >= high          -> alarm   (njoftim i menjehershem)
"""

from __future__ import annotations

CLEAR = "clear"
REVIEW = "review"
ALARM = "alarm"
ZONES = (CLEAR, REVIEW, ALARM)


def validate_gates(low: float, high: float) -> tuple[float, float]:
    """Kthen portat si float ose ngre ValueError nese nuk formojne tri zona."""
    low = float(low)
    high = float(high)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(
            f"Duhet 0 <= low <= high <= 1; u gjet low={low}, high={high}."
        )
    return low, high


def zone_for(score: float, low: float, high: float) -> str:
    """Zona per nje rezultat. 'high' ka perparesi nese low == high."""
    if score >= high:
        return ALARM
    if score < low:
        return CLEAR
    return REVIEW


def label_for(zone: str) -> str:
    """Vendimi binar i skanimit: çdo kuader qe nuk pastrohet trajtohet si Fire."""
    if zone not in ZONES:
        raise ValueError(f"Zone e panjohur: {zone!r}.")
    return "No_Fire" if zone == CLEAR else "Fire"
