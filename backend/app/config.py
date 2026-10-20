"""Konfigurimi i sherbimit, i lexuar nga ambienti.

Nje vend i vetem per çdo vlere qe ndryshon mes zhvillimit, testimit dhe
demonstrimit. Vlerat e parazgjedhura e mbajne projektin te ekzekutueshem
pa caktuar asnje variabel ambienti.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Modeli i flakes (Fire = flake e dukshme). final_fire_v1, modeli i tymit,
# mbetet i disponueshem me FIRE_MODEL_RUN=final_fire_v1 per krahasim.
DEFAULT_RUN = "fire_flame_scene_repair_v1"
DEFAULT_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
DEFAULT_MAX_MB = 10
FALSEY = {"0", "false", "no", "off"}


def _split(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


@dataclass(frozen=True)
class Settings:
    """Vlerat e ekzekutimit. E pandryshueshme: lexohet njehere, ne nisje."""

    model_run: str
    cors_origins: tuple[str, ...]
    max_file_bytes: int
    log_level: str
    verify_model_hash: bool

    @property
    def max_file_mb(self) -> int:
        return self.max_file_bytes // (1024 * 1024)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Ngre ValueError nese ambienti permban vlere te paperdorshme."""
    source = os.environ if env is None else env

    raw_mb = source.get("FIRE_MAX_UPLOAD_MB", str(DEFAULT_MAX_MB))
    try:
        max_mb = int(raw_mb)
    except ValueError as error:
        raise ValueError(
            f"FIRE_MAX_UPLOAD_MB duhet te jete numer i plote; u gjet {raw_mb!r}."
        ) from error
    if max_mb < 1:
        raise ValueError("FIRE_MAX_UPLOAD_MB duhet te jete 1 ose me i madh.")

    origins = _split(source.get("FIRE_CORS_ORIGINS", ""))

    return Settings(
        model_run=source.get("FIRE_MODEL_RUN", DEFAULT_RUN),
        cors_origins=tuple(origins) if origins else DEFAULT_ORIGINS,
        max_file_bytes=max_mb * 1024 * 1024,
        log_level=source.get("FIRE_LOG_LEVEL", "INFO").upper(),
        verify_model_hash=source.get("FIRE_VERIFY_HASH", "1").lower() not in FALSEY,
    )
