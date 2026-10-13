"""Leximi i konfigurimit nga ambienti."""

from __future__ import annotations

import pytest

from app.config import DEFAULT_ORIGINS, DEFAULT_RUN, load_settings


def test_defaults_without_any_environment_variable():
    settings = load_settings({})
    assert settings.model_run == DEFAULT_RUN
    assert settings.cors_origins == DEFAULT_ORIGINS
    assert settings.max_file_bytes == 10 * 1024 * 1024
    assert settings.max_file_mb == 10
    assert settings.verify_model_hash is True


def test_cors_origins_are_split_and_trimmed():
    settings = load_settings(
        {"FIRE_CORS_ORIGINS": " http://a.test , , http://b.test "}
    )
    assert settings.cors_origins == ("http://a.test", "http://b.test")


def test_blank_cors_origins_fall_back_to_defaults():
    assert load_settings({"FIRE_CORS_ORIGINS": "  ,  "}).cors_origins == DEFAULT_ORIGINS


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_upload_limit_must_be_at_least_one_mb(raw):
    with pytest.raises(ValueError):
        load_settings({"FIRE_MAX_UPLOAD_MB": raw})


def test_upload_limit_must_be_an_integer():
    with pytest.raises(ValueError):
        load_settings({"FIRE_MAX_UPLOAD_MB": "dhjete"})


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_hash_verification_can_be_disabled(raw):
    assert load_settings({"FIRE_VERIFY_HASH": raw}).verify_model_hash is False


def test_hash_verification_is_on_by_default():
    assert load_settings({"FIRE_VERIFY_HASH": "1"}).verify_model_hash is True
