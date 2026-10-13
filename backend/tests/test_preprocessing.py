"""Kontrata e hyrjes.

Nje ndryshim ne preprocessing e prish krahasueshmerine me rezultatet e
raportuara ne punim, prandaj forma dhe diapazoni fiksohen ne test.
Kerkon TensorFlow; anashkalohet nese nuk eshte i instaluar.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow nuk eshte i instaluar")

from PIL import Image  # noqa: E402

from app.ml.preprocessing import decode, prepare_batch  # noqa: E402

SIZE = (254, 254)


def _png(width: int, height: int, color=(200, 40, 10)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_rejects_empty_bytes():
    with pytest.raises(ValueError):
        decode(b"")


def test_decode_rejects_bytes_that_are_not_an_image():
    with pytest.raises(ValueError):
        decode(b"kjo nuk eshte foto" * 10)


def test_decode_returns_three_channels():
    assert decode(_png(8, 12)).shape[-1] == 3


@pytest.mark.parametrize("width, height", [(16, 16), (640, 480), (33, 7)])
def test_prepare_batch_always_returns_the_contract_shape(width, height):
    batch = prepare_batch(_png(width, height), SIZE)
    assert tuple(batch.shape) == (1, *SIZE, 3)


def test_prepare_batch_scales_into_the_unit_range():
    batch = prepare_batch(_png(64, 64, (255, 255, 255)), SIZE)
    assert float(batch.numpy().min()) >= 0.0
    assert float(batch.numpy().max()) <= 1.0


def test_white_image_maps_to_one_and_black_to_zero():
    white = prepare_batch(_png(32, 32, (255, 255, 255)), SIZE).numpy()
    black = prepare_batch(_png(32, 32, (0, 0, 0)), SIZE).numpy()
    assert float(white.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(black.max()) == pytest.approx(0.0, abs=1e-6)


def test_grayscale_input_is_expanded_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (24, 24), 128).save(buffer, format="PNG")
    assert tuple(prepare_batch(buffer.getvalue(), SIZE).shape) == (1, *SIZE, 3)
