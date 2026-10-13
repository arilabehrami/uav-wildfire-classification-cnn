"""Grad-CAM: rindertimi i grafit duhet te perputhet me modelin e vertete.

Gabimi qe ky test e ruan: kur fire_flame_v6 kaloi ne concat(avg, max) pas
shtylles, explain.py vazhdonte te llogariste vetem mesataren. Koka refuzonte
formen, harta kthehej None per çdo foto, dhe logu thoshte 'Grad-CAM aktiv'.

Kerkon TensorFlow. Testet me modelet e ruajtura anashkalohen kur fajlli
.keras mungon (eshte jashte git-it).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow nuk eshte i instaluar")

from PIL import Image  # noqa: E402

from app.ml.explain import Explainer  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
K = tf.keras


def _tiny(tail: str):
    """Model i vogel me te njejten forme si modelet reale.

    tail: "inside" -> pooling mesatar brenda shtylles (si final_fire_v1)
          "avgmax" -> concat(avg, max) jashte shtylles (si fire_flame_v6)
          "maxavg" -> po ai concat, me rend te kundert
          "flatten"-> nje bisht qe Explainer nuk e njeh
    """
    inp = K.Input((32, 32, 3), name="rgb_0_to_1")
    x = K.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_scaling")(inp)

    b_in = K.Input((32, 32, 3))
    y = K.layers.Conv2D(8, 3, strides=4, name="Conv_1")(b_in)
    y = K.layers.ReLU(name="out_relu")(y)
    if tail == "inside":
        y = K.layers.GlobalAveragePooling2D()(y)
    backbone = K.Model(b_in, y, name="mobilenetv2_tiny")

    f = backbone(x)
    if tail in ("avgmax", "maxavg"):
        a = K.layers.GlobalAveragePooling2D(name="avg_features")(f)
        m = K.layers.GlobalMaxPooling2D(name="max_features")(f)
        f = K.layers.Concatenate(name="avg_and_max")([a, m] if tail == "avgmax" else [m, a])
    elif tail == "flatten":
        f = K.layers.Flatten(name="flat")(f)

    head = K.Sequential(
        [K.layers.Dropout(0.3), K.layers.Dense(1, activation="sigmoid")],
        name="fire_head",
    )
    return K.Model(inp, head(f))


def _png(size=32, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("tail", ["inside", "avgmax", "maxavg"])
def test_reconstruction_matches_the_model(tail):
    model = _tiny(tail)
    explainer = Explainer(model)
    assert explainer.available, explainer.reason

    batch = tf.random.stateless_uniform((3, 32, 32, 3), seed=(1, 2))
    np.testing.assert_allclose(
        np.asarray(explainer._forward(batch)),
        np.asarray(model(batch, training=False)),
        atol=1e-5,
    )


@pytest.mark.parametrize("tail", ["inside", "avgmax"])
def test_heatmap_has_the_conv_grid_shape(tail):
    explainer = Explainer(_tiny(tail))
    result = explainer.heatmap(_png(), (32, 32))

    assert result is not None
    assert (result["rows"], result["cols"]) == explainer.shape
    values = np.array(result["grid"])
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_unknown_tail_is_reported_not_guessed():
    explainer = Explainer(_tiny("flatten"))
    assert not explainer.available
    assert "Pooling" in explainer.reason


@pytest.mark.parametrize("run", ["final_fire_v1", "fire_flame_v6"])
def test_saved_models_get_a_working_heatmap(run):
    path = ARTIFACTS / run / "model.keras"
    if not path.is_file():
        pytest.skip(f"{path} mungon (modelet jane jashte git-it)")

    model = tf.keras.models.load_model(path, compile=False)
    explainer = Explainer(model)
    assert explainer.available, explainer.reason

    size = tuple(model.input_shape[1:3])
    result = explainer.heatmap(_png(size[0]), size)
    assert result is not None
    assert (result["rows"], result["cols"]) == (7, 7)
