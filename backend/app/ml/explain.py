"""Grad-CAM per modelin ne sherbim.

Harta tregon se ku u aktivizua rrjeti me shume kur mori vendimin. Ajo NUK
eshte pozicioni i zjarrit: modeli jep nje numer te vetem per gjithe foton
dhe nuk eshte trajnuar kurre me koordinata.

Grafi ndahet ne kater pjese, te gjitha te marra nga vete modeli:

    stem:  Input -> Resizing -> Rescaling
    conv:  shtylla kurrizore deri te shtresa e fundit konvolucionale (7x7)
    pool:  pooling-u qe modeli perdor pas saj
    head:  fire_head -> Dense(1, sigmoid)

Pooling-u nuk supozohet. Dy arkitektura jane ne perdorim:

    final_fire_v1:  MobileNetV2(pooling="avg") -> fire_head
                    (mesatarja eshte brenda shtylles)
    fire_flame_v6:  MobileNetV2 -> concat(GlobalAvgPool, GlobalMaxPool) -> fire_head
                    (pooling-u eshte jashte, ne rendin e lexuar nga lidhjet)

Ne nisje, rindertimi krahasohet me daljen e modelit te vertete mbi dy foto
provuese. Nese nuk perputhen, Explainer mbetet i padisponueshem me arsye te
shkruar ne log. Kjo e ndalon deshtimin e heshtur: me pare, nje mospërputhje
ktheheshte si 'None' per çdo foto ndersa logu thoshte 'Grad-CAM aktiv'.

Klasifikimi nuk preket kurre nga ky modul.
"""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf

from .preprocessing import prepare_batch

logger = logging.getLogger("fire.api")

BACKBONE_PREFIXES = ("mobilenetv2", "xception")
LAST_CONV = {"mobilenetv2": "out_relu", "xception": "block14_sepconv2_act"}
POOLING = (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalMaxPooling2D)
TOLERANCE = 1e-4

NOTE = (
    "Zona që ndikoi më shumë në vendimin e modelit. "
    "Nuk është pozicioni i zjarrit — modeli nuk e lokalizon atë."
)


def _find_backbone(model):
    found = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and layer.name.lower().startswith(BACKBONE_PREFIXES)
    ]
    if len(found) != 1:
        raise ValueError("Nuk u identifikua nje shtylle e vetme kurrizore.")
    return found[0]


def _last_conv_name(backbone):
    preferred = LAST_CONV.get(backbone.name.lower().split("_")[0])
    if preferred:
        try:
            backbone.get_layer(preferred)
            return preferred
        except ValueError:
            pass
    for layer in reversed(backbone.layers):
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name
    raise ValueError("Asnje shtrese 4D brenda shtylles kurrizore.")


def _inbound_names(model, layer_name):
    """Emrat e shtresave qe ushqejne 'layer_name', ne rendin e lidhjes."""
    for spec in model.get_config()["layers"]:
        if spec["config"].get("name") != layer_name:
            continue
        args = spec["inbound_nodes"][0]["args"][0]
        if isinstance(args, dict):
            args = [args]
        return [a["config"]["keras_history"][0] for a in args]
    raise ValueError(f"Shtresa {layer_name!r} nuk u gjet ne konfigurim.")


def _pooling_after(model, backbone, last_conv):
    """Funksioni features (1,H,W,C) -> vektori qe modeli i jep kokes."""
    # Rasti 1: pooling-u eshte brenda shtylles, pas shtreses konvolucionale.
    inner = backbone.layers
    names = [layer.name for layer in inner]
    after_conv = inner[names.index(last_conv) + 1:]
    inner_pools = [layer for layer in after_conv if isinstance(layer, POOLING)]
    if inner_pools:
        if len(inner_pools) != 1 or after_conv[-1] is not inner_pools[0]:
            raise ValueError("Pooling i panjohur brenda shtylles kurrizore.")
        pool = inner_pools[0]
        return lambda features: pool(features), pool.name

    # Rasti 2: pooling-u eshte jashte, ne modelin e jashtem.
    outer = model.layers
    names = [layer.name for layer in outer]
    tail = outer[names.index(backbone.name) + 1:]
    pools = {layer.name: layer for layer in tail if isinstance(layer, POOLING)}
    concats = [layer for layer in tail if isinstance(layer, tf.keras.layers.Concatenate)]

    if len(pools) == 1 and not concats:
        pool = next(iter(pools.values()))
        return lambda features: pool(features), pool.name

    if pools and len(concats) == 1:
        order = _inbound_names(model, concats[0].name)
        if sorted(order) != sorted(pools):
            raise ValueError(f"Concatenate lidh {order}, pooling-u gjetur: {sorted(pools)}.")
        layers = [pools[name] for name in order]
        axis = concats[0].axis

        def pooled(features):
            return tf.concat([layer(features) for layer in layers], axis=axis)

        return pooled, f"concat({', '.join(order)})"

    raise ValueError("Pooling-u pas shtylles kurrizore nuk u njoh.")


class Explainer:
    """Ndan modelin nje here, e verifikon, dhe e riperdor per çdo kerkese."""

    def __init__(self, model):
        self.available = False
        self.reason = None
        self.layer = None
        self.shape = None
        self.pooling = None
        self._warned = False

        try:
            backbone = _find_backbone(model)

            scaling = [l for l in model.layers if l.name.endswith("_scaling")]
            if not scaling:
                scaling = [
                    l for l in model.layers
                    if isinstance(l, tf.keras.layers.Rescaling)
                ]
            if not scaling:
                raise ValueError("Nuk u gjet shtresa e shkallezimit.")

            self._stem = tf.keras.Model(model.inputs, scaling[-1].output)
            self.layer = _last_conv_name(backbone)
            self._conv = tf.keras.Model(
                backbone.input, backbone.get_layer(self.layer).output
            )
            self._pool, self.pooling = _pooling_after(model, backbone, self.layer)

            try:
                self._head = model.get_layer("fire_head")
            except ValueError:
                tail = [l for l in model.layers if isinstance(l, tf.keras.Sequential)]
                if not tail:
                    raise ValueError("Nuk u gjet koka 'fire_head'.")
                self._head = tail[-1]

            self.shape = tuple(self._conv.output.shape[1:3])
            self._verify(model)
            self.available = True
        except Exception as error:  # nuk duhet te rrezoje kurre sherbimin
            self.reason = str(error)

    def _forward(self, batch):
        features = self._conv(self._stem(batch, training=False), training=False)
        return self._head(self._pool(features), training=False)

    def _verify(self, model):
        """Rindertimi duhet te jape te njejtin numer si modeli i vertete."""
        height, width = model.input_shape[1:3]
        probe = tf.random.stateless_uniform((2, height, width, 3), seed=(7, 13))
        expected = np.asarray(model(probe, training=False)).ravel()
        got = np.asarray(self._forward(probe)).ravel()
        if not np.allclose(expected, got, atol=TOLERANCE):
            raise ValueError(
                f"Rindertimi i grafit nuk perputhet me modelin "
                f"(modeli {expected.round(6).tolist()}, rindertimi "
                f"{got.round(6).tolist()}); harta do te ishte e pavlefshme."
            )

    def heatmap(self, image_bytes: bytes, input_size) -> dict | None:
        """Harta HxW e normalizuar ne [0,1], ose None nese s'eshte e mundur."""
        if not self.available:
            return None

        try:
            batch = prepare_batch(image_bytes, input_size)

            with tf.GradientTape() as tape:
                features = self._conv(self._stem(batch, training=False),
                                      training=False)
                tape.watch(features)
                score = self._head(self._pool(features), training=False)[0, 0]

            grads = tape.gradient(score, features)
            if grads is None:
                raise ValueError("Gradienti doli None.")

            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))

            peak = float(tf.reduce_max(cam))
            grid = (cam / peak if peak > 0 else cam).numpy()

            rows, cols = grid.shape
            return {
                "grid": [[round(float(v), 4) for v in row] for row in grid],
                "rows": int(rows),
                "cols": int(cols),
                "layer": self.layer,
                "peak_raw": peak,
                "note": NOTE,
            }
        except Exception as error:
            # Logohet njehere per proces, qe deshtimi te mos jete i heshtur.
            if not self._warned:
                logger.warning("Grad-CAM deshtoi, klasifikimi vazhdon pa harte: %s", error)
                self._warned = True
            return None
