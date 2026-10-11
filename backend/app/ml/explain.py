"""Grad-CAM per modelin ne sherbim.

Harta tregon se ku u aktivizua rrjeti me shume kur mori vendimin. Ajo NUK
eshte pozicioni i zjarrit: modeli jep nje numer te vetem per gjithe foton
dhe nuk eshte trajnuar kurre me koordinata. Matjet mbi dy kuadro te setit
te testimit e treguan kete qarte — qeliza me e nxehte nuk perkoi me flaken.

Grafi, sipas training/fire_final.py:

    Input(254,254,3) -> Resizing(224,224) -> Rescaling(2.0,-1.0)
                     -> MobileNetV2(include_top=False, pooling="avg")
                     -> fire_head: Dropout -> Dense(1, sigmoid)

Shtylla kurrizore e ka pooling-un brenda, prandaj harta merret nje shtrese
me heret, te 'out_relu', ku dalja eshte ende 7x7.

Nese ndarja e grafit deshton, Explainer.available mbetet False dhe sherbimi
vazhdon pa harte. Klasifikimi nuk preket kurre.
"""

from __future__ import annotations

import tensorflow as tf

from .preprocessing import prepare_batch

BACKBONE_PREFIXES = ("mobilenetv2", "xception")
LAST_CONV = {"mobilenetv2": "out_relu", "xception": "block14_sepconv2_act"}

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


class Explainer:
    """Ndan modelin nje here dhe e riperdor per çdo kerkese."""

    def __init__(self, model):
        self.available = False
        self.reason = None
        self.layer = None
        self.shape = None

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

            try:
                self._head = model.get_layer("fire_head")
            except ValueError:
                tail = [l for l in model.layers if isinstance(l, tf.keras.Sequential)]
                if not tail:
                    raise ValueError("Nuk u gjet koka 'fire_head'.")
                self._head = tail[-1]

            self.shape = tuple(self._conv.output_shape[1:3])
            self.available = True
        except Exception as error:  # nuk duhet te rrezoje kurre sherbimin
            self.reason = str(error)

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
                pooled = tf.reduce_mean(features, axis=(1, 2))
                score = self._head(pooled, training=False)[0, 0]

            grads = tape.gradient(score, features)
            if grads is None:
                return None

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
        except Exception:
            return None
