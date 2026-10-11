"""Grad-CAM mbi kuadro te vetme, me te njejten hyrje qe perdor sherbimi.

Pyetja qe i pergjigjet: kur modeli e humb nje zjarr, a e sheh fare flaken?

Grafi i vertete, sipas training/fire_final.py:

    Input(254,254,3)  ->  Resizing(224,224)  ->  Rescaling(2.0, -1.0)
                      ->  MobileNetV2(include_top=False, pooling="avg")
                      ->  fire_head: Dropout(0.35) -> Dense(1, sigmoid)

Dy pasoja qe e percaktojne kete skript:

  * Shtylla kurrizore e ka GlobalAveragePooling BRENDA saj (pooling="avg"),
    prandaj dalja e saj eshte (batch, 1280), jo 4D. Harta e aktivizimit
    merret nje shtrese me heret, te 'out_relu'.
  * Modeli e zvogelon vete 254 -> 224. Rrjeti i veçorive del 7x7, jo 8x8.

Hyrja merret nga app.ml.preprocessing.prepare_batch dhe modeli nga kontrata
inference.json, qe harta te tregoje saktesisht ate qe shikon API-ja.

    cd backend
    python training/inspect_gradcam.py --images ../dataset/Test/Fire/a.jpg ../dataset/Test/Fire/b.jpg

Del ne backend/artifacts/inspection/gradcam/:
    <emri>_cam.jpg   origjinali me harten e mbivendosur
    <emri>_raw.jpg   vetem harta
    summary.json     rezultati, zona dhe statistikat per secilen foto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import tensorflow as tf  # noqa: E402

from app.ml.classifier import FireClassifier  # noqa: E402
from app.ml.preprocessing import prepare_batch  # noqa: E402

DEFAULT_IMAGES = ["../dataset/Test/Fire/resized_test_fire_frame94.jpg"]

BACKBONE_PREFIXES = ("mobilenetv2", "xception")
# Shtresa e fundit konvolucionale perpara pooling-ut te brendshem.
LAST_CONV = {"mobilenetv2": "out_relu", "xception": "block14_sepconv2_act"}


# --------------------------------------------------------------------------
# Ndarja e grafit
# --------------------------------------------------------------------------

def find_backbone(model):
    """Nen-modeli MobileNetV2/Xception brenda modelit te ruajtur."""
    found = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and layer.name.lower().startswith(BACKBONE_PREFIXES)
    ]
    if len(found) != 1:
        names = [f"{l.name} ({type(l).__name__})" for l in model.layers]
        raise SystemExit(
            "Nuk u identifikua nje shtylle e vetme kurrizore.\n"
            f"Shtresat e modelit: {names}"
        )
    return found[0]


def last_conv_name(backbone):
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
    raise SystemExit("Asnje shtrese 4D brenda shtylles kurrizore.")


def build_parts(model):
    """Kthen (stem, conv, head, conv_name).

    stem : Input(254) -> dalja e Rescaling-ut, pra hyrja e shtylles
    conv : hyrja e shtylles -> harta 7x7 e veçorive
    head : (1280,) -> sigmoid
    """
    backbone = find_backbone(model)

    scaling = [l for l in model.layers if l.name.endswith("_scaling")]
    if not scaling:
        scaling = [l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)]
    if not scaling:
        raise SystemExit("Nuk u gjet shtresa e shkallezimit perpara shtylles.")
    stem = tf.keras.Model(model.inputs, scaling[-1].output, name="stem")

    name = last_conv_name(backbone)
    conv = tf.keras.Model(backbone.input, backbone.get_layer(name).output, name="conv")

    try:
        head = model.get_layer("fire_head")
    except ValueError:
        tail = [l for l in model.layers if isinstance(l, tf.keras.Sequential)]
        if not tail:
            raise SystemExit("Nuk u gjet koka 'fire_head'.")
        head = tail[-1]

    return stem, conv, head, name


def forward(stem, conv, head, batch):
    """Rindertimi i daljes. Duhet te perputhet me model(batch)."""
    features = conv(stem(batch, training=False), training=False)
    pooled = tf.reduce_mean(features, axis=(1, 2))      # == pooling="avg"
    return features, head(pooled, training=False)[0, 0]


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------

def grad_cam(stem, conv, head, batch):
    with tf.GradientTape() as tape:
        features = conv(stem(batch, training=False), training=False)
        tape.watch(features)
        pooled = tf.reduce_mean(features, axis=(1, 2))
        score = head(pooled, training=False)[0, 0]

    grads = tape.gradient(score, features)
    if grads is None:
        raise SystemExit("Gradienti doli None; grafi u pre diku.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1)).numpy()

    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    return cam, float(score), peak


# --------------------------------------------------------------------------
# Vizatimi
# --------------------------------------------------------------------------

def colorize(cam01: np.ndarray) -> np.ndarray:
    stops = np.array([
        [0.0, 10, 18, 16],
        [0.5, 228, 176, 74],
        [1.0, 255, 106, 31],
    ])
    out = np.zeros((*cam01.shape, 3), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(cam01, stops[:, 0], stops[:, channel + 1])
    return out


def overlay(image: Image.Image, cam01: np.ndarray, alpha: float) -> Image.Image:
    cam_img = Image.fromarray((cam01 * 255).astype(np.uint8), mode="L")
    resized = np.asarray(cam_img.resize(image.size, Image.BILINEAR)).astype(np.float32) / 255.0
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    weight = (resized * alpha)[..., None]
    blended = base * (1 - weight) + colorize(resized) * weight
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    parser.add_argument("--out", default=None)
    parser.add_argument("--alpha", type=float, default=0.65)
    args = parser.parse_args()

    classifier = FireClassifier(run=args.run) if args.run else FireClassifier()
    stem, conv, head, conv_name = build_parts(classifier.model)

    out_dir = Path(args.out) if args.out else BACKEND / "artifacts" / "inspection" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"modeli     : {classifier.model_version}")
    print(f"hyrja      : {classifier.input_size} RGB [0,1]  ->  brenda: {stem.output_shape[1:3]}")
    print(f"portat     : {classifier.low:.4f} / {classifier.high:.4f}")
    print(f"harta      : {conv_name}, {conv.output_shape[1:3]} qeliza x {conv.output_shape[-1]} kanale")
    print(f"dalja      : {out_dir}\n")

    summary = []
    for raw in args.images:
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            print(f"  ! nuk u gjet: {path}")
            continue

        batch = prepare_batch(path.read_bytes(), classifier.input_size)

        # Kontroll: rindertimi duhet te jape te njejten dalje si modeli i plote.
        reference = float(classifier.model(batch, training=False)[0, 0])
        _, rebuilt = forward(stem, conv, head, batch)
        drift = abs(reference - float(rebuilt))
        if drift > 1e-4:
            raise SystemExit(
                f"Rindertimi nuk perputhet me modelin ({reference:.6f} kundrejt "
                f"{float(rebuilt):.6f}). Harta do te ishte e pavlefshme."
            )

        cam, score, peak = grad_cam(stem, conv, head, batch)
        zone = classifier.zone_for(score)

        original = Image.open(path).convert("RGB")
        stem_name = path.stem
        overlay(original, cam, args.alpha).save(out_dir / f"{stem_name}_cam.jpg", quality=95)
        Image.fromarray(colorize(cam).astype(np.uint8)).resize(
            original.size, Image.NEAREST
        ).save(out_dir / f"{stem_name}_raw.jpg", quality=95)

        hot = float((cam > 0.5).mean())
        y, x = np.unravel_index(int(cam.argmax()), cam.shape)

        summary.append({
            "image": str(path),
            "fire_score": score,
            "zone": zone,
            "model_check_drift": drift,
            "cam_shape": list(cam.shape),
            "cam_peak_raw": peak,
            "cam_hot_fraction": hot,
            "cam_argmax_cell": [int(y), int(x)],
        })

        flag = "" if peak > 0 else "   <- harta krejt zero"
        print(
            f"  {stem_name:38s} score={score:.4f}  zona={zone:<6s} "
            f"qeliza me e nxehte=({y},{x})  e nxehte={hot:.3f}{flag}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps({"model_version": classifier.model_version, "results": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nu shkrua {out_dir / 'summary.json'}")
    print(
        f"\nHarta eshte {conv.output_shape[1]}x{conv.output_shape[2]} per nje hyrje "
        f"{stem.output_shape[1]}x{stem.output_shape[2]}.\n"
        "Zmadhimi ne _cam.jpg eshte interpolim, jo lokalizim i sakte i flakes."
    )


if __name__ == "__main__":
    main()
