"""Nxjerr portat per fire_flame_v6 dhe shkruan inference.json e tij.

Portat nxirren mbi validation_flame_reviewed_v1 dhe mandej zbatohen NJE HERE
mbi test_final (i rietiketuar sipas politikes se flakes) vetem per raportim.
Asnje porte nuk pershtatet mbi setin e raportimit.

Rregulli:
    low  = pragu ME I LARTE qe ende mban recall >= 99% per flaken
    high = pragu me i ulet qe jep specificity >= 90% mbi jo-flaken

Precision NUK perdoret si cak, ndryshe nga final_fire_v1. Precision varet nga
sa foto me flake ka ne set: validation ka 62% flake, test_final vetem 24%.
Nje prag i pershtatur mbi precision ne njerin set shembet ne tjetrin.

Me tej: 67.6% e negativave te validation jane kuadro liqeni, nje fluturim pa
zjarr qe dallohet trivialisht nga nje pyll me bore qe digjet. Negativat e
test_final jane 71.2% pyll me mjegull dhe tym — detyra e vertete. Prandaj as
specificity nuk transferohet nese llogaritet mbi te gjithe negativat.

Portat pershtaten vetem mbi NEGATIVAT E VESHTIRA (pyll), dhe liqeni raportohet
vecmas si kontroll. Kjo nuk e permireson modelin; e ndalon raportimin optimist.

Ekzekuto nga rrenja e projektit:
    python backend/training/fit_flame_gates.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "backend/artifacts"
DEFAULT_RUN = "fire_flame_v6"
DEFAULT_VALIDATION = "validation_flame_reviewed_v1"
DEFAULT_TEST_LABELS = PROJECT / "backend/artifacts/flame_label_audit/review_2026-10-13/test_final_flame_labels.csv"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def score(model, paths, size, batch=32):
    out = np.empty(len(paths), dtype=np.float32)
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            raw = tf.io.read_file(str(PROJECT / "dataset" / p))
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            imgs.append(tf.image.resize(tf.cast(img, tf.float32), size) / 255.0)
        out[i:i + len(imgs)] = model(tf.stack(imgs), training=False).numpy().ravel()
        if i % (batch * 25) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return out


def negative_source(path: str) -> str:
    """Burimi i nje negativi: liqeni eshte i lehte, pylli eshte detyra e vertete."""
    if "lake" in path:
        return "liqen (i lehte)"
    if path.startswith("Test/No_Fire"):
        return "pyll me mjegull/tym (i veshtire)"
    return "pyll tjeter (i veshtire)"


def per_source(paths, y, s, t):
    """Specificity sipas burimit te negativit, ne nje prag."""
    print(f"\n  specificity ne prag {t:.4f}, sipas burimit:")
    groups = {}
    for i, p in enumerate(paths):
        if y[i] == 0:
            groups.setdefault(negative_source(p), []).append(s[i] < t)
    for name in sorted(groups):
        v = groups[name]
        print(f"    {name:36} n={len(v):5}  spec={sum(v)/len(v):.4f}")


def fit_gates(y, s, min_recall=0.99, min_specificity=0.90):
    """low: pragu me i larte qe mban recall; high: me i ulet qe mban specificity."""
    cand = np.unique(np.round(s, 4))

    low = 0.0
    for t in cand:
        if (s[y == 1] >= t).mean() >= min_recall:
            low = float(t)

    high = float(cand.max())
    for t in cand:
        if (s[y == 0] >= t).mean() <= 1.0 - min_specificity:
            high = float(t); break

    if high <= low:
        # Zona e shqyrtimit do te zhdukej: kjo do te thote se modeli nuk i ndan
        # dot klasat mjaftueshem per dy porta te ndryshme. Raportohet, nuk fshihet.
        print(f"  KUJDES: high ({high:.4f}) <= low ({low:.4f}); zona e shqyrtimit "
              f"eshte bosh. Modeli nuk i ndan dot klasat me keto caqe.")
    return low, high


def operating_points(y, s):
    """Tabele pragjesh, qe te zgjidhet pika e punes me sy, jo me besim."""
    print(f"\n{'prag':>8} {'recall':>8} {'spec':>8} {'precision':>10} {'alarm %':>8}")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        p = s >= t
        prec = float(y[p].mean()) if p.any() else float("nan")
        print(f"{t:8.3f} {float(p[y == 1].mean()):8.4f} "
              f"{float((~p[y == 0]).mean()):8.4f} {prec:10.4f} {float(p.mean()):8.4f}")


def summarize(y, s, low, high):
    zone = np.where(s >= high, "alarm", np.where(s < low, "clear", "review"))
    out = {"n": int(len(y)), "fire": int(y.sum()),
           "auto_decision_share": float(np.mean(zone != "review")),
           "clear_share": float(np.mean(zone == "clear")),
           "alarm_share": float(np.mean(zone == "alarm"))}
    alarm = zone == "alarm"
    out["alarm_precision"] = float(y[alarm].mean()) if alarm.any() else None
    clear = zone == "clear"
    out["missed_fire_rate"] = float((y[clear] == 1).sum() / max(int(y.sum()), 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=DEFAULT_RUN, help="dosja nen backend/artifacts/")
    ap.add_argument("--validation", default=DEFAULT_VALIDATION, help="emri i split-it ne dataset/splits/")
    ap.add_argument("--test-labels", default=str(DEFAULT_TEST_LABELS),
                    help="CSV me path,flame_label vetem per raportim; 'none' e anashkalon")
    args = ap.parse_args()

    run_dir = ARTIFACTS / args.run
    model_path = run_dir / "model.keras"
    validation = PROJECT / "dataset/splits" / f"{args.validation}.csv"

    model = tf.keras.models.load_model(model_path, compile=False)
    size = tuple(model.input_shape[1:3])
    print(f"modeli: {args.run}/{model_path.name}  hyrja: {size}")

    rows = list(csv.DictReader(validation.open(encoding="utf-8-sig")))
    y = np.array([1 if r["label"] == "Fire" else 0 for r in rows])
    print(f"\nvalidation: {len(rows)} foto ({int(y.sum())} me flake)")
    s = score(model, [r["path"] for r in rows], size)

    print(f"perberja e validation: {100*y.mean():.1f}% me flake")
    operating_points(y, s)
    # Portat pershtaten mbi flaket + negativat e veshtira; liqeni perjashtohet.
    hard = np.array([not ("lake" in r["path"]) for r in rows])
    n_lake = int((~hard).sum())
    print(f"\nperjashtohen {n_lake} kuadro liqeni nga pershtatja e portave; "
          f"mbeten {int(hard.sum())} foto ({int(y[hard].sum())} me flake)")
    low, high = fit_gates(y[hard], s[hard])
    print(f"\nportat (mbi negativat e veshtira): low={low:.4f}  high={high:.4f}")
    per_source([r["path"] for r in rows], y, s, high)
    print("mbi validation:", json.dumps(summarize(y, s, low, high), indent=1))

    measured = None
    if args.test_labels.lower() != "none":
        test = [r for r in csv.DictReader(Path(args.test_labels).open(encoding="utf-8-sig"))
                if r["flame_label"] != "Uncertain"]
        yt = np.array([1 if r["flame_label"] == "Fire" else 0 for r in test])
        print(f"\ntest_final: {len(test)} foto ({int(yt.sum())} me flake, "
              f"{100*yt.mean():.1f}%)")
        st = score(model, [r["path"] for r in test], size)
        operating_points(yt, st)
        per_source([r["path"] for r in test], yt, st, high)
        measured = summarize(yt, st, low, high)
        measured["fire_prevalence"] = float(yt.mean())
        print("mbi test_final:", json.dumps(measured, indent=1))

    lake_share = n_lake / max(int((y == 0).sum()), 1)

    contract = {
        "model": "model.keras",
        "model_sha256": sha256(model_path),
        "labels": {"No_Fire": 0, "Fire": 1},
        "threshold": 0.5,
        "input_size": [int(size[0]), int(size[1]), 3],
        "input_range": [0, 1],
        "color": "RGB",
        "resize": "tf.image.resize bilinear; then /255 once",
        "output": "Sigmoid flame score; not a calibrated confidence.",
        "policy": "Fire = flake e dukshme. Tym pa flake eshte No_Fire.",
        "threshold_note": "Vendimi ne sherbim perdor 'gates', jo pragun 0.5.",
        "gates": {
            "low": round(low, 4), "high": round(high, 4),
            "fitted_on": (f"dataset/splits/{args.validation}.csv, pa kuadrot e "
                          f"liqenit ({int(hard.sum())} nga {len(rows)} foto)"),
            "rule": ("low = pragu me i larte qe mban recall >= 99% per flaken; "
                     "high = pragu me i ulet qe jep specificity >= 90% mbi jo-flaken. "
                     "Precision nuk perdoret si cak sepse varet nga perberja e setit. "
                     "Asnjera porte nuk u pershtat mbi test_final."),
            "zones": {"clear": "score < low", "review": "low <= score < high",
                      "alarm": "score >= high"},
        },
        "calibrated": False,
        "calibration_note": "Dalja eshte sigmoid, jo probabilitet i kalibruar.",
        "negative_composition_note": (
            f"Portat u pershtaten pa kuadrot e liqenit, qe jane {100*lake_share:.1f}% e "
            "negativave te validation dhe dallohen trivialisht. Specificity e llogaritur "
            "mbi te gjithe negativat e validation eshte optimiste dhe nuk transferohet."),
        "limitation": ("Etiketat e trajnimit u nxorren nga nje rregull (shih "
                       "flame_label_audit/), jo nga shqyrtim foto-per-foto. Vleresimi i "
                       "pavarur behet kundrejt kontrollit te verber njerezor. test_final "
                       "vjen nga te njejtat regjistrime si trajnimi, pra pergjithesimi ne "
                       "nje fluturim te ri mbetet i paprovuar."),
    }
    if measured is not None:
        contract["measured_on_test_final"] = measured
        contract["precision_note"] = (
            "alarm_precision varet nga sa foto me flake ka seti: "
            f"ky set ka {100*measured['fire_prevalence']:.1f}%. Nje set me perberje tjeter "
            "do te jepte precision tjeter me te njejtat porta.")
    out = run_dir / "inference.json"
    out.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nu shkrua: {out}")
    print(f"\nPer ta sherbyer kete model:  $env:FIRE_MODEL_RUN='{args.run}'")


if __name__ == "__main__":
    main()
