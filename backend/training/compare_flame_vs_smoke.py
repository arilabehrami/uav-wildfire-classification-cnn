"""Krahason modelin e tymit (final_fire_v1) me modelin e flakes (fire_flame_v6).

Terreni: dataset/splits/test_final.csv, i rietiketuar sipas politikes se flakes
me te njejtin rregull te verifikuar (backend/artifacts/flame_label_audit/
review_2026-10-13/). test_final nuk eshte perdorur ne trajnimin e asnjerit
model dhe nuk mbivendoset me asnje split te ri.

Ekzekuto nga rrenja e projektit:
    python backend/training/compare_flame_vs_smoke.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[2]
LABELS = PROJECT / "backend/artifacts/flame_label_audit/review_2026-10-13/test_final_flame_labels.csv"
MODELS = {
    "tymi (final_fire_v1)": "backend/artifacts/final_fire_v1/model.keras",
    "flaka (fire_flame_v6)": "backend/artifacts/fire_flame_v6/model.keras",
}


def load_rows():
    with LABELS.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["flame_label"] != "Uncertain"]
    for r in rows:
        # Grupi diagnostik: pse nje foto eshte negative tani.
        if r["flame_label"] == "Fire":
            r["group"] = "A. flake e dukshme"
        elif r["old_label"] == "Fire":
            r["group"] = "B. tym pa flake (ish-Fire)"
        else:
            r["group"] = "C. pa zjarr fare"
    return rows


def scores_for(model_path: Path, rows, batch=32):
    model = tf.keras.models.load_model(model_path, compile=False)
    h, w = model.input_shape[1:3]
    out = np.empty(len(rows), dtype=np.float32)

    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        imgs = []
        for r in chunk:
            raw = tf.io.read_file(str(PROJECT / "dataset" / r["path"]))
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(tf.cast(img, tf.float32), (h, w)) / 255.0
            imgs.append(img)
        out[i:i + len(chunk)] = model(tf.stack(imgs), training=False).numpy().ravel()
        if i % (batch * 20) == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return out


def roc_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def report(name, rows, s):
    y = np.array([1 if r["flame_label"] == "Fire" else 0 for r in rows])
    groups = np.array([r["group"] for r in rows])

    print(f"\n{'='*64}\n{name}\n{'='*64}")
    print(f"ROC AUC (flake kundrejt jo-flake): {roc_auc(y, s):.4f}")

    best = None
    print(f"\n{'prag':>6} {'recall flake':>13} {'spec. jo-flake':>15} {'bal.acc':>9}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        p = s >= t
        rec = float(p[y == 1].mean())
        spec = float((~p[y == 0]).mean())
        bal = (rec + spec) / 2
        print(f"{t:6.2f} {rec:13.4f} {spec:15.4f} {bal:9.4f}")
        if best is None or bal > best[1]:
            best = (t, bal, rec, spec)

    print(f"\nme i miri: prag {best[0]:.2f}  bal.acc {best[1]:.4f} "
          f"(recall {best[2]:.4f}, spec {best[3]:.4f})")

    print(f"\n{'grupi':30} {'n':>5} {'score mes.':>11} {'% >= 0.5':>9}")
    for g in sorted(set(groups)):
        m = groups == g
        print(f"{g:30} {int(m.sum()):5} {float(s[m].mean()):11.4f} "
              f"{100*float((s[m] >= 0.5).mean()):8.1f}%")

    return {
        "roc_auc": roc_auc(y, s),
        "best_threshold": best[0],
        "best_balanced_accuracy": best[1],
        "fire_recall_at_best": best[2],
        "specificity_at_best": best[3],
        "by_group": {
            g: {"n": int((groups == g).sum()),
                "mean_score": float(s[groups == g].mean()),
                "share_above_0.5": float((s[groups == g] >= 0.5).mean())}
            for g in sorted(set(groups))
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "backend/artifacts/fire_flame_v6/comparison_test_final_flame.json")
    args = ap.parse_args()

    rows = load_rows()
    print(f"terreni: {len(rows)} foto "
          f"({sum(1 for r in rows if r['flame_label']=='Fire')} me flake)")

    results, preds = {}, {}
    for name, rel in MODELS.items():
        path = PROJECT / rel
        if not path.is_file():
            print(f"\nmungon modeli: {path}")
            continue
        print(f"\nduke vleresuar: {name}")
        s = scores_for(path, rows)
        preds[name] = s
        results[name] = report(name, rows, s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "test_set": "dataset/splits/test_final.csv, i rietiketuar sipas politikes se flakes",
        "n": len(rows),
        "fire": sum(1 for r in rows if r["flame_label"] == "Fire"),
        "note": ("test_final nuk u perdor ne trajnimin e asnjerit model dhe nuk "
                 "mbivendoset me split-et e reja. Fotot Uncertain jane perjashtuar."),
        "models": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nu shkrua: {args.out}")

    with (args.out.parent / "predictions_test_final_flame.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "flame_label", "group", *preds.keys()])
        for i, r in enumerate(rows):
            w.writerow([r["path"], r["flame_label"], r["group"],
                        *[f"{preds[k][i]:.6f}" for k in preds]])


if __name__ == "__main__":
    main()
