"""Etiketat dhe split-et e flakes, versioni 2.

Rregulli v2, i zgjedhur mbi grupin e pare te kontrollit te verber njerezor
(human_check_2026-10-13, 159 pamje te sigurta), mes 19 kandidateve:

    Fire  =  etiketa origjinale eshte Fire  DHE  n >= 1
             (te pakten nje piksel me r >= g >= b dhe r - b > 40)
    No_Fire  ne çdo rast tjeter

Kundrejt njeriut: recall 0.967, precision 0.952, specificity 0.969. Rregulli v1
(bright >= 16) kishte recall 0.344: i humbi 40 nga 61 flaket qe pa njeriu.
Meqe rregulli u zgjodh mbi ate grup, vleresimi perfundimtar behet mbi nje grup
te dyte te verber, te pavarur.

extra_raw/ mbetet i perjashtuar si ne v1: zjarr i kontrolluar me dru mbi bar,
domen tjeter, dhe ngjyra roze e tokes e ndez statistiken.

Ndarja: train_with_adapt -> train; te gjitha te tjerat -> validation (si v1).
Pastaj hiqen nga validation fotot me dHash Hamming <= 8 ndaj ndonje foto te train.

v1 nuk preket. Shkruan:
    dataset/splits/train_flame_reviewed_v2.csv
    dataset/splits/validation_flame_reviewed_v2.csv
    backend/artifacts/flame_label_audit/review_v2_2026-10-13/labels_v2.csv
    backend/artifacts/flame_label_audit/review_v2_2026-10-13/split_report_v2.json

    python backend/training/prepare_flame_splits_v2.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "backend/artifacts/flame_label_audit/review_v2_2026-10-13"
FEATURES = AUDIT / "features.csv"
SPLITS = PROJECT / "dataset/splits"
THRESHOLD = 8
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def label_v2(row) -> tuple[str, str]:
    if row["path"].startswith("extra_raw/"):
        return "Uncertain", "extra_raw: domen tjeter, i perjashtuar si ne v1"
    if row["original_label"] != "Fire":
        return "No_Fire", "etiketa origjinale No_Fire"
    if int(row["n"]) >= 1:
        return "Fire", f"origjinale Fire dhe {row['n']} pixela portokalli"
    return "No_Fire", "origjinale Fire por asnje piksel portokalli: pa gjurme flake"


def packed(hashes):
    return np.array(hashes, dtype=np.uint64).view(np.uint8).reshape(-1, 8)


def min_distance(query, base, chunk=150):
    out = np.empty(len(query), dtype=np.uint8)
    for i in range(0, len(query), chunk):
        block = query[i:i + chunk]
        out[i:i + chunk] = POPCOUNT[np.bitwise_xor(block[:, None, :], base[None, :, :])].sum(2).min(1)
    return out


def main():
    with FEATURES.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labelled = []
    for r in rows:
        label, reason = label_v2(r)
        labelled.append({**r, "label_v2": label, "reason": reason})

    with (AUDIT / "labels_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "path", "original_label", "n", "bright",
                                          "label_v2", "reason"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in labelled)

    usable = [r for r in labelled if r["label_v2"] != "Uncertain"]
    train = [r for r in usable if r["split"] == "train_with_adapt"]
    valid = [r for r in usable if r["split"] != "train_with_adapt"]

    dist = min_distance(packed([int(r["dhash"]) for r in valid]),
                        packed([int(r["dhash"]) for r in train]))
    kept = [r for r, d in zip(valid, dist) if d > THRESHOLD]

    for name, part in (("train_flame_reviewed_v2", train), ("validation_flame_reviewed_v2", kept)):
        with (SPLITS / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label"])
            w.writeheader()
            w.writerows({"path": r["path"], "label": r["label_v2"]} for r in part)

    changes = Counter((r["original_label"], r["label_v2"]) for r in labelled)
    report = {
        "rregulli": "Fire = origjinale Fire DHE n >= 1; extra_raw perjashtohet",
        "zgjedhur_mbi": "human_check_2026-10-13 (159 pamje te sigurta)",
        "nga_ne": {f"{a} -> {b}": n for (a, b), n in sorted(changes.items())},
        "train": dict(Counter(r["label_v2"] for r in train)),
        "validation_para_filtrimit": dict(Counter(r["label_v2"] for r in valid)),
        "validation": dict(Counter(r["label_v2"] for r in kept)),
        "hequr_nga_dhash": len(valid) - len(kept),
        "dhash_prag": THRESHOLD,
        "test_final_prekur": False,
    }
    (AUDIT / "split_report_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
