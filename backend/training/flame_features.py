"""Veçorite per çdo foto te review_manifest.csv: ngjyra e flakes dhe dHash.

    n       pixela me r >= g >= b dhe r - b > 40 (portokalli/kuqerremte)
    bright  ata prej tyre me r > 150
    dhash   dHash 64-bit (gri 9x8, krahasim horizontal), per filtrimin e
            fotove pothuajse te njejta mes train dhe validation

E rifillueshme: fotot qe jane tashme ne fajllin dales anashkalohen, dhe
--budget e ndal pas N sekondash. Ekzekuto disa here derisa te shkruaje DONE.

    python backend/training/flame_features.py
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / "backend/artifacts/flame_label_audit/review_manifest.csv"
OUT = PROJECT / "backend/artifacts/flame_label_audit/review_v2_2026-10-13/features.csv"
FIELDS = ["split", "path", "original_label", "n", "bright", "dhash"]


def color(a: np.ndarray):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r >= g) & (g >= b) & ((r - b) > 40)
    return int(m.sum()), int((m & (r > 150)).sum())


def dhash(img: Image.Image) -> int:
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--budget", type=float, default=float("inf"), help="sekonda per ekzekutim")
    args = ap.parse_args()

    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    done = set()
    if args.out.is_file():
        with args.out.open(encoding="utf-8", newline="") as f:
            done = {(r["split"], r["path"]) for r in csv.DictReader(f)}
    todo = [r for r in rows if (r["split"], r["path"]) not in done]
    print(f"mbeten {len(todo)} nga {len(rows)}", flush=True)
    if not todo:
        print("DONE"); return

    new = not args.out.is_file()
    start = time.time(); k = 0
    with args.out.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        for r in todo:
            img = Image.open(PROJECT / "dataset" / r["path"]).convert("RGB")
            n, bright = color(np.asarray(img, dtype=np.int16))
            w.writerow([r["split"], r["path"], r["original_label"], n, bright, dhash(img)])
            k += 1
            if k % 500 == 0:
                f.flush()
                if time.time() - start > args.budget:
                    break
    print(f"u shtuan {k}, mbeten {len(todo) - k}", flush=True)


if __name__ == "__main__":
    main()
