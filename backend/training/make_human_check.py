"""Pergatit kontrollin e verber njerezor te etiketave mbi test_final.

Pse: etiketat e flakes, si per trajnim ashtu edhe per test_final, u nxorren
nga i njejti rregull ngjyre. Nese rregulli gabon, gabimet jane te njejta ne
te dy anet dhe testi e shperblen modelin qe i perserit. Nje njeri qe etiketon
me sy, pa pare as rregullin as modelin, jep etiketa te pavarura.

Mostra: 100 foto te rastesishme nga secila klase ORIGJINALE e test_final
(Fire / No_Fire te dataset-it, jo etiketat e rregullit), me fare te fiksuar,
pra e riprodhueshme. Rendi perzihet dhe emrat e fajllave fshihen, sepse emri
'resized_test_fire_frame...' e zbulon klasen origjinale.

Krijon ne <grupi>/:
    etiketo.html   faqja e etiketimit (fotot brenda saj)
    key.csv        ID -> fajlli; MOS E HAP para etiketimit

Ekzekuto nga rrenja e projektit:
    python backend/training/make_human_check.py
    python backend/training/make_human_check.py --batch human_check_2_2026-10-13 \
        --seed 20261014 --prefix G --exclude human_check_2026-10-13

Grupi i dyte perjashton fotot e te parit: rregulli v2 u zgjodh mbi grupin e
pare, prandaj vleresimi perfundimtar kerkon pamje qe nuk jane pare.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "backend/artifacts/flame_label_audit"
RULE_LABELS = PROJECT / "backend/artifacts/flame_label_audit/review_2026-10-13/test_final_flame_labels.csv"
PER_CLASS = 100

PAGE = Path(__file__).with_name("human_check_page.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="human_check_2026-10-13")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--prefix", default="F", help="shkronja e ID-ve, p.sh. F001")
    ap.add_argument("--exclude", nargs="*", default=[], help="grupe te meparshme, fotot e te cileve perjashtohen")
    args = ap.parse_args()
    out = AUDIT / args.batch
    if (out / "etiketat_njerezore.csv").exists():
        raise SystemExit(f"{out} ka tashme etiketa njerezore; nuk mbishkruhet.")

    excluded = set()
    for prev in args.exclude:
        with (AUDIT / prev / "key.csv").open(encoding="utf-8") as f:
            excluded |= {r["path"] for r in csv.DictReader(f)}

    with (PROJECT / "dataset/splits/test_final.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with RULE_LABELS.open(encoding="utf-8") as f:
        rule = {r["path"]: r["flame_label"] for r in csv.DictReader(f)}

    rng = random.Random(args.seed)
    picked = []
    for label in ("Fire", "No_Fire"):
        pool = sorted(r["path"] for r in rows if r["label"] == label and r["path"] not in excluded)
        picked += [(path, label) for path in rng.sample(pool, PER_CLASS)]
    rng.shuffle(picked)

    out.mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for i, (path, original) in enumerate(picked, start=1):
        item_id = f"{args.prefix}{i:03d}"
        data = (PROJECT / "dataset" / path).read_bytes()
        items.append({"id": item_id,
                      "src": "data:image/jpeg;base64," + base64.b64encode(data).decode()})
        key.append({"id": item_id, "path": path, "original_label": original,
                    "rule_label": rule.get(path, "")})

    with (out / "key.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "path", "original_label", "rule_label"])
        w.writeheader()
        w.writerows(key)

    html = (PAGE.read_text(encoding="utf-8")
            .replace("__ITEMS__", json.dumps(items))
            .replace("__STORE__", f"flame-{args.batch}")
            .replace("__BATCH__", args.batch))
    (out / "etiketo.html").write_text(html, encoding="utf-8")

    (out / "README.md").write_text(
        "# Kontrolli i verber njerezor\n\n"
        f"Mostra: {PER_CLASS} pamje te rastesishme nga secila klase origjinale e test_final, "
        f"fara {args.seed}"
        + (f", pa pamjet e grupeve {', '.join(args.exclude)}" if args.exclude else "")
        + ". Krijuar nga `backend/training/make_human_check.py`.\n\n"
        "1. Hap `etiketo.html` ne shfletues dhe etiketo te 200 pamjet.\n"
        "2. **Mos e hap `key.csv` para se te perfundosh** — e zbulon klasen.\n"
        "3. Shkarko CSV-ne dhe vendose ketu si `etiketat_njerezore.csv`.\n"
        f"4. Nga rrenja: `python backend/training/evaluate_human_labels.py --batch {args.batch}`\n",
        encoding="utf-8")

    print(f"u krijua: {out / 'etiketo.html'}  ({len(items)} pamje, perjashtuar {len(excluded)})")
    print(f"çelesi:   {out / 'key.csv'}  (mos e hap para etiketimit)")


if __name__ == "__main__":
    main()
