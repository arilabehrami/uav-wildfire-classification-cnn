"""Nxjerr pamje nga gabimet e ruajtura, te grupuara sipas segmentit te fluturimit.

Pyetja: cfare ka ne fotot No_Fire te Test-it qe modeli i lexon si zjarr, dhe
pse njeri segment del thuajse i pastër? Kjo skripta i ve pergjithe-perqark.

Lexon vetem CSV-te e parashikimeve. Nuk ngarkon TensorFlow, nuk trajnon.

    python inspect_test_negatives.py
    python inspect_test_negatives.py --run final_fire_v1 --bands 12 --per-band 6
"""

import argparse
import csv
import math
import re
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

TILE = (300, 300)
CAPTION = 62
COLUMNS = 6


def font(size):
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_number(path):
    match = re.search(r"(\d+)$", PurePosixPath(path).stem)
    return int(match.group(1)) if match else None


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV bosh: {path}")
    for row in rows:
        row["fire_score"] = float(row["fire_score"])
        row["frame"] = frame_number(row["path"])
        if row["label"] not in ("Fire", "No_Fire"):
            raise ValueError(f"Etikete e panjohur: {row['label']}")
    return rows


def spread(rows, limit):
    """Shembuj te shperndare, jo vetem te parët."""
    if not rows:
        return []
    if len(rows) <= limit:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[index] for index in indices]


def make_bands(rows, bands):
    negatives = [row for row in rows if row["label"] == "No_Fire" and row["frame"] is not None]
    if len(negatives) < bands * 2:
        return []
    negatives.sort(key=lambda row: row["frame"])
    edges = np.unique(
        np.quantile([row["frame"] for row in negatives], np.linspace(0, 1, bands + 1))
    )
    output = []
    for start, end in zip(edges[:-1], edges[1:]):
        last = end == edges[-1]
        members = [
            row for row in negatives
            if row["frame"] >= start and (row["frame"] <= end if last else row["frame"] < end)
        ]
        if not members:
            continue
        errors = sum(row["fire_score"] >= 0.5 for row in members)
        output.append({
            "range": (int(start), int(end)),
            "rows": members,
            "count": len(members),
            "false_positives": errors,
            "rate": errors / len(members),
        })
    return output


def draw_group(canvas, draw, title, subtitle, rows, data_root, top, warnings):
    draw.text((12, top + 6), title, font=font(23), fill="black")
    draw.text((12, top + 34), subtitle, font=font(16), fill="#444444")
    top += 62

    for index, row in enumerate(rows):
        x = (index % COLUMNS) * TILE[0]
        y = top + (index // COLUMNS) * (TILE[1] + CAPTION)
        source = data_root / row["path"]

        if not source.is_file():
            warnings.append(f"mungon: {row['path']}")
            draw.rectangle((x + 4, y, x + TILE[0] - 6, y + TILE[1]), outline="#999999")
            draw.text((x + 10, y + 10), "mungon", font=font(15), fill="#999999")
            continue

        with Image.open(source) as original:
            size = original.size
            thumbnail = ImageOps.contain(original.convert("RGB"), (TILE[0] - 12, TILE[1] - 8))

        canvas.paste(thumbnail, (x + 6 + (TILE[0] - 12 - thumbnail.width) // 2, y))

        predicted_fire = row["fire_score"] >= 0.5
        correct = predicted_fire == (row["label"] == "Fire")
        color = "#166534" if correct else "#b91c1c"
        draw.rectangle((x + 4, y, x + TILE[0] - 6, y + TILE[1]), outline=color, width=3)

        name = PurePosixPath(row["path"]).name
        draw.text((x + 8, y + TILE[1] + 4), name[:34], font=font(14), fill="black")
        draw.text(
            (x + 8, y + TILE[1] + 23),
            f"etiketa {row['label']} -> {'Fire' if predicted_fire else 'No_Fire'}",
            font=font(15), fill=color,
        )
        draw.text(
            (x + 8, y + TILE[1] + 42),
            f"score {row['fire_score']:.4f}   {size[0]}x{size[1]}",
            font=font(14), fill="#444444",
        )

    rows_used = max(1, math.ceil(len(rows) / COLUMNS))
    return top + rows_used * (TILE[1] + CAPTION) + 18


def build(groups, data_root, output, header):
    height = 54 + sum(
        62 + max(1, math.ceil(len(rows) / COLUMNS)) * (TILE[1] + CAPTION) + 18
        for _, _, rows in groups
    )
    canvas = Image.new("RGB", (COLUMNS * TILE[0], height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), header, font=font(26), fill="black")

    top = 54
    warnings = []
    for title, subtitle, rows in groups:
        top = draw_group(canvas, draw, title, subtitle, rows, data_root, top, warnings)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--per-band", type=int, default=6)
    args = parser.parse_args()

    project = args.project_root.resolve()
    run_dir = project / "backend" / "artifacts" / args.run
    data_root = project / "dataset"
    if not run_dir.is_dir():
        raise SystemExit(f"Nuk ekziston: {run_dir}")

    test = load(run_dir / "known_test_predictions.csv")
    bands = make_bands(test, args.bands)
    if not bands:
        raise SystemExit("Nuk u ndertuan banda; emrat e fajllave nuk kane numra frame.")

    cleanest = min(bands, key=lambda band: band["rate"])
    worst = sorted(bands, key=lambda band: band["rate"], reverse=True)[:3]

    negatives = [row for row in test if row["label"] == "No_Fire"]
    positives = [row for row in test if row["label"] == "Fire"]
    false_positives = sorted(
        (row for row in negatives if row["fire_score"] >= 0.5),
        key=lambda row: row["fire_score"], reverse=True,
    )
    false_negatives = sorted(
        (row for row in positives if row["fire_score"] < 0.5),
        key=lambda row: row["fire_score"],
    )

    # --- Flete 1: kontrasti qe vendos diagnozen
    groups = [
        (
            f"A. Segmenti me i pastër: frame {cleanest['range'][0]}-{cleanest['range'][1]}",
            f"{cleanest['false_positives']}/{cleanest['count']} alarme te rreme "
            f"({cleanest['rate']:.1%}). Keto i trajton sakte.",
            spread(cleanest["rows"], COLUMNS),
        )
    ]
    for band in worst:
        groups.append((
            f"B. Segment problematik: frame {band['range'][0]}-{band['range'][1]}",
            f"{band['false_positives']}/{band['count']} alarme te rreme "
            f"({band['rate']:.1%}). Krahaso me rreshtin A.",
            spread(band["rows"], COLUMNS),
        ))
    groups.append((
        "C. Alarmet me te sigurta te rreme",
        f"No_Fire me score-t me te larte, nga {len(false_positives)} alarme te rreme gjithsej.",
        false_positives[:COLUMNS],
    ))
    if false_negatives:
        groups.append((
            "D. Zjarre te humbura",
            f"Fire me score-t me te uleta, nga {len(false_negatives)} te humbura gjithsej.",
            false_negatives[:COLUMNS],
        ))

    sheet_a = run_dir / "inspection" / "test_contrast.jpg"
    warnings = build(
        groups, data_root, sheet_a,
        f"known_test — kontrasti mes segmentit te pastër dhe atyre problematike ({args.run})",
    )

    # --- Flete 2: cdo bande, per te pare kalimin
    band_groups = [
        (
            f"frame {band['range'][0]}-{band['range'][1]}",
            f"{band['false_positives']}/{band['count']} alarme te rreme ({band['rate']:.1%})",
            spread(band["rows"], args.per_band),
        )
        for band in bands
    ]
    sheet_b = run_dir / "inspection" / "test_bands.jpg"
    warnings += build(
        band_groups, data_root, sheet_b,
        f"known_test — fotot No_Fire sipas segmentit te fluturimit ({args.run})",
    )

    # --- Flete 3: e njejta dobesi ne validation
    sheet_c = None
    validation_path = run_dir / "validation_predictions.csv"
    if validation_path.is_file():
        validation = load(validation_path)
        validation_bands = make_bands(validation, args.bands)
        if validation_bands:
            top_band = max(validation_bands, key=lambda band: band["rate"])
            errors = sorted(
                (row for row in top_band["rows"] if row["fire_score"] >= 0.5),
                key=lambda row: row["fire_score"], reverse=True,
            )
            others = [
                row for band in validation_bands if band is not top_band
                for row in band["rows"] if row["fire_score"] >= 0.5
            ]
            sheet_c = run_dir / "inspection" / "validation_weak_band.jpg"
            warnings += build(
                [
                    (
                        f"Segmenti i dobët i validimit: frame "
                        f"{top_band['range'][0]}-{top_band['range'][1]}",
                        f"{top_band['false_positives']}/{top_band['count']} "
                        f"({top_band['rate']:.1%}). A ngjajne me rreshtin B te fletes tjeter?",
                        spread(errors, COLUMNS),
                    ),
                    (
                        "Alarme te rreme ne pjesen tjeter te validimit",
                        f"{len(others)} gjithsej jashte atij segmenti.",
                        spread(others, COLUMNS),
                    ),
                ],
                data_root, sheet_c,
                f"validation — ku grumbullohen alarmet e rreme ({args.run})",
            )

    print(f"\nknown_test: {len(negatives)} No_Fire, {len(false_positives)} alarme te rreme "
          f"({len(false_positives) / len(negatives):.1%})")
    print(f"Segmenti me i pastër: frame {cleanest['range'][0]}-{cleanest['range'][1]} "
          f"-> {cleanest['rate']:.1%}")
    print("Segmentet me te keqija:")
    for band in worst:
        print(f"  frame {band['range'][0]}-{band['range'][1]}: {band['rate']:.1%}")

    print(f"\nFletet:\n  {sheet_a}\n  {sheet_b}")
    if sheet_c:
        print(f"  {sheet_c}")
    if warnings:
        print(f"\nVerejtje ({len(warnings)}):")
        for item in warnings[:10]:
            print(f"  {item}")
    print("\nAsnje model, etikete ose prag nuk u ndryshua.")


if __name__ == "__main__":
    main()