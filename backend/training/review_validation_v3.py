"""Vendose ne backend/training/. Krijon pamje nga rezultatet e ruajtura."""

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
ARTIFACTS = ROOT / "backend" / "artifacts"
SCORES = ARTIFACTS / "metrics/validation_audit_v3/validation_predictions_v3.csv"
OUTPUT = ARTIFACTS / "inspection/validation_errors_v3.jpg"


def select_examples(rows, limit=8):
    """Mbulon intervalin e rezultateve; perfshin edhe skajet."""
    if len(rows) <= limit:
        return rows
    indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]
    return [rows[index] for index in indices]


def error_groups(rows):
    groups = []
    for split, source_title in (
        ("validation_temporal", "Validimi baze"),
        ("validation_extra_v3", "Validimi shtese"),
    ):
        for label, error_title in (
            ("Fire", "Fire -> No_Fire"),
            ("No_Fire", "No_Fire -> Fire"),
        ):
            errors = [
                row for row in rows
                if row["split"] == split
                and row["label"] == label
                and (row["fire_score"] >= 0.5) != (label == "Fire")
            ]
            errors.sort(key=lambda row: row["fire_score"], reverse=(label == "No_Fire"))
            groups.append((f"{source_title}: {error_title}", errors))
    return groups


def make_preview(groups):
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        title_font = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        font = title_font = ImageFont.load_default()

    columns, cell_width, cell_height = 4, 296, 305
    selected = [(title, len(rows), select_examples(rows)) for title, rows in groups]
    height = 62 + sum(42 + math.ceil(len(rows) / columns) * cell_height for _, _, rows in selected)
    canvas = Image.new("RGB", (columns * cell_width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), "V3: shembuj gabimesh ne validim | pragu 0.5", font=title_font, fill="black")
    draw.text((10, 34), "Etiketat jane marre nga CSV-ja; kontrollojme permbajtjen vizuale.", font=font, fill="black")
    top = 62

    for title, total, rows in selected:
        draw.text((10, top + 8), f"{title}: {total} gjithsej, shfaqen {len(rows)}", font=title_font, fill="black")
        top += 42
        for index, row in enumerate(rows):
            x = (index % columns) * cell_width
            y = top + (index // columns) * cell_height
            with Image.open(DATA_DIR / row["path"]) as original:
                thumb = ImageOps.contain(original.convert("RGB"), (cell_width - 12, 232))
            canvas.paste(thumb, (x + (cell_width - thumb.width) // 2, y + (232 - thumb.height) // 2))
            draw.rectangle((x + 2, y, x + cell_width - 3, y + 234), outline="#b91c1c", width=2)
            filename = Path(row["path"]).name
            if len(filename) > 35:
                filename = "..." + filename[-32:]
            predicted = "Fire" if row["fire_score"] >= 0.5 else "No_Fire"
            draw.text((x + 6, y + 240), filename, font=font, fill="black")
            draw.text((x + 6, y + 260), f"Etiketa: {row['label']} | Dalja: {predicted}", font=font, fill="black")
            draw.text((x + 6, y + 280), f"Fire score: {row['fire_score']:.6f}", font=font, fill="black")
        top += math.ceil(len(rows) / columns) * cell_height
    return canvas


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Pamja ekziston: {OUTPUT}")
    with SCORES.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("CSV-ja e rezultateve eshte bosh.")
    for row in rows:
        row["fire_score"] = float(row["fire_score"])
        if row["label"] not in {"Fire", "No_Fire"} or row["split"] not in {"validation_temporal", "validation_extra_v3"}:
            raise ValueError("U gjet nje etikete ose liste e panjohur ne CSV.")
        if not math.isfinite(row["fire_score"]) or not 0 <= row["fire_score"] <= 1:
            raise ValueError("Fire score i pavlefshem ne CSV.")

    groups = error_groups(rows)
    for title, errors in groups:
        print(f"{title}: {len(errors)} gabime")
    preview = make_preview(groups)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    preview.save(OUTPUT, quality=95)
    print(f"\nPamja: {OUTPUT}")
    print("U perdoren rezultatet e ruajtura. Modeli dhe dataset-i mbeten te pandryshuar.")


if __name__ == "__main__":
    main()
