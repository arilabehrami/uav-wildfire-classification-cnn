"""Vendose ne backend/training/. Lexon modelin; nuk ben trajnim."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def make_preview(groups, data_dir, output):
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    columns, width, height = 4, 260, 292
    selected_groups = []
    for title, rows in groups:
        if rows:
            indices = np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=int)
            selected = [rows[index] for index in indices]
        else:
            selected = []
        selected_groups.append((title, len(rows), selected))

    total_height = 42 + sum(
        36 + max(1, math.ceil(len(rows) / columns)) * height
        for _, _, rows in selected_groups
    )
    canvas = Image.new("RGB", (columns * width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "Etiketa ne CSV: No_Fire | Pragu: 0.5", fill="black", font=font)
    top = 42

    for title, total, rows in selected_groups:
        draw.text((8, top + 8), f"{title} ({len(rows)}/{total})", fill="black", font=font)
        top += 36
        if not rows:
            draw.text((8, top + 10), "Nuk ka foto ne kete grup.", fill="black", font=font)
        for index, row in enumerate(rows):
            left = (index % columns) * width
            y = top + (index // columns) * height
            with Image.open(data_dir / row["path"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (248, 236))
            canvas.paste(thumb, (left + 6 + (248 - thumb.width) // 2, y + (236 - thumb.height) // 2))
            color = "#b91c1c" if row["predicted"] == "Fire" else "#166534"
            draw.rectangle((left + 2, y, left + width - 3, y + 238), outline=color, width=3)
            draw.text((left + 6, y + 242), Path(row["path"]).name, fill="black", font=font)
            draw.text((left + 6, y + 264), f"Fire score: {row['fire_score']:.4f} | {row['predicted']}", fill=color, font=font)
        top += max(1, math.ceil(len(rows) / columns)) * height

    canvas.save(output, quality=92)


def main():
    import tensorflow as tf
    from dataset_loader import DATA_DIR
    from finetune_mobilenet_v2 import ARTIFACTS, OUTPUT, load_split

    model = tf.keras.models.load_model(OUTPUT, compile=False)
    records = []

    for split in ("validation_extra_v2", "holdout_extra_v2"):
        rows, _, dataset = load_split(split)
        scores = model.predict(dataset, verbose=0).reshape(-1)
        if len(rows) != len(scores) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Dalje te pavlefshme per {split}.")
        for row, score in zip(rows, scores):
            if row["label"] == "No_Fire":
                records.append({
                    "split": split,
                    "path": row["path"],
                    "expected": "No_Fire",
                    "predicted": "Fire" if score >= 0.5 else "No_Fire",
                    "fire_score": float(score),
                })

    groups = [
        ("Validim shtese", [row for row in records if row["split"] == "validation_extra_v2"]),
        ("Fotot e rezervuara: alarme te gabuara", [row for row in records if row["split"] == "holdout_extra_v2" and row["predicted"] == "Fire"]),
        ("Fotot e rezervuara: klasifikime te sakta", [row for row in records if row["split"] == "holdout_extra_v2" and row["predicted"] == "No_Fire"]),
    ]
    output_dir = ARTIFACTS / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "no_fire_scores_v2.csv"
    image_path = output_dir / "no_fire_review_v2.jpg"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "path", "expected", "predicted", "fire_score"])
        writer.writeheader()
        writer.writerows(records)
    make_preview(groups, DATA_DIR, image_path)
    for title, rows in groups:
        print(f"{title}: {len(rows)} foto")
    print(f"Foto: {image_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
