import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

INPUT_CSV = (
    ARTIFACTS
    / "inspection"
    / "holdout_v3_predictions.csv"
)

OUTPUT_IMAGE = (
    ARTIFACTS
    / "inspection"
    / "holdout_v3_errors.jpg"
)

THUMB_SIZE = (220, 220)
TEXT_HEIGHT = 70
COLUMNS = 4


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Nuk u gjet CSV: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    errors = [
        row
        for row in rows
        if row["correct"].lower() == "false"
    ]

    if not errors:
        raise ValueError(
            "Nuk u gjet asnje gabim ne CSV."
        )

    print(
        f"Gabime totale: {len(errors)}"
    )

    rows_count = math.ceil(
        len(errors) / COLUMNS
    )

    cell_width = THUMB_SIZE[0]
    cell_height = THUMB_SIZE[1] + TEXT_HEIGHT

    sheet_width = COLUMNS * cell_width
    sheet_height = rows_count * cell_height

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            14,
        )
    except OSError:
        font = ImageFont.load_default()

    for index, row in enumerate(errors):
        image_path = (
            DATA_DIR / row["path"]
        )

        if not image_path.exists():
            print(
                f"Mungon: {image_path}"
            )
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        image.thumbnail(
            THUMB_SIZE
        )

        canvas = Image.new(
            "RGB",
            THUMB_SIZE,
            "black",
        )

        x_offset = (
            THUMB_SIZE[0]
            - image.width
        ) // 2

        y_offset = (
            THUMB_SIZE[1]
            - image.height
        ) // 2

        canvas.paste(
            image,
            (
                x_offset,
                y_offset,
            ),
        )

        col = index % COLUMNS
        row_index = index // COLUMNS

        x = col * cell_width
        y = row_index * cell_height

        sheet.paste(
            canvas,
            (x, y),
        )

        probability = float(
            row["fire_probability"]
        )

        filename = Path(
            row["path"]
        ).name

        text = (
            f"{filename}\n"
            f"True: {row['true_label']}\n"
            f"Pred: {row['predicted_label']} "
            f"({probability:.3f})"
        )

        draw.multiline_text(
            (
                x + 5,
                y + THUMB_SIZE[1] + 5,
            ),
            text,
            fill="black",
            font=font,
            spacing=2,
        )

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        OUTPUT_IMAGE,
        quality=95,
    )

    print(
        f"\nContact sheet u ruajt te:\n"
        f"{OUTPUT_IMAGE}"
    )


if __name__ == "__main__":
    main()