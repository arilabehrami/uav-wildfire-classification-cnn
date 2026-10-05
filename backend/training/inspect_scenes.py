from pathlib import Path
from PIL import Image, ImageDraw
import re

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = DATA_DIR / "inspection"


def frame_number(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label in ["Fire", "No_Fire"]:
        paths = sorted(
            (
                path
                for path in (DATA_DIR / "Training" / label).glob("*.jpg")
                if not path.name.startswith("lake_")
            ),
            key=frame_number,
        )

        if not paths:
            raise ValueError(f"Nuk ka foto për {label}.")

        count = min(12, len(paths))
        indices = [
            round(i * (len(paths) - 1) / max(count - 1, 1))
            for i in range(count)
        ]

        sheet = Image.new("RGB", (880, 720), "white")
        draw = ImageDraw.Draw(sheet)

        for position, index in enumerate(indices):
            path = paths[index]
            x = (position % 4) * 220
            y = (position // 4) * 240

            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((210, 210))
                sheet.paste(image, (x, y))

            draw.text((x, y + 212), path.name, fill="black")

        output = OUTPUT_DIR / f"scenes_{label}.jpg"
        sheet.save(output, quality=95)
        print(f"U krijua: {output}")


if __name__ == "__main__":
    main()