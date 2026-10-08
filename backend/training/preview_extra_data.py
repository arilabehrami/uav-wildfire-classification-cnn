from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageOps

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = DATA_DIR / "inspection"


def number(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def main():
    folders = sorted((DATA_DIR / "extra_raw").iterdir())
    folders = [folder for folder in folders if folder.is_dir()]

    width, height = 220, 220
    sheet = Image.new(
        "RGB", (width * 4, len(folders) * (height * 2 + 40)), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for group, folder in enumerate(folders):
        images = sorted(
            [
                path for path in folder.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=number,
        )

        if not images:
            raise ValueError(f"Dosja është bosh: {folder.name}")

        top = group * (height * 2 + 40)
        draw.text((5, top + 5), folder.name, fill="black")

        # Tetë foto të shpërndara përgjatë listës.
        indices = sorted({
            round(i * (len(images) - 1) / 7)
            for i in range(8)
        })

        for position, index in enumerate(indices):
            path = images[index]
            x = (position % 4) * width
            y = top + 40 + (position // 4) * height

            with Image.open(path) as image:
                thumbnail = ImageOps.contain(
                    image.convert("RGB"), (width, height - 25)
                )

            sheet.paste(thumbnail, (x, y))
            draw.text(
                (x + 5, y + height - 22),
                path.name,
                fill="black",
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "extra_raw_preview.jpg"
    sheet.save(output, quality=90)

    print(f"Pamjet u ruajtën te: {output}")


if __name__ == "__main__":
    main()