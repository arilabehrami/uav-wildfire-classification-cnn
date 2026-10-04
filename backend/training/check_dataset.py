from pathlib import Path
from collections import Counter
from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    total_bad = 0

    for split in ["Training", "Test"]:
        for label in ["Fire", "No_Fire"]:
            folder = DATA_DIR / split / label

            if not folder.is_dir():
                print(f"Mungon folderi: {folder}")
                continue

            files = [
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            ]

            sizes = Counter()
            bad = 0

            for path in files:
                try:
                    with Image.open(path) as img:
                        img.load()
                        sizes[img.size] += 1
                except Exception as error:
                    bad += 1
                    print(f"Imazh i dëmtuar: {path.name}: {error}")

            total_bad += bad
            print(f"\n{split}/{label}")
            print(f"Numri i imazheve: {len(files)}")
            print(f"Përmasat dhe numri për secilën: {dict(sizes)}")
            print(f"Të dëmtuara: {bad}")

    print(f"\nGjithsej imazhe të dëmtuara: {total_bad}")


if __name__ == "__main__":
    main()