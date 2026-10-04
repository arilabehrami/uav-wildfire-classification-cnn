import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
SPLITS_DIR = DATA_DIR / "splits"


def main():
    with (SPLITS_DIR / "train_v2.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    extra_paths = sorted(
        (DATA_DIR / "extra_raw").rglob("*.png")
    )

    if not extra_paths:
        raise ValueError("Nuk u gjetën foto PNG te extra_raw.")

    existing_paths = {row["path"] for row in rows}
    added = 0

    for path in extra_paths:
        relative_path = path.relative_to(DATA_DIR).as_posix()

        if relative_path not in existing_paths:
            rows.append({
                "path": relative_path,
                "label": "No_Fire",
            })
            existing_paths.add(relative_path)
            added += 1

    output = SPLITS_DIR / "train_v5.csv"

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Foto të reja No_Fire: {added}")
    print(f"Gjithsej për trajnim: {len(rows)}")
    print(f"Lista u ruajt te: {output}")


if __name__ == "__main__":
    main()