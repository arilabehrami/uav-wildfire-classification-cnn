import csv
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = DATA_DIR / "splits"

FOLDERS = {
    "Pre-Evening Fire Incident_raw_img": "Fire",
    "Pre-evening Forest condition_raw_img": "No_Fire",
}

GAP = 20  # Përjashtohen 20 foto para segmentit të kontrollit.


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return [
            (row["path"], row["label"])
            for row in csv.DictReader(file)
        ]


def frame_number(path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Emër i panjohur: {path.name}")
    return int(match.group(1))


def main():
    train = read_csv(OUTPUT_DIR / "train_temporal.csv")
    validation = read_csv(OUTPUT_DIR / "validation_temporal.csv")
    extra_validation = []

    for folder, label in FOLDERS.items():
        images = sorted(
            [
                path
                for path in (DATA_DIR / "extra_raw" / folder).rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=frame_number,
        )

        boundary = int(len(images) * 0.8)

        if boundary <= GAP or boundary >= len(images):
            raise ValueError(f"Nuk ka mjaft foto te: {folder}")

        def row(path):
            return (path.relative_to(DATA_DIR).as_posix(), label)

        new_train = [row(path) for path in images[:boundary - GAP]]
        new_validation = [row(path) for path in images[boundary:]]

        train.extend(new_train)
        extra_validation.extend(new_validation)

        print(
            f"{label}: {len(new_train)} shtohen në train, "
            f"{len(new_validation)} rezervohen, "
            f"{GAP} përjashtohen."
        )

    splits = {
        "train_mixed": train,
        "validation_mixed": validation,
        "validation_extra": extra_validation,
    }

    all_paths = []

    for name, rows in splits.items():
        paths = [path for path, _ in rows]

        if len(paths) != len(set(paths)):
            raise ValueError(f"Ka rrugë të përsëritura te {name}.")

        all_paths.extend(paths)

        counts = Counter(label for _, label in rows)
        if any(counts[label] == 0 for label in ["Fire", "No_Fire"]):
            raise ValueError(f"{name} nuk ka të dyja klasat.")

    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Ka foto të përbashkëta mes listave.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        output = OUTPUT_DIR / f"{name}.csv"

        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])
            writer.writerows(rows)

        print(
            f"{name}: {len(rows)} foto — "
            f"{dict(Counter(label for _, label in rows))}"
        )


if __name__ == "__main__":
    main()