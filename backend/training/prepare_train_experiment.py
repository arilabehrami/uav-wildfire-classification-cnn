from pathlib import Path
from collections import Counter
import csv
import random

SPLITS = Path(__file__).resolve().parents[2] / "dataset" / "splits"


def main():
    with (SPLITS / "train.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    fire = [r for r in rows if r["label"] == "Fire"]

    lake = [
        r for r in rows
        if r["label"] == "No_Fire"
        and Path(r["path"]).name.startswith("lake_")
    ]

    other_no_fire = [
        r for r in rows
        if r["label"] == "No_Fire"
        and not Path(r["path"]).name.startswith("lake_")
    ]

    if not other_no_fire:
        raise ValueError("Nuk u gjetën imazhe No_Fire pa prefiks lake_.")

    rng = random.Random(42)
    selected_lake = rng.sample(
        lake, min(len(lake), len(other_no_fire))
    )

    selected = fire + other_no_fire + selected_lake
    rng.shuffle(selected)

    with (SPLITS / "train_experiment.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(selected)

    print("Imazhe liqeni të përzgjedhura:", len(selected_lake))
    print("No_Fire të tjera:", len(other_no_fire))
    print("Gjithsej:", dict(Counter(r["label"] for r in selected)))


if __name__ == "__main__":
    main()