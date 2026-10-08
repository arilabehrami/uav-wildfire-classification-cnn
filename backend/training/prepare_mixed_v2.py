import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
SPLIT_DIR = DATA_DIR / "splits"

GAP = 20
LABELS = {"Fire", "No_Fire"}


def read_split(name):
    with (SPLIT_DIR / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows or any(row["label"] not in LABELS for row in rows):
        raise ValueError(f"Lista bosh ose etiketa te panjohura: {name}")

    paths = [row["path"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError(f"Rruge te perseritura: {name}")

    return [
        {"path": row["path"], "label": row["label"]}
        for row in rows
    ]


def frame_number(row):
    match = re.search(r"(\d+)$", Path(row["path"]).stem)

    if match is None:
        raise ValueError(f"Emri nuk ka numer frame: {row['path']}")

    return int(match.group(1))


def main():
    old_train = read_split("train_temporal")
    old_validation = read_split("validation_temporal")
    mixed_train = read_split("train_mixed")
    holdout = read_split("validation_extra")

    old_map = {
        row["path"]: row["label"]
        for row in old_train
    }
    mixed_map = {
        row["path"]: row["label"]
        for row in mixed_train
    }

    if any(
        mixed_map.get(path) != label
        for path, label in old_map.items()
    ):
        raise ValueError(
            "train_mixed nuk permban sakte train_temporal."
        )

    if any(
        Path(path).parts[0] == "extra_raw"
        for path in old_map
    ):
        raise ValueError(
            "train_temporal tashme ka extra_raw; "
            "ndarja duhet rishikuar."
        )

    groups = defaultdict(list)
    folder_labels = defaultdict(set)

    for row in mixed_train:
        if row["path"] in old_map:
            continue

        path = Path(row["path"])

        if path.parts[0] != "extra_raw":
            raise ValueError(f"Burim shtese i papritur: {path}")

        folder = path.parent.as_posix()
        groups[folder].append(row)
        folder_labels[folder].add(row["label"])

    if not groups or any(
        len(labels) != 1
        for labels in folder_labels.values()
    ):
        raise ValueError(
            "Burimet shtese mungojne ose kane etiketa te perziera."
        )

    train = list(old_train)
    extra_validation = []
    excluded = []

    for folder, rows in sorted(groups.items()):
        rows = sorted(rows, key=frame_number)
        numbers = [frame_number(row) for row in rows]

        if len(numbers) != len(set(numbers)):
            raise ValueError(
                f"Numra frame te perseritur ne {folder}"
            )

        boundary = int(len(rows) * 0.8)

        if boundary <= GAP or boundary >= len(rows):
            raise ValueError(
                f"Nuk ka mjaft foto per train, gap dhe validation: "
                f"{folder}"
            )

        new_train = rows[:boundary - GAP]
        new_validation = rows[boundary:]

        train.extend(new_train)
        extra_validation.extend(new_validation)
        excluded.extend(rows[boundary - GAP:boundary])

        print(
            f"{folder}: train={len(new_train)}, "
            f"validation={len(new_validation)}, gap={GAP}"
        )

    # Fotot e rezervuara mbeten te njejtat.
    outputs = {
        "train_mixed_v2": train,
        "validation_extra_v2": extra_validation,
        "holdout_extra_v2": holdout,
    }

    checks = {
        **outputs,
        "validation_temporal": old_validation,
    }

    seen = set()

    for name, rows in checks.items():
        paths = [row["path"] for row in rows]

        if (
            len(paths) != len(set(paths))
            or seen.intersection(paths)
        ):
            raise ValueError(
                f"Foto te perseritura ose te perbashketa ne {name}"
            )

        if set(row["label"] for row in rows) != LABELS:
            raise ValueError(f"Mungon njera klase ne {name}")

        seen.update(paths)

    if seen.intersection(row["path"] for row in excluded):
        raise ValueError(
            "Fotot e gap-it jane perfshire ne ndonje liste."
        )

    destinations = {
        name: SPLIT_DIR / f"{name}.csv"
        for name in outputs
    }

    if any(path.exists() for path in destinations.values()):
        raise FileExistsError(
            "Ekziston nje liste v2. Nuk u ndryshua asnje liste."
        )

    for name, rows in outputs.items():
        with destinations[name].open(
            "x", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["path", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)

        counts = Counter(row["label"] for row in rows)

        print(
            f"{name}: {len(rows)} foto; {dict(counts)}"
        )

    print(
        "Ndarja perfundoi. "
        "Listat e meparshme mbeten te pandryshuara."
    )


if __name__ == "__main__":
    main()