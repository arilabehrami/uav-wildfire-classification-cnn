import csv
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
SPLIT_DIR = DATA_DIR / "splits"

# Dosja e re No_Fire që nuk është përdorur në v2
NEW_NO_FIRE_DIR = (
    DATA_DIR
    / "extra_raw"
    / "Evening Forest condition_raw_img"
)

GAP = 20
TRAIN_RATIO = 0.8
LABELS = {"Fire", "No_Fire"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_split(name):
    path = SPLIT_DIR / f"{name}.csv"

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Lista eshte bosh: {name}")

    if any(row["label"] not in LABELS for row in rows):
        raise ValueError(
            f"Etiketa te panjohura ne {name}"
        )

    paths = [row["path"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError(
            f"Rruge te perseritura ne {name}"
        )

    return [
        {
            "path": row["path"],
            "label": row["label"]
        }
        for row in rows
    ]


def frame_number(path):
    match = re.search(r"(\d+)$", Path(path).stem)

    if match is None:
        raise ValueError(
            f"Emri nuk ka numer frame: {path}"
        )

    return int(match.group(1))


def load_new_no_fire():
    if not NEW_NO_FIRE_DIR.exists():
        raise FileNotFoundError(
            f"Nuk ekziston dosja:\n{NEW_NO_FIRE_DIR}"
        )

    files = [
        path
        for path in NEW_NO_FIRE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )
    ]

    if not files:
        raise ValueError(
            "Nuk u gjet asnje foto ne "
            "Evening Forest condition_raw_img."
        )

    files = sorted(
        files,
        key=lambda path: frame_number(path)
    )

    numbers = [
        frame_number(path)
        for path in files
    ]

    if len(numbers) != len(set(numbers)):
        raise ValueError(
            "Ka numra frame te perseritur "
            "ne dosjen e re."
        )

    rows = []

    for path in files:
        relative_path = path.relative_to(DATA_DIR)

        rows.append(
            {
                "path": relative_path.as_posix(),
                "label": "No_Fire"
            }
        )

    return rows


def check_no_overlap(named_splits):
    seen = set()

    for name, rows in named_splits.items():
        paths = [row["path"] for row in rows]

        if len(paths) != len(set(paths)):
            raise ValueError(
                f"Ka foto te perseritura brenda {name}"
            )

        overlap = seen.intersection(paths)

        if overlap:
            example = next(iter(overlap))

            raise ValueError(
                f"Ka foto te perbashketa ne {name}: "
                f"{example}"
            )

        seen.update(paths)


def write_split(name, rows):
    destination = SPLIT_DIR / f"{name}.csv"

    if destination.exists():
        raise FileExistsError(
            f"{destination.name} ekziston. "
            "Nuk u ndryshua asnje file."
        )

    with destination.open(
        "x",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "label"]
        )

        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(
        row["label"]
        for row in rows
    )

    print(
        f"{name}: "
        f"{len(rows)} foto; "
        f"{dict(counts)}"
    )


def main():
    # =========================
    # 1. Lexojme versionin v2
    # =========================

    train_v2 = read_split(
        "train_mixed_v2"
    )

    validation_v2 = read_split(
        "validation_extra_v2"
    )

    holdout_v2 = read_split(
        "holdout_extra_v2"
    )

    print("\n=== V2 ===")

    print(
        f"Train v2: {len(train_v2)}"
    )

    print(
        f"Validation v2: "
        f"{len(validation_v2)}"
    )

    print(
        f"Holdout v2: "
        f"{len(holdout_v2)}"
    )

    # =========================
    # 2. Lexojme dosjen e re
    # =========================

    new_rows = load_new_no_fire()

    print(
        "\nEvening Forest condition_raw_img:"
    )

    print(
        f"Foto totale: {len(new_rows)}"
    )

    # Sigurohemi qe keto foto nuk jane
    # perdorur ne v2.

    v2_paths = {
        row["path"]
        for row in (
            train_v2
            + validation_v2
            + holdout_v2
        )
    }

    overlap = (
        v2_paths
        .intersection(
            row["path"]
            for row in new_rows
        )
    )

    if overlap:
        example = next(iter(overlap))

        raise ValueError(
            "Nje foto nga dosja e re "
            "eshte perdorur tashme ne v2: "
            f"{example}"
        )

    # =========================
    # 3. Temporal split
    # =========================

    boundary = int(
        len(new_rows) * TRAIN_RATIO
    )

    train_end = boundary - GAP

    if train_end <= 0:
        raise ValueError(
            "Nuk ka mjaft foto per "
            "train + GAP + validation."
        )

    if boundary >= len(new_rows):
        raise ValueError(
            "Nuk ka foto te mjaftueshme "
            "per validation."
        )

    new_train = new_rows[:train_end]

    gap_rows = new_rows[
        train_end:boundary
    ]

    new_validation = new_rows[
        boundary:
    ]

    print("\n=== Ndarja e re ===")

    print(
        f"Train No_Fire: "
        f"{len(new_train)}"
    )

    print(
        f"GAP: {len(gap_rows)}"
    )

    print(
        f"Validation No_Fire: "
        f"{len(new_validation)}"
    )

    if len(gap_rows) != GAP:
        raise ValueError(
            f"GAP nuk eshte {GAP}."
        )

    # =========================
    # 4. Krijojme v3
    # =========================

    train_v3 = (
        train_v2
        + new_train
    )

    validation_v3 = (
        validation_v2
        + new_validation
    )

    # SHUME E RENDESISHME:
    # holdout mbetet IDENTIK me v2.
    # Keto jane 202 fotot e kontrollit.
    holdout_v3 = list(holdout_v2)

    outputs = {
        "train_mixed_v3": train_v3,
        "validation_extra_v3":
            validation_v3,
        "holdout_extra_v3":
            holdout_v3,
    }

    # =========================
    # 5. Kontroll per leakage
    # =========================

    check_no_overlap(outputs)

    gap_paths = {
        row["path"]
        for row in gap_rows
    }

    used_paths = {
        row["path"]
        for rows in outputs.values()
        for row in rows
    }

    if gap_paths.intersection(
        used_paths
    ):
        raise ValueError(
            "Fotot e GAP-it jane futur "
            "gabimisht ne split."
        )

    # Sigurohemi qe holdout-i
    # eshte identik me v2.

    if holdout_v3 != holdout_v2:
        raise ValueError(
            "Holdout-i eshte ndryshuar!"
        )

    # =========================
    # 6. Ruajme CSV-te
    # =========================

    destinations = [
        SPLIT_DIR
        / f"{name}.csv"
        for name in outputs
    ]

    if any(
        path.exists()
        for path in destinations
    ):
        raise FileExistsError(
            "Ekziston nje file v3. "
            "Nuk u shkrua asnje liste."
        )

    print("\n=== V3 ===")

    for name, rows in outputs.items():
        write_split(
            name,
            rows
        )

    print(
        "\nNdarja v3 perfundoi."
    )

    print(
        "V2 mbeti e pandryshuar."
    )

    print(
        "202 fotot e holdout-it "
        "mbeten te rezervuara."
    )


if __name__ == "__main__":
    main()