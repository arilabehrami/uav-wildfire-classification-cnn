from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import re

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = DATA_DIR / "splits"

BLOCK_SIZE = 1000
MARGIN = 100
VALIDATION_RATIO = 0.2


def main():
    blocks = defaultdict(list)

    for label in ["Fire", "No_Fire"]:
        for path in (DATA_DIR / "Training" / label).glob("*.jpg"):
            match = re.fullmatch(r"(.*?)(\d+)", path.stem)

            if not match:
                raise ValueError(f"Emër i panjohur: {path.name}")

            source, number = match.groups()
            frame = int(number)

            # Përjashtojmë skajet e çdo blloku nga të dy ndarjet.
            position = frame % BLOCK_SIZE
            if not MARGIN <= position < BLOCK_SIZE - MARGIN:
                continue

            block = (source, frame // BLOCK_SIZE)
            relative_path = path.relative_to(DATA_DIR).as_posix()

            blocks[block].append((relative_path, label, source))

    keys = sorted(blocks)
    if len(keys) < 2:
        raise ValueError("Nuk ka mjaft blloqe për ndarje.")

    block_counts = {
        key: Counter((label, source) for _, label, source in rows)
        for key, rows in blocks.items()
    }

    totals = Counter()
    for counts in block_counts.values():
        totals.update(counts)

    rng = random.Random(42)
    validation_blocks = max(
        1, min(len(keys) - 1, round(len(keys) * VALIDATION_RATIO))
    )

    best_selection = None
    best_score = float("inf")

    # Kërkojmë një ndarje me përfaqësim të të gjitha grupeve.
    for _ in range(2000):
        selection = set(rng.sample(keys, validation_blocks))
        counts = Counter()

        for key in selection:
            counts.update(block_counts[key])

        ratios = [counts[group] / total for group, total in totals.items()]

        # Çdo grup duhet të ketë 10–35% të imazheve në validation.
        if not all(0.10 <= ratio <= 0.35 for ratio in ratios):
            continue

        score = sum(
            abs(ratio - VALIDATION_RATIO) for ratio in ratios
        )

        if score < best_score:
            best_score = score
            best_selection = selection

    if best_selection is None:
        raise ValueError(
            "Nuk u gjet ndarje e përshtatshme. "
            "Mos nis trajnim; duhet rishikuar madhësia e blloqeve."
        )

    splits = {"train_v2": [], "validation_v2": []}

    for key in keys:
        name = (
            "validation_v2"
            if key in best_selection
            else "train_v2"
        )
        splits[name].extend(blocks[key])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        print(f"\n{name}: {len(rows)} imazhe")

        counts = Counter((label, source) for _, label, source in rows)
        for (label, source), count in sorted(counts.items()):
            print(f"  {label} / {source}: {count}")

        with (OUTPUT_DIR / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])
            writer.writerows((path, label) for path, label, _ in rows)

    print("\nListat v2 u krijuan. Fotot origjinale nuk u ndryshuan.")


if __name__ == "__main__":
    main()