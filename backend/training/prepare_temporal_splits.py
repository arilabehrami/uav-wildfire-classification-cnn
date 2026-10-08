from pathlib import Path
from collections import defaultdict, Counter
import csv
import re

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = DATA_DIR / "splits"
GAP = 500


def main():
    groups = defaultdict(list)

    for label in ["Fire", "No_Fire"]:
        for path in (DATA_DIR / "Training" / label).glob("*.jpg"):
            match = re.fullmatch(r"(.*?)(\d+)", path.stem)
            if not match:
                raise ValueError(f"Emër i panjohur: {path.name}")

            prefix, number = match.groups()
            groups[prefix].append((int(number), path, label))

    splits = {"train_temporal": [], "validation_temporal": []}

    for prefix, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0])

        # Segmenti i mesëm: 40–60% e sekuencës.
        start = items[int(len(items) * 0.4)][0]
        end = items[int(len(items) * 0.6)][0]

        for frame, path, label in items:
            row = (path.relative_to(DATA_DIR).as_posix(), label)

            if start <= frame < end:
                splits["validation_temporal"].append(row)
            elif frame < start - GAP or frame >= end + GAP:
                splits["train_temporal"].append(row)

        print(f"{prefix}: validation nga frame {start} deri {end - 1}")

    for name, rows in splits.items():
        counts = Counter(label for _, label in rows)
        if any(counts[label] == 0 for label in ["Fire", "No_Fire"]):
            raise ValueError(f"{name} nuk i ka të dyja klasat.")

        print(f"{name}: {len(rows)} foto — {dict(counts)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        with (OUTPUT_DIR / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()