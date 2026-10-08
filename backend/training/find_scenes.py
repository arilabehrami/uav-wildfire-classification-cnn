import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PATTERN = re.compile(r"^(.*?)(\d+)$")


def thumbnail(path):
    with Image.open(path) as image:
        image = image.convert("L").resize((32, 32))
        return np.asarray(image, dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--max-gap", type=int, default=30)
    args = parser.parse_args()

    if not 0 < args.threshold <= 1 or args.max_gap < 1:
        raise ValueError("Pragu ose max-gap është i pavlefshëm.")

    root = args.root.resolve()
    groups = defaultdict(list)

    for label in ["Fire", "No_Fire"]:
        folder = root / "Training" / label
        if not folder.is_dir():
            raise ValueError(f"Mungon folderi: {folder}")

        for path in folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            match = PATTERN.fullmatch(path.stem)
            if not match:
                raise ValueError(f"Emër i panjohur: {path.name}")

            prefix, number = match.groups()
            groups[prefix].append((int(number), path, label))

    if not groups:
        raise ValueError("Nuk u gjetën foto.")

    rows, pairs = [], []

    for prefix, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0])

        numbers = [item[0] for item in items]
        if len(numbers) != len(set(numbers)):
            raise ValueError(
                f"Numra frame-sh të përsëritur te {prefix}. "
                "Duhet verifikuar burimi para bashkimit."
            )

        scene = 0
        previous = None
        differences = []

        for number, path, label in items:
            current = thumbnail(path)
            difference, gap, reason = 0.0, 0, ""

            if previous is not None:
                prev_number, prev_path, prev_image = previous
                difference = float(np.abs(current - prev_image).mean())
                gap = number - prev_number
                differences.append(difference)

                reasons = []
                if difference > args.threshold:
                    reasons.append("visual")
                if gap > args.max_gap:
                    reasons.append("gap")

                reason = "+".join(reasons)
                if reason:
                    scene += 1

                pairs.append({
                    "a": prev_path,
                    "b": path,
                    "diff": difference,
                    "gap": gap,
                    "reason": reason,
                })

            rows.append({
                "path": path.relative_to(root).as_posix(),
                "label": label,
                "prefix": prefix,
                "frame": number,
                "scene_id": f"{prefix}|{scene}",
                "diff_prev": round(difference, 6),
                "gap_prev": gap,
                "boundary_reason": reason,
            })

            previous = number, path, current

        print(f"{prefix}: {len(items)} foto, {scene + 1} skena kandidate")
        if differences:
            values = np.percentile(differences, [50, 90, 99])
            print("  diff p50/p90/p99:", np.round(values, 4))

    output = root / "inspection"
    output.mkdir(parents=True, exist_ok=True)

    with (output / "scenes_candidate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Kontrollojmë kufij vizualë, hendekë dhe raste afër pragut.
    selected = []
    seen = set()

    def add_candidates(candidates, limit):
        added = 0
        for pair in candidates:
            key = (pair["a"], pair["b"])
            if key not in seen:
                seen.add(key)
                selected.append(pair)
                added += 1
                if added == limit:
                    break

    add_candidates(
        sorted(
            [p for p in pairs if p["gap"] > args.max_gap],
            key=lambda p: p["gap"], reverse=True,
        ), 8,
    )
    add_candidates(
        sorted(
            [p for p in pairs if p["diff"] > args.threshold],
            key=lambda p: p["diff"], reverse=True,
        ), 8,
    )
    add_candidates(
        sorted(pairs, key=lambda p: abs(p["diff"] - args.threshold)),
        24 - len(selected),
    )

    if selected:
        sheet = Image.new(
            "RGB", (720, 210 * ((len(selected) + 1) // 2)), "white"
        )
        draw = ImageDraw.Draw(sheet)

        for index, pair in enumerate(selected):
            x = (index % 2) * 360
            y = (index // 2) * 210

            for offset, key in enumerate(["a", "b"]):
                with Image.open(pair[key]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((175, 175))
                    sheet.paste(image, (x + offset * 180, y))
                draw.text(
                    (x + offset * 180, y + 176),
                    pair[key].stem,
                    fill="black",
                )

            draw.text(
                (x, y + 191),
                f"diff={pair['diff']:.3f}, gap={pair['gap']}, "
                f"{pair['reason'] or 'pa kufi'}",
                fill="black",
            )

        sheet.save(output / "scene_pairs.jpg", quality=95)

    print(f"\nRezultatet u ruajtën te: {output}")
    print("Listat train/validation nuk u ndryshuan.")


if __name__ == "__main__":
    main()