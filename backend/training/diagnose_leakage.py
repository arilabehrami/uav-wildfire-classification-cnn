"""Mat kontaminimin mes ndarjeve te dataset-it. Vetem lexon; nuk ndryshon asgje.

Kontrollon tri gjera qe SHA256 nuk i kap:
  1. Kuazi-duplikate mes train dhe validation/holdout/Test (dHash, distanca Hamming).
  2. Foto thuajse identike me etiketa te kunderta (etiketim i gabuar).
  3. Afersine e numrave te frame-ve mes ndarjeve nga i njejti burim.

Perdorimi:
    python diagnose_leakage.py --root dataset
    python diagnose_leakage.py --root dataset --max-distance 8
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)
HASH_SIZE = 8


# ---------------------------------------------------------------- hash-imi

def dhash(path):
    """64-bit difference hash: i qendrueshem ndaj rikompresimit dhe rishkallezimit."""
    with Image.open(path) as image:
        small = image.convert("L").resize(
            (HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS
        )
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return np.packbits(bits)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_all(data_root, relative_paths, cache_path):
    """Kthen {rruga: {"sha256": ..., "dhash": bytes}} me cache sipas mtime/size."""
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            cache = {}

    result = {}
    missing = []
    total = len(relative_paths)

    for index, relative in enumerate(sorted(relative_paths), start=1):
        path = data_root / relative
        if not path.is_file():
            missing.append(relative)
            continue

        stat = path.stat()
        entry = cache.get(relative)

        if (
            entry is None
            or entry.get("size") != stat.st_size
            or entry.get("mtime_ns") != stat.st_mtime_ns
        ):
            entry = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
                "dhash": dhash(path).tobytes().hex(),
            }
            cache[relative] = entry

        result[relative] = entry

        if index % 1000 == 0 or index == total:
            print(f"  gjurme: {index}/{total}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    if missing:
        print(f"\nKUJDES: {len(missing)} foto te listuara nuk ekzistojne ne disk.")
        for relative in missing[:10]:
            print(f"  mungon: {relative}")

    return result, missing


# ---------------------------------------------------------------- distancat

def to_matrix(entries):
    if not entries:
        return np.zeros((0, 8), dtype=np.uint8)
    return np.stack([
        np.frombuffer(bytes.fromhex(entry["dhash"]), dtype=np.uint8)
        for entry in entries
    ])


def nearest(left, right, chunk=256):
    """Per secilen rresht te 'left': distanca minimale dhe indeksi ne 'right'."""
    distances = np.full(len(left), 64, dtype=np.int16)
    indices = np.full(len(left), -1, dtype=np.int64)

    if not len(left) or not len(right):
        return distances, indices

    for start in range(0, len(left), chunk):
        block = left[start:start + chunk]
        matrix = POPCOUNT[block[:, None, :] ^ right[None, :, :]].sum(
            axis=2, dtype=np.int16
        )
        best = matrix.argmin(axis=1)
        distances[start:start + chunk] = matrix[np.arange(len(block)), best]
        indices[start:start + chunk] = best

    return distances, indices


# ---------------------------------------------------------------- ndarjet

def normalize(value):
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def read_split_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    output = []
    for row in rows:
        if "path" not in row or "label" not in row:
            raise ValueError(f"{path.name}: pritet kolona path dhe label.")
        output.append({"path": normalize(row["path"]), "label": row["label"]})
    return output


def scan_folder_split(data_root, folder):
    rows = []
    for label_dir in sorted((data_root / folder).iterdir()):
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                rows.append({
                    "path": path.relative_to(data_root).as_posix(),
                    "label": label_dir.name,
                })
    return rows


def load_all_splits(data_root):
    splits = {}

    splits_dir = data_root / "splits"
    if splits_dir.is_dir():
        for path in sorted(splits_dir.glob("*.csv")):
            try:
                rows = read_split_csv(path)
            except ValueError as error:
                print(f"  kapercehet {path.name}: {error}")
                continue
            if rows:
                splits[path.stem] = rows

    for folder in ("Test",):
        if (data_root / folder).is_dir():
            rows = scan_folder_split(data_root, folder)
            if rows:
                splits[f"folder:{folder}"] = rows

    return splits


def frame_key(relative):
    path = PurePosixPath(relative)
    match = re.fullmatch(r"(.*?)(\d+)", path.stem)
    if match is None:
        return None
    return f"{path.parent.as_posix()}|{match[1]}", int(match[2])


# ---------------------------------------------------------------- raporti

def is_training_split(name):
    return name.startswith("train")


def report_pair(train_name, train_rows, eval_name, eval_rows, prints, limit):
    train_matrix = to_matrix([row["entry"] for row in train_rows])
    eval_matrix = to_matrix([row["entry"] for row in eval_rows])
    distances, indices = nearest(eval_matrix, train_matrix)

    buckets = {
        "identike (d=0)": int(np.sum(distances == 0)),
        "d<=2": int(np.sum(distances <= 2)),
        "d<=6": int(np.sum(distances <= 6)),
        "d<=10": int(np.sum(distances <= 10)),
    }
    share = buckets["d<=6"] / len(eval_rows) if eval_rows else 0.0

    conflicts = []
    examples = []
    order = np.argsort(distances)

    for position in order:
        if distances[position] > limit:
            break
        eval_row = eval_rows[position]
        train_row = train_rows[indices[position]]
        item = {
            "distance": int(distances[position]),
            "eval_path": eval_row["path"],
            "eval_label": eval_row["label"],
            "train_path": train_row["path"],
            "train_label": train_row["label"],
        }
        if eval_row["label"] != train_row["label"]:
            conflicts.append(item)
        if len(examples) < 5:
            examples.append(item)

    prints.append(
        f"\n{eval_name}  vs  {train_name}"
        f"\n  {len(eval_rows)} foto per vleresim; "
        f"{buckets['d<=6']} kane nje binjak ne train (d<=6) = {share:.1%}"
        f"\n  {buckets}"
    )
    for item in examples:
        prints.append(
            f"    d={item['distance']:>2}  {item['eval_path']}"
            f"\n           ~  {item['train_path']}"
        )
    if conflicts:
        prints.append(
            f"  ETIKETA TE KUNDERTA ne {len(conflicts)} cifte thuajse identike:"
        )
        for item in conflicts[:5]:
            prints.append(
                f"    d={item['distance']:>2}  "
                f"{item['eval_path']} [{item['eval_label']}]"
                f"  vs  {item['train_path']} [{item['train_label']}]"
            )

    return {
        "eval_count": len(eval_rows),
        "train_count": len(train_rows),
        "buckets": buckets,
        "contaminated_share_d6": share,
        "label_conflicts": len(conflicts),
        "label_conflict_examples": conflicts[:20],
        "nearest_examples": examples,
    }


def frame_proximity(train_rows, eval_rows):
    train_frames = defaultdict(list)
    for row in train_rows:
        key = frame_key(row["path"])
        if key:
            train_frames[key[0]].append(key[1])

    closest = None
    for row in eval_rows:
        key = frame_key(row["path"])
        if not key or key[0] not in train_frames:
            continue
        gap = min(abs(number - key[1]) for number in train_frames[key[0]])
        if closest is None or gap < closest[0]:
            closest = (gap, row["path"])

    return {"min_frame_gap": closest[0], "example": closest[1]} if closest else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--max-distance", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_root = args.root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Nuk ekziston dosja: {data_root}")

    splits = load_all_splits(data_root)
    if not splits:
        raise SystemExit("Nuk u gjet asnje liste ne splits/ dhe asnje dosje Test.")

    print(f"Dataset: {data_root}")
    print(f"Lista te gjetura: {len(splits)}\n")
    for name, rows in sorted(splits.items()):
        counts = defaultdict(int)
        for row in rows:
            counts[row["label"]] += 1
        print(f"  {name}: {len(rows)} foto {dict(sorted(counts.items()))}")

    every_path = {row["path"] for rows in splits.values() for row in rows}
    print(f"\nGjurme per {len(every_path)} foto unike:")
    fingerprints, missing = fingerprint_all(
        data_root, every_path, data_root / "inspection" / "hash_cache.json"
    )

    for name in splits:
        splits[name] = [
            dict(row, entry=fingerprints[row["path"]])
            for row in splits[name]
            if row["path"] in fingerprints
        ]

    # --- duplikate byte-identike brenda e mes listave
    by_sha = defaultdict(list)
    for name, rows in splits.items():
        for row in rows:
            by_sha[row["entry"]["sha256"]].append((name, row["path"], row["label"]))

    exact = [group for group in by_sha.values() if len(group) > 1]
    cross = [g for g in exact if len({item[0] for item in g}) > 1]
    conflicting = [g for g in exact if len({item[2] for item in g}) > 1]

    print("\n" + "=" * 60)
    print("1. DUPLIKATE BYTE-IDENTIKE")
    print("=" * 60)
    print(f"Grupe me foto identike: {len(exact)}")
    print(f"  prej tyre ndarjeve te ndryshme: {len(cross)}")
    print(f"  prej tyre me etiketa te kunderta: {len(conflicting)}")
    for group in cross[:5]:
        print("  " + " | ".join(f"{n}:{p}" for n, p, _ in group))

    # --- kuazi-duplikate mes train dhe cdo liste vleresimi
    print("\n" + "=" * 60)
    print("2. KUAZI-DUPLIKATE MES TRAIN DHE VLERESIMIT")
    print("=" * 60)

    pairs = {}
    prints = []
    train_names = [name for name in sorted(splits) if is_training_split(name)]
    eval_names = [name for name in sorted(splits) if not is_training_split(name)]

    for train_name in train_names:
        for eval_name in eval_names:
            pairs[f"{eval_name} vs {train_name}"] = report_pair(
                train_name, splits[train_name],
                eval_name, splits[eval_name],
                prints, args.max_distance,
            )

    print("\n".join(prints))

    # --- afersia e frame-ve
    print("\n" + "=" * 60)
    print("3. DISTANCA MINIMALE E FRAME-VE (i njejti burim)")
    print("=" * 60)

    proximity = {}
    for train_name in train_names:
        for eval_name in eval_names:
            result = frame_proximity(splits[train_name], splits[eval_name])
            if result:
                proximity[f"{eval_name} vs {train_name}"] = result
                print(
                    f"{eval_name} vs {train_name}: "
                    f"{result['min_frame_gap']} frame  ({result['example']})"
                )

    # --- permbledhje
    print("\n" + "=" * 60)
    print("PERMBLEDHJE")
    print("=" * 60)

    worst = sorted(
        pairs.items(),
        key=lambda item: item[1]["contaminated_share_d6"],
        reverse=True,
    )[:8]
    for name, result in worst:
        print(f"  {result['contaminated_share_d6']:>7.1%} kontaminim (d<=6)  {name}")

    report = {
        "dataset_root": str(data_root),
        "hash": "dHash 8x8 (64 bit); distanca Hamming",
        "max_distance_reported": args.max_distance,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "missing_files": missing,
        "exact_duplicate_groups": len(exact),
        "exact_duplicates_across_splits": len(cross),
        "exact_duplicates_conflicting_labels": len(conflicting),
        "pairs": pairs,
        "frame_proximity": proximity,
        "limitation": (
            "dHash kap rikompresim, rishkallezim dhe ndryshime te vogla. "
            "Nuk kap rrotullim, prerje ose kalim te ngadalshem skene. "
            "Distanca e vogel nuk e provon vete korrelimin kohor; "
            "distanca e madhe nuk e provon pavaresine e skenes."
        ),
    }

    output = args.output or (data_root / "inspection" / "leakage_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRaporti: {output}")


if __name__ == "__main__":
    main()