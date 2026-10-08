"""Ndan dosjen Test ne tri pjese kohore, qe modeli te marre kundershembujt
qe i mungojne (tym pa flake = No_Fire) pa e humbur nje test te paprekur.

  test_adapt_train   ~40%  ->  shtohet ne trajnim
  (hendek)
  test_adapt_val     ~15%  ->  validim qe PERMBAN tym pa flake
  (hendek)
  test_final         ~45%  ->  nuk preket deri ne matjen perfundimtare

Ndarja behet sipas numrit te frame-it, veç per Fire dhe veç per No_Fire, qe
raporti i klasave te ruhet ne secilen pjese. Hendeqet largojne kornizat
fqinje, qe jane thuajse identike.

Nuk trajnon, nuk prek asnje foto, nuk mbishkruan asnje liste ekzistuese.

    python prepare_test_split.py --dry-run
    python prepare_test_split.py
"""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from diagnose_leakage import POPCOUNT, dhash, nearest
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "Vendose kete skript prane diagnose_leakage.py te backend/training/."
    ) from error

LABELS = ("Fire", "No_Fire")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_number(path):
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Emri nuk ka numer frame: {path.name}")
    return int(match.group(1))


def scan(test_root, data_root):
    groups = {}
    for label in LABELS:
        folder = test_root / label
        if not folder.is_dir():
            raise SystemExit(f"Mungon dosja: {folder}")

        files = [
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
        if not files:
            raise SystemExit(f"Nuk u gjet asnje foto te: {folder}")

        rows = sorted(
            (
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "label": label,
                    "frame": frame_number(path),
                }
                for path in files
            ),
            key=lambda row: row["frame"],
        )

        numbers = [row["frame"] for row in rows]
        if len(numbers) != len(set(numbers)):
            raise SystemExit(f"Numra frame te perseritur te {label}.")

        groups[label] = rows
    return groups


def cut(rows, train_ratio, val_ratio, gap):
    """Tri segmente te njepasnjeshme, me hendek mes tyre."""
    total = len(rows)
    train_end = int(total * train_ratio)
    val_start = train_end + gap
    val_end = int(total * (train_ratio + val_ratio))
    final_start = val_end + gap

    if not 0 < train_end < val_start < val_end < final_start < total:
        raise SystemExit(
            f"Nuk ka mjaft foto per kete ndarje ({total} foto, hendek {gap}). "
            "Provo nje --gap me te vogel."
        )

    return {
        "test_adapt_train": rows[:train_end],
        "test_adapt_val": rows[val_start:val_end],
        "test_final": rows[final_start:],
        "_excluded": rows[train_end:val_start] + rows[val_end:final_start],
    }


def fingerprints(rows, data_root, cache_path):
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            cache = {}

    vectors, computed = [], 0
    for row in rows:
        entry = cache.get(row["path"])
        if entry and "dhash" in entry:
            vectors.append(np.frombuffer(bytes.fromhex(entry["dhash"]), dtype=np.uint8))
        else:
            vectors.append(dhash(data_root / row["path"]))
            computed += 1

    if computed:
        print(f"  (u llogaritën {computed} gjurmë të reja)")
    return np.stack(vectors) if vectors else np.zeros((0, 8), dtype=np.uint8)


ORDER = ["test_adapt_train", "test_adapt_val", "test_final"]


def compare(splits, matrices, threshold, title):
    print(f"\n{title}")
    results = {}
    for index, left in enumerate(ORDER):
        for right in ORDER[index + 1:]:
            distances, _ = nearest(matrices[right], matrices[left])
            close = int(np.sum(distances <= threshold))
            minimum = int(distances.min()) if len(distances) else None
            results[f"{right} vs {left}"] = {
                "pairs_within_threshold": close,
                "min_distance": minimum,
                "share": close / len(splits[right]) if splits[right] else 0.0,
            }
            flag = "OK" if close == 0 else "KUJDES"
            print(f"  {flag}  {right} vs {left}: {close} foto në d<={threshold} "
                  f"(distanca minimale {minimum})")
    return results


def check_separation(splits, data_root, cache_path, threshold, drop):
    matrices = {
        name: fingerprints(splits[name], data_root, cache_path) for name in ORDER
    }
    before = compare(
        splits, matrices, threshold,
        "Kontroll: a ka foto thuajse identike mes pjesëve?",
    )

    removed = {}
    if not drop:
        return {"before": before, "after": before, "removed": removed}

    # test_adapt_train eshte referenca: hiqen fotot nga pjeset e vleresimit.
    for index, name in enumerate(ORDER[1:], start=1):
        keep = np.ones(len(splits[name]), dtype=bool)
        for reference in ORDER[:index]:
            if not len(matrices[reference]) or not len(matrices[name]):
                continue
            distances, _ = nearest(matrices[name], matrices[reference])
            keep &= distances > threshold

        dropped = [row for row, flag in zip(splits[name], keep) if not flag]
        if dropped:
            removed[name] = [row["path"] for row in dropped]
            splits[name] = [row for row, flag in zip(splits[name], keep) if flag]
            matrices[name] = matrices[name][keep]
            print(f"\nU hoqën {len(dropped)} foto nga {name}.")

    after = compare(
        splits, matrices, threshold,
        "Rikontroll pas heqjes:",
    ) if removed else before

    return {"before": before, "after": after, "removed": removed}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_split(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [
            {"path": row["path"], "label": row["label"]}
            for row in csv.DictReader(file)
        ]


def summary(rows):
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-ratio", type=float, default=0.40)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--gap", type=int, default=200)
    parser.add_argument("--base-train", default="train_mixed_v3")
    parser.add_argument("--distance", type=int, default=6)
    parser.add_argument(
        "--keep-near-duplicates", action="store_true",
        help="Mos i hiq fotot qe kane binjak ne nje pjese te meparshme.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        parser.error("Raportet duhet te jene mes 0 dhe 1.")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("train-ratio + val-ratio duhet te jete nen 1.")
    if args.gap < 1:
        parser.error("Hendeku duhet te jete te pakten 1 frame.")

    project = args.project_root.resolve()
    data_root = project / "dataset"
    splits_dir = data_root / "splits"
    cache_path = data_root / "inspection" / "hash_cache.json"

    groups = scan(data_root / "Test", data_root)
    print(f"Dosja Test: " + ", ".join(
        f"{label} {len(rows)}" for label, rows in groups.items()
    ))

    splits = {name: [] for name in
              ("test_adapt_train", "test_adapt_val", "test_final", "_excluded")}

    for label, rows in groups.items():
        parts = cut(rows, args.train_ratio, args.val_ratio, args.gap)
        boundaries = {
            name: (part[0]["frame"], part[-1]["frame"])
            for name, part in parts.items() if part and name != "_excluded"
        }
        print(f"\n{label}: " + "  ".join(
            f"{name} frame {low}-{high} ({len(parts[name])})"
            for name, (low, high) in boundaries.items()
        ))
        for name, part in parts.items():
            splits[name].extend(part)

    excluded = splits.pop("_excluded")
    print(f"\nTe perjashtuara ne hendeqe: {len(excluded)} foto")

    for name, rows in splits.items():
        print(f"{name}: {len(rows)} foto {summary(rows)}")

    paths = {row["path"] for rows in splits.values() for row in rows}
    if len(paths) != sum(len(rows) for rows in splits.values()):
        raise SystemExit("Nje foto ndodhet ne me shume se nje pjese.")
    if paths & {row["path"] for row in excluded}:
        raise SystemExit("Fotot e hendekut hyne ne nje pjese.")

    separation = check_separation(
        splits, data_root, cache_path, args.distance, not args.keep_near_duplicates
    )

    if separation["removed"]:
        print("\nPas pastrimit:")
        for name, rows in splits.items():
            print(f"  {name}: {len(rows)} foto {summary(rows)}")
        paths = {row["path"] for rows in splits.values() for row in rows}

    base_path = splits_dir / f"{args.base_train}.csv"
    combined = None
    if base_path.is_file():
        base = read_split(base_path)
        overlap = {row["path"] for row in base} & paths
        if overlap:
            raise SystemExit(
                f"{args.base_train} permban tashme foto te Test-it: {len(overlap)}"
            )
        combined = base + [
            {"path": row["path"], "label": row["label"]}
            for row in splits["test_adapt_train"]
        ]
        print(f"\ntrain_with_adapt: {len(combined)} foto {summary(combined)}")
        print(f"  ({len(base)} nga {args.base_train} + "
              f"{len(splits['test_adapt_train'])} nga Test)")
    else:
        print(f"\n(nuk u gjet {base_path.name}; nuk u ndertua lista e trajnimit)")

    if args.dry_run:
        print("\n--dry-run: nuk u shkrua asnje fajll.")
        return

    outputs = dict(splits)
    if combined is not None:
        outputs["train_with_adapt"] = combined

    destinations = {name: splits_dir / f"{name}.csv" for name in outputs}
    existing = [path.name for path in destinations.values() if path.exists()]
    if existing:
        raise SystemExit(
            f"Ekzistojne tashme: {', '.join(existing)}. "
            "Asnje fajll nuk u ndryshua."
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(destinations[name], rows)
        print(f"U shkrua: {destinations[name]}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "dataset/Test",
        "rule": (
            "Ndarje kohore sipas numrit te frame-it, veç per secilen klase: "
            f"{args.train_ratio:.0%} adapt_train, hendek {args.gap}, "
            f"{args.val_ratio:.0%} adapt_val, hendek {args.gap}, pjesa tjeter final."
        ),
        "gap_frames": args.gap,
        "counts": {name: summary(rows) for name, rows in outputs.items()},
        "excluded_in_gaps": len(excluded),
        "near_duplicate_rule": (
            f"dHash 8x8, distanca Hamming. Fotot e test_adapt_val dhe test_final "
            f"me distance <= {args.distance} ndaj nje pjese te meparshme u hoqen."
            if not args.keep_near_duplicates else
            f"Vetem raportim, pa heqje (prag {args.distance})."
        ),
        "near_duplicate_check": separation,
        "base_train_split": args.base_train if combined is not None else None,
        "limitation": (
            "test_final comes from the same video and camera as test_adapt_*, "
            "separated only in time. It is a held-out temporal segment, not an "
            "independent recording. Any result on it must be reported as such."
        ),
    }
    manifest_path = splits_dir / "test_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"U shkrua: {manifest_path}")
    print("\nDosja Test dhe listat e meparshme mbeten te paprekura.")


if __name__ == "__main__":
    main()