"""Analizon rezultatet e ruajtura: AUC, pragjet, dhe ku grumbullohen gabimet.

Nuk ngarkon TensorFlow, nuk trajnon, nuk e ndryshon modelin. Lexon vetem
CSV-te e parashikimeve qe fire_final.py i ka shkruar ne run-in e dhene.

AUC e ndan pyetjen vendimtare:
  AUC e larte + specificity e ulet  ->  problem i pragut/kalibrimit
  AUC e ulet                        ->  modeli nuk i dallon skenat e reja

    python analyze_scores.py
    python analyze_scores.py --run final_fire_v1 --target-recall 0.95 0.98
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

FILES = ("validation", "known_test", "diagnostic_202")
SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ---------------------------------------------------------------- metrikat

def roc_auc(labels, scores):
    """Mann-Whitney U mbi rangjet mesatare; trajton barazimet sakte."""
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]

    index = 0
    while index < len(sorted_scores):
        stop = index
        while stop + 1 < len(sorted_scores) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1

    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def average_precision(labels, scores):
    positives = int(np.sum(labels == 1))
    if not positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order] == 1
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / positives)


def confusion(labels, scores, threshold):
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted))
    ratio = lambda a, b: a / b if b else None
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "threshold": threshold,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "fire_recall": recall,
        "no_fire_specificity": specificity,
        "fire_precision": ratio(tp, tp + fp),
        "balanced_accuracy": (
            (recall + specificity) / 2 if recall is not None and specificity is not None else None
        ),
        "accuracy": ratio(tp + tn, len(labels)),
    }


def threshold_for_recall(labels, scores, target):
    """Pragu me i larte qe mban fire_recall >= target; jep specificity atje."""
    positives = scores[labels == 1]
    if not len(positives) or target <= 0:
        return None
    # Pragu = kuantili (1-target) i rezultateve te klasës Fire.
    threshold = float(np.quantile(positives, 1.0 - target, method="lower"))
    return confusion(labels, scores, threshold)


def quantiles(values):
    if not len(values):
        return None
    probabilities = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    names = ["min", "p10", "p25", "median", "p75", "p90", "max"]
    return dict(zip(names, (float(v) for v in np.quantile(values, probabilities))))


# ---------------------------------------------------------------- grupimi

def frame_number(path):
    match = re.search(r"(\d+)$", PurePosixPath(path).stem)
    return int(match.group(1)) if match else None


def error_bands(rows, labels, scores, threshold, bands=12):
    """Ku grumbullohen alarmet e rreme sipas numrit te frame-it."""
    numbers = np.array(
        [frame_number(row["path"]) or -1 for row in rows], dtype=np.int64
    )
    usable = (labels == 0) & (numbers >= 0)
    if usable.sum() < bands:
        return None

    values = numbers[usable]
    predicted = scores[usable] >= threshold
    edges = np.quantile(values, np.linspace(0, 1, bands + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None

    output = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (values >= start) & (
            (values < end) if end != edges[-1] else (values <= end)
        )
        if not mask.sum():
            continue
        output.append({
            "frame_range": [int(start), int(end)],
            "no_fire_images": int(mask.sum()),
            "false_positives": int(predicted[mask].sum()),
            "false_positive_rate": float(predicted[mask].mean()),
        })
    return output


def by_folder(rows, labels, scores, threshold):
    groups = defaultdict(lambda: {"n": 0, "errors": 0, "label": None})
    for row, label, score in zip(rows, labels, scores):
        key = PurePosixPath(row["path"]).parent.as_posix()
        entry = groups[key]
        entry["n"] += 1
        entry["label"] = "Fire" if label == 1 else "No_Fire"
        if (score >= threshold) != (label == 1):
            entry["errors"] += 1
    return {
        name: {**entry, "error_rate": entry["errors"] / entry["n"]}
        for name, entry in sorted(groups.items())
    }


# ---------------------------------------------------------------- leximi

def load(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV bosh: {path}")

    labels, scores = [], []
    for row in rows:
        if row["label"] not in ("Fire", "No_Fire"):
            raise ValueError(f"Etikete e panjohur: {row['label']}")
        score = float(row["fire_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Fire score jashte [0,1]: {score}")
        labels.append(1 if row["label"] == "Fire" else 0)
        scores.append(score)

    return rows, np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float64)


def analyze(name, path, targets):
    rows, labels, scores = load(path)
    auc = roc_auc(labels, scores)
    at_half = confusion(labels, scores, 0.5)

    sweep = [confusion(labels, scores, value) for value in SWEEP]
    best = max(
        (item for item in sweep if item["balanced_accuracy"] is not None),
        key=lambda item: item["balanced_accuracy"],
        default=None,
    )
    at_recall = {
        f"{target:.2f}": threshold_for_recall(labels, scores, target)
        for target in targets
    }

    print("\n" + "=" * 68)
    print(f"{name}   ({len(rows)} foto: "
          f"{int((labels == 1).sum())} Fire, {int((labels == 0).sum())} No_Fire)")
    print("=" * 68)

    print(f"\nROC AUC            {auc:.4f}" if auc is not None else "\nROC AUC   n/a")
    ap = average_precision(labels, scores)
    if ap is not None:
        print(f"Average precision  {ap:.4f}")

    print(f"\nNe pragun 0.5:")
    print(f"  Fire recall        {at_half['fire_recall']:.2%}  "
          f"({at_half['TP']}/{at_half['TP'] + at_half['FN']})")
    print(f"  No_Fire specificity {at_half['no_fire_specificity']:.2%}  "
          f"({at_half['TN']}/{at_half['TN'] + at_half['FP']})")
    print(f"  Alarme te rreme    {at_half['FP']}")
    print(f"  Balanced accuracy  {at_half['balanced_accuracy']:.2%}")

    print("\nShperndarja e rezultateve:")
    for label, value in (("Fire", 1), ("No_Fire", 0)):
        stats = quantiles(scores[labels == value])
        if stats:
            print(f"  {label:<8} " + "  ".join(
                f"{key}={stats[key]:.4f}" for key in ("min", "p10", "median", "p90", "max")
            ))

    print("\nPragu   Fire recall   No_Fire spec.   Bal.acc   Alarme te rreme")
    for item in sweep:
        print(f"{item['threshold']:>5.2f}   "
              f"{item['fire_recall']:>10.2%}   "
              f"{item['no_fire_specificity']:>13.2%}   "
              f"{item['balanced_accuracy']:>7.2%}   "
              f"{item['FP']:>15}")

    if best:
        print(f"\nBalanced accuracy maksimale ne pragun {best['threshold']:.2f}: "
              f"Fire {best['fire_recall']:.2%}, No_Fire {best['no_fire_specificity']:.2%}")

    print("\nMe recall-in e Fire te fiksuar:")
    for target, item in at_recall.items():
        if item:
            print(f"  Fire >= {float(target):.0%}  ->  pragu {item['threshold']:.4f}, "
                  f"No_Fire {item['no_fire_specificity']:.2%}, "
                  f"alarme te rreme {item['FP']}")

    folders = by_folder(rows, labels, scores, 0.5)
    if len(folders) > 1:
        print("\nGabime sipas dosjes (pragu 0.5):")
        for folder, entry in folders.items():
            print(f"  {folder} [{entry['label']}]: "
                  f"{entry['errors']}/{entry['n']} = {entry['error_rate']:.2%}")

    bands = error_bands(rows, labels, scores, 0.5)
    if bands:
        print("\nAlarme te rreme sipas numrit te frame-it (No_Fire, pragu 0.5):")
        for band in bands:
            start, end = band["frame_range"]
            bar = "#" * int(round(band["false_positive_rate"] * 40))
            print(f"  {start:>7}-{end:<7} {band['false_positives']:>5}/"
                  f"{band['no_fire_images']:<5} {band['false_positive_rate']:>7.1%} {bar}")

    return {
        "file": str(path),
        "count": len(rows),
        "fire_count": int((labels == 1).sum()),
        "no_fire_count": int((labels == 0).sum()),
        "roc_auc": auc,
        "average_precision": ap,
        "at_threshold_0.5": at_half,
        "threshold_sweep": sweep,
        "best_balanced_accuracy": best,
        "at_fixed_fire_recall": at_recall,
        "score_quantiles": {
            "Fire": quantiles(scores[labels == 1]),
            "No_Fire": quantiles(scores[labels == 0]),
        },
        "by_folder": folders,
        "false_positive_bands": bands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument(
        "--target-recall", type=float, nargs="+", default=[0.95, 0.98, 0.99]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.project_root.resolve() / "backend" / "artifacts" / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"Nuk ekziston: {run_dir}")

    report = {"run": str(run_dir), "sets": {}}

    for name in FILES:
        path = run_dir / f"{name}_predictions.csv"
        if not path.is_file():
            print(f"\n(mungon {path.name}, kapercehet)")
            continue
        report["sets"][name] = analyze(name, path, args.target_recall)

    aucs = {
        name: data["roc_auc"]
        for name, data in report["sets"].items()
        if data["roc_auc"] is not None
    }

    if len(aucs) > 1:
        print("\n" + "=" * 68)
        print("KRAHASIMI I AUC-se")
        print("=" * 68)
        for name, value in aucs.items():
            print(f"  {name:<16} {value:.4f}")
        if "validation" in aucs and "known_test" in aucs:
            gap = aucs["validation"] - aucs["known_test"]
            print(f"\n  Hendeku validation -> known_test: {gap:+.4f}")
            print("\n  Lexim: hendek i vogel me specificity te ulet = problem pragu.")
            print("         Hendek i madh = modeli nuk i pergjithesON skenat e reja.")

    report["auc_comparison"] = aucs
    report["note"] = (
        "AUC dhe average precision nuk varen nga pragu. Shperndarja e "
        "rezultateve dhe sweep-i i pragjeve tregojne nese ekziston nje prag "
        "i vetem i pranueshem. Bandat sipas frame-it tregojne grumbullim "
        "gabimesh, por numrat e frame-ve nuk jane ID skenash te verifikuara."
    )

    output = args.output or (run_dir / "score_analysis.json")
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRaporti: {output}")


if __name__ == "__main__":
    main()