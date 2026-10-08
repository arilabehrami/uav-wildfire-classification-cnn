"""Vendose ne backend/training/. Vlereson validimin; nuk ben trajnim."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "models/fire_mobilenet_mixed_v3_finetuned.keras"
OUTPUT_DIR = ARTIFACTS / "metrics/validation_audit_v3"
SPLITS = ("validation_temporal", "validation_extra_v3")


def read_rows(data_dir, name):
    with (data_dir / "splits" / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    if not rows or {row["label"] for row in rows} != {"Fire", "No_Fire"}:
        raise ValueError(f"Lista {name} duhet t'i kete te dyja klasat.")
    return rows


def check_disjoint(splits):
    seen = set()
    for name, rows in splits.items():
        paths = [row["path"].replace("\\", "/").casefold() for row in rows]
        if len(paths) != len(set(paths)) or seen.intersection(paths):
            raise ValueError(f"Foto te perseritura brenda ose mes listave: {name}")
        seen.update(paths)


def source_group(path):
    parts = path.replace("\\", "/").split("/")
    if parts[0] == "extra_raw" and len(parts) > 2:
        return parts[1]
    return "dataset_baze/lake" if parts[-1].startswith("lake_") else "dataset_baze/te_tjera"


def metrics(records, threshold=0.5):
    positives = [r["fire_score"] for r in records if r["label"] == "Fire"]
    negatives = [r["fire_score"] for r in records if r["label"] == "No_Fire"]
    tp = sum(score >= threshold for score in positives)
    tn = sum(score < threshold for score in negatives)
    fn, fp = len(positives) - tp, len(negatives) - tn
    recall = tp / len(positives) if positives else None
    specificity = tn / len(negatives) if negatives else None
    min_fire = min(positives) if positives else None
    max_no_fire = max(negatives) if negatives else None
    return {
        "count": len(records),
        "threshold": threshold,
        "Fire": {"correct": tp, "total": len(positives), "missed": fn},
        "No_Fire": {"correct": tn, "total": len(negatives), "false_alarms": fp},
        "accuracy": (tp + tn) / len(records) if records else None,
        "fire_recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
        "min_fire_score": min_fire,
        "max_no_fire_score": max_no_fire,
        "perfect_threshold_exists_on_this_group": (
            max_no_fire < min_fire if positives and negatives else None
        ),
    }


def print_metrics(title, result):
    fire, no_fire = result["Fire"], result["No_Fire"]
    print(f"\n{title} ({result['count']} foto, pragu 0.5)")
    if fire["total"]:
        print(f"  Fire: {fire['correct']}/{fire['total']} sakte; te humbura: {fire['missed']}")
    if no_fire["total"]:
        print(f"  No_Fire: {no_fire['correct']}/{no_fire['total']} sakte; alarme te rreme: {no_fire['false_alarms']}")
    if result["perfect_threshold_exists_on_this_group"] is not None:
        print(f"  Min Fire={result['min_fire_score']:.6f}; max No_Fire={result['max_no_fire_score']:.6f}")


def main():
    import numpy as np
    import tensorflow as tf
    from dataset_loader import DATA_DIR, LABELS, BATCH_SIZE, load_image

    if LABELS != {"No_Fire": 0, "Fire": 1}:
        raise ValueError(f"Kontrollo etiketat: {LABELS}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    report_path = OUTPUT_DIR / "validation_audit_v3.json"
    scores_path = OUTPUT_DIR / "validation_predictions_v3.csv"
    for path in (report_path, scores_path):
        if path.exists():
            raise FileExistsError(f"Raporti ekziston: {path}")

    # CSV-te e train/holdout lexohen vetem per kontrollin e ndarjeve.
    all_rows = {
        name: read_rows(DATA_DIR, name)
        for name in ("train_mixed_v3", *SPLITS, "holdout_extra_v2")
    }
    check_disjoint(all_rows)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    records = []

    for name in SPLITS:
        rows = all_rows[name]
        paths = [str(DATA_DIR / row["path"]) for row in rows]
        labels = np.asarray([LABELS[row["label"]] for row in rows], dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        # Pa shuffle, qe rezultatet te perputhen me rreshtat e CSV-se.
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        images, _ = next(iter(dataset))
        tf.debugging.assert_all_finite(images, "Imazhe te pavlefshme.")
        if tuple(images.shape[1:]) != (254, 254, 3):
            raise ValueError(f"Permasat e loader-it jane te papritura: {images.shape}")
        if float(tf.reduce_min(images)) < -1e-6 or float(tf.reduce_max(images)) > 1.000001:
            raise ValueError("load_image duhet te jape vlera ne [0,1].")

        print(f"\nVleresim: {name}")
        scores = model.predict(dataset, verbose=1).reshape(-1)
        if len(scores) != len(rows) or not np.all(np.isfinite(scores)):
            raise ValueError("Numri ose vlerat e rezultateve jane te pavlefshme.")
        if np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Modeli duhet te jape nje Fire score ne [0,1].")
        for row, score in zip(rows, scores, strict=True):
            records.append({
                "split": name,
                "source": source_group(row["path"]),
                "path": row["path"],
                "label": row["label"],
                "fire_score": float(score),
            })

    by_split, by_source = defaultdict(list), defaultdict(list)
    for row in records:
        by_split[row["split"]].append(row)
        by_source[f"{row['split']} / {row['source']}"].append(row)
    split_metrics = {name: metrics(rows) for name, rows in by_split.items()}
    source_metrics = {name: metrics(rows) for name, rows in by_source.items()}
    for name, result in split_metrics.items():
        print_metrics(name, result)
    for name, result in source_metrics.items():
        print_metrics(name, result)

    with MODEL_PATH.open("rb") as file:
        model_hash = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "model": str(MODEL_PATH),
        "model_sha256": model_hash,
        "tensorflow_version": tf.__version__,
        "by_split": split_metrics,
        "by_source": source_metrics,
        "all_validation": metrics(records),
        "note": "Validation diagnostics only. No model fitting or threshold change. The 202 previously inspected control images were not evaluated here; final generalization requires an independent test.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with scores_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "source", "path", "label", "fire_score"])
        writer.writeheader()
        writer.writerows(records)
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\nRaporti: {report_path}")
    print(f"Rezultatet per foto: {scores_path}")
    print("Kontrolli perfundoi. Modeli mbeti i pandryshuar.")


if __name__ == "__main__":
    main()
