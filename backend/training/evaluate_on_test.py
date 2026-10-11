"""Vlereson modele te ruajtura mbi setin e testimit te dhene me --test-csv.

Krahason drejtperdrejt v3 (vetem average pooling) me v4 (average + max), per
te pare nese max pooling i kap flaket e vogla qe v3 i humb. Baseline-i merret
nga CSV-ja ekzistuese e final_fire_v1, pra ai model nuk ri-ekzekutohet.

Kontrata e inputit merret nga inference.json: RGB, 254x254, [0,1].

Ekzekutohet nga rrenja e projektit, sepse te gjitha shtigjet ndertohen nga
--project-root (parazgjedhje: dosja aktuale).

    python backend/training/evaluate_on_test.py
    python backend/training/evaluate_on_test.py \
        --test-csv dataset/splits/test_final.csv \
        --models final_fire_v1/model.keras
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from analyze_scores import (
    SWEEP, average_precision, by_folder, confusion, error_bands,
    quantiles, roc_auc, threshold_for_recall,
)

LABELS = {"No_Fire": 0, "Fire": 1}
INPUT_SIZE = (254, 254)
DEFAULT_MODELS = (
    "fire_mobilenet_mixed_v3_finetuned.keras",
    "fire_mobilenet_avgmax_v4.keras",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV bosh: {path}")
    return rows


def build_dataset(tf, rows, data_root, batch_size):
    paths = [str(data_root / row["path"]) for row in rows]

    def read(path):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), INPUT_SIZE) / 255.0
        return image

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    # Pa shuffle: rendi duhet te perputhet me rreshtat e CSV-se.
    dataset = dataset.map(read, num_parallel_calls=2, deterministic=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.batch(batch_size).prefetch(1).with_options(options)


def evaluate(tf, model_path, rows, labels, data_root, batch_size):
    model = tf.keras.models.load_model(model_path, compile=False)

    if tuple(model.input_shape[1:]) != (*INPUT_SIZE, 3):
        raise ValueError(
            f"{model_path.name}: pritet input {INPUT_SIZE}, u gjet {model.input_shape}"
        )
    if model.output_shape[-1] != 1:
        raise ValueError(f"{model_path.name}: pritet nje dalje sigmoid.")

    print(f"\nEkzekutim: {model_path.name} mbi {len(rows)} foto")
    scores = model.predict(
        build_dataset(tf, rows, data_root, batch_size), verbose=1
    ).reshape(-1)

    if len(scores) != len(rows) or not np.all(np.isfinite(scores)):
        raise ValueError("Numri ose vlerat e rezultateve jane te pavlefshme.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Fire score duhet te jete ne [0,1].")

    del model
    tf.keras.backend.clear_session()
    return scores


def summarize(name, rows, labels, scores, targets):
    auc = roc_auc(labels, scores)
    half = confusion(labels, scores, 0.5)
    sweep = [confusion(labels, scores, value) for value in SWEEP]
    best = max(sweep, key=lambda item: item["balanced_accuracy"])

    print("\n" + "=" * 68)
    print(f"{name}")
    print("=" * 68)
    print(f"ROC AUC            {auc:.4f}")
    print(f"Average precision  {average_precision(labels, scores):.4f}")
    print(f"\nNe pragun 0.5:")
    print(f"  Fire recall         {half['fire_recall']:.2%}  "
          f"({half['TP']}/{half['TP'] + half['FN']})")
    print(f"  No_Fire specificity {half['no_fire_specificity']:.2%}  "
          f"({half['TN']}/{half['TN'] + half['FP']})")
    print(f"  Alarme te rreme     {half['FP']}")
    print(f"  Balanced accuracy   {half['balanced_accuracy']:.2%}")

    print("\nShperndarja:")
    for label, value in (("Fire", 1), ("No_Fire", 0)):
        stats = quantiles(scores[labels == value])
        print(f"  {label:<8} " + "  ".join(
            f"{key}={stats[key]:.4f}" for key in ("min", "p10", "median", "p90", "max")
        ))

    print("\nPragu   Fire recall   No_Fire spec.   Bal.acc   Alarme te rreme")
    for item in sweep:
        print(f"{item['threshold']:>5.2f}   {item['fire_recall']:>10.2%}   "
              f"{item['no_fire_specificity']:>13.2%}   "
              f"{item['balanced_accuracy']:>7.2%}   {item['FP']:>15}")

    print(f"\nBal.acc maksimale ne pragun {best['threshold']:.2f}: "
          f"Fire {best['fire_recall']:.2%}, No_Fire {best['no_fire_specificity']:.2%}")

    at_recall = {}
    print("\nMe recall-in e Fire te fiksuar:")
    for target in targets:
        item = threshold_for_recall(labels, scores, target)
        at_recall[f"{target:.2f}"] = item
        if item:
            print(f"  Fire >= {target:.0%}  ->  pragu {item['threshold']:.4f}, "
                  f"No_Fire {item['no_fire_specificity']:.2%}, "
                  f"alarme te rreme {item['FP']}")

    bands = error_bands(rows, labels, scores, 0.5)
    if bands:
        print("\nAlarme te rreme sipas segmentit (No_Fire, pragu 0.5):")
        for band in bands:
            start, end = band["frame_range"]
            bar = "#" * int(round(band["false_positive_rate"] * 40))
            print(f"  {start:>5}-{end:<5} {band['false_positives']:>4}/"
                  f"{band['no_fire_images']:<4} {band['false_positive_rate']:>7.1%} {bar}")

    return {
        "roc_auc": auc,
        "average_precision": average_precision(labels, scores),
        "at_threshold_0.5": half,
        "threshold_sweep": sweep,
        "best_balanced_accuracy": best,
        "at_fixed_fire_recall": at_recall,
        "score_quantiles": {
            "Fire": quantiles(scores[labels == 1]),
            "No_Fire": quantiles(scores[labels == 0]),
        },
        "by_folder": by_folder(rows, labels, scores, 0.5),
        "false_positive_bands": bands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run", default="final_fire_v1")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--test-csv", type=Path, default=None,
                        help="CSV me path,label. Si parazgjedhje known_test.csv "
                             "e run-it; perdor dataset/splits/test_final.csv per "
                             "matjen perfundimtare.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-recall", type=float, nargs="+",
                        default=[0.95, 0.98, 0.99])
    args = parser.parse_args()

    project = args.project_root.resolve()
    run_dir = project / "backend" / "artifacts" / args.run
    model_dir = project / "backend" / "artifacts" / "models"
    data_root = project / "dataset"

    if args.test_csv is not None:
        test_csv = args.test_csv if args.test_csv.is_absolute() else project / args.test_csv
    else:
        test_csv = run_dir / "known_test.csv"
    if not test_csv.is_file():
        raise SystemExit(f"Nuk u gjet: {test_csv}")

    print(f"Seti i testimit: {test_csv}")
    rows = read_csv(test_csv)
    for row in rows:
        row["path"] = row["path"].replace("\\", "/")
        row.setdefault("stratum", "test")
    labels = np.array([LABELS[row["label"]] for row in rows], dtype=np.int64)
    print(f"{test_csv.stem}: {len(rows)} foto "
          f"({int((labels == 1).sum())} Fire, {int((labels == 0).sum())} No_Fire)")

    results = {}

    # --- baseline-i merret nga CSV-ja ekzistuese; modeli nuk ri-ekzekutohet
    baseline_csv = run_dir / "known_test_predictions.csv"
    if baseline_csv.is_file():
        baseline_rows = read_csv(baseline_csv)
        if [row["path"] for row in baseline_rows] != [row["path"] for row in rows]:
            print("\nKUJDES: rendi i CSV-se baseline nuk perputhet; kapercehet.")
        else:
            scores = np.array(
                [float(row["fire_score"]) for row in baseline_rows], dtype=np.float64
            )
            results["final_fire_v1/model.keras (i ruajtur)"] = summarize(
                "final_fire_v1/model.keras   (nga CSV-ja ekzistuese)",
                rows, labels, scores, args.target_recall,
            )

    # --- modelet qe duhen ekzekutuar
    missing = []
    to_run = []
    for name in args.models:
        given = Path(name)
        candidates = [given] if given.is_absolute() else [
            model_dir / name,
            project / name,
            project / "backend" / "artifacts" / name,
        ]
        found = next((path for path in candidates if path.is_file()), None)
        (to_run.append(found) if found else missing.append(given))

    for path in missing:
        print(f"\nKUJDES: nuk u gjet modeli {path}")

    if to_run:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))

        for path in to_run:
            scores = evaluate(tf, path, rows, labels, data_root, args.batch_size)

            tag = f"{test_csv.stem}_{path.parent.name}_{path.stem}"
            output = run_dir / f"predictions_{tag}.csv"
            with output.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["path", "label", "stratum", "fire_score",
                                      "predicted_label"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row, score in zip(rows, scores):
                    writer.writerow({
                        **row,
                        "fire_score": float(score),
                        "predicted_label": "Fire" if score >= 0.5 else "No_Fire",
                    })

            results[f"{path.parent.name}/{path.name}"] = summarize(
                f"{path.parent.name}/{path.name}", rows, labels, scores,
                args.target_recall,
            )
            print(f"\nParashikimet: {output}")

    # --- krahasimi
    if len(results) > 1:
        print("\n" + "=" * 78)
        print(f"KRAHASIMI MBI {test_csv.stem.upper()}")
        print("=" * 78)
        print(f"{'Modeli':<44}{'AUC':>8}{'Fire':>9}{'No_Fire':>10}{'Bal.acc':>9}")
        print("-" * 78)
        for name, data in results.items():
            half = data["at_threshold_0.5"]
            print(f"{name[:43]:<44}{data['roc_auc']:>8.4f}"
                  f"{half['fire_recall']:>9.2%}{half['no_fire_specificity']:>10.2%}"
                  f"{half['balanced_accuracy']:>9.2%}")
        print("\n(Fire dhe No_Fire ne pragun 0.5. AUC nuk varet nga pragu.)")

    report = {
        "test_set": str(test_csv),
        "count": len(rows),
        "models": results,
        "note": (
            f"Te gjitha modelet jane vleresuar mbi te njejtat {len(rows)} "
            f"foto te setit '{test_csv.stem}', me te njejten kontrate inputi "
            "(RGB, 254x254, [0,1]). Asnje model nuk u trajnua dhe asnje prag "
            "nuk u ndryshua."
        ),
    }
    output = run_dir / f"comparison_{test_csv.stem}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRaporti: {output}")


if __name__ == "__main__":
    main()