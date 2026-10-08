"""Trajnon modelin me kundershembujt e rinj (mjegull/re = No_Fire).

Politika aktuale: vetem flake te dukshme = Fire; tym pa flake = No_Fire.
Listat train_flame_reviewed_v1 dhe validation_flame_reviewed_v1 krijohen
vetem pas auditimit te plote me prepare_reviewed_flame_splits.py.

Ndryshimet nga v3/v4:
  - trajnohet mbi train_flame_reviewed_v1 me etiketa te shqyrtuara
  - validimi perdor validation_flame_reviewed_v1
  - koka perdor average + max pooling; tani dega max ka arsye te ekzistoje,
    sepse tymi s'mjafton me per t'i ndare klasat
  - augmentim fotometrik me i forte, plus flip vertikal (pamje nadir)

test_final NUK preket: eshte perdorur tashme dhe nuk sherben per perzgjedhje.

    python train_adapted.py --dry-run
    python train_adapted.py
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

LABELS = {"No_Fire": 0, "Fire": 1}
INPUT_SIZE = (254, 254)
VALIDATION_SPLITS = ("validation_flame_reviewed_v1",)


# ---------------------------------------------------------------- te dhenat

def read_split(splits_dir, name):
    path = splits_dir / f"{name}.csv"
    if not path.is_file():
        raise SystemExit(f"Nuk u gjet lista: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"Lista eshte bosh: {name}")
    for row in rows:
        row["path"] = row["path"].replace("\\", "/")
        if row["label"] not in LABELS:
            raise SystemExit(f"Etikete e panjohur ne {name}: {row['label']}")
        row["split"] = name
        row["stratum"] = stratum(row["path"])
    return rows


def stratum(path):
    parts = PurePosixPath(path).parts
    if parts[0] == "Test":
        return "test_domain"
    if parts[0] == "extra_raw":
        return f"extra/{parts[1]}" if len(parts) > 2 else "extra/other"
    if PurePosixPath(path).name.startswith("lake_"):
        return "base/lake"
    return "base/forest"


def check_disjoint(groups):
    seen = {}
    for name, rows in groups.items():
        for row in rows:
            key = row["path"].casefold()
            if key in seen:
                raise SystemExit(
                    f"Foto e perbashket mes {seen[key]} dhe {name}: {row['path']}"
                )
            seen[key] = name


def sample_weights(rows):
    """50% peshe per klase; brenda klases, pesha e nje strati ~ sqrt(n)."""
    counts = Counter((row["label"], row["stratum"]) for row in rows)
    totals = {
        label: sum(
            math.sqrt(count) for (other, _), count in counts.items() if other == label
        )
        for label in LABELS
    }
    weights = []
    for row in rows:
        count = counts[(row["label"], row["stratum"])]
        weights.append(
            len(rows) * 0.5 / (math.sqrt(count) * totals[row["label"]])
        )
    return np.asarray(weights, dtype=np.float32)


def summary(rows):
    return dict(sorted(Counter(
        f"{row['label']}/{row['stratum']}" for row in rows
    ).items()))


# ---------------------------------------------------------------- modeli

def build(tf, dropout=0.35):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3), name="rgb_0_to_1")
    x = tf.keras.layers.Resizing(224, 224, name="resize_224")(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_scaling")(x)
    maps = base(x, training=False)

    average = tf.keras.layers.GlobalAveragePooling2D(name="avg_features")(maps)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="max_features")(maps)
    features = tf.keras.layers.Concatenate(name="avg_and_max")([average, maximum])
    extractor = tf.keras.Model(inputs, features, name="extractor_avgmax")

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[-1],)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(
            1, activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
    ], name="fire_head")

    model = tf.keras.Model(inputs, head(extractor.output), name="fire_adapted")
    return model, extractor, head


def augmenter(tf, strength):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect", seed=43),
        tf.keras.layers.RandomBrightness(0.25 * strength, value_range=(0, 1), seed=44),
        tf.keras.layers.RandomContrast(0.25 * strength, seed=45),
    ], name="augment")


def dataset(tf, rows, data_root, batch_size, augment=None):
    paths = [str(data_root / row["path"]) for row in rows]

    def read(path):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        return tf.image.resize(tf.cast(image, tf.float32), INPUT_SIZE) / 255.0

    data = tf.data.Dataset.from_tensor_slices(paths)
    data = data.map(read, num_parallel_calls=2, deterministic=True).batch(batch_size)
    if augment is not None:
        data = data.map(
            lambda images: tf.clip_by_value(augment(images, training=True), 0.0, 1.0),
            num_parallel_calls=1, deterministic=True,
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return data.with_options(options).prefetch(1)


# ---------------------------------------------------------------- metrikat

def group_metrics(rows, scores, threshold=0.5):
    predicted = np.asarray(scores).reshape(-1) >= threshold
    truth = np.asarray([LABELS[row["label"]] == 1 for row in rows])

    cells = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(row["split"], row["label"])].append(index)

    per_group, recalls = {}, defaultdict(list)
    for (split, label), indices in sorted(cells.items()):
        correct = int(np.sum(predicted[indices] == truth[indices]))
        rate = correct / len(indices)
        per_group[f"{split}/{label}"] = {
            "correct": correct, "total": len(indices), "rate": rate,
        }
        recalls[label].append(rate)

    balanced = float(np.mean([np.mean(values) for values in recalls.values()]))
    return {"group_balanced_accuracy": balanced, "per_group": per_group}


def show(title, result):
    print(f"\n{title}: grup bal.acc = {result['group_balanced_accuracy']:.4f}")
    for name, item in result["per_group"].items():
        print(f"    {name:<36} {item['correct']:>5}/{item['total']:<5} {item['rate']:>7.2%}")


# ---------------------------------------------------------------- trajnimi

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-split", default="train_flame_reviewed_v1")
    parser.add_argument("--validation-splits", nargs="+", default=list(VALIDATION_SPLITS),
                        help="Ndarjet e validimit te shqyrtuara per flake te dukshme.")
    parser.add_argument("--name", default="fire_flame_v6")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--fine-epochs", type=int, default=0)
    parser.add_argument("--views", type=int, default=2,
                        help="Pamje te augmentuara per foto, pervec origjinalit.")
    parser.add_argument("--augment-strength", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(name == "test_final" for name in [args.train_split, *args.validation_splits]):
        raise SystemExit("test_final nuk lejohet per trajnim ose perzgjedhje.")

    project = args.project_root.resolve()
    data_root = project / "dataset"
    splits_dir = data_root / "splits"
    artifacts = project / "backend" / "artifacts" / args.name

    train_rows = read_split(splits_dir, args.train_split)
    validation_rows = []
    for name in args.validation_splits:
        validation_rows.extend(read_split(splits_dir, name))

    check_disjoint({args.train_split: train_rows, "validation": validation_rows})

    print(f"Trajnim:  {len(train_rows)} foto")
    for key, count in summary(train_rows).items():
        print(f"    {key:<40} {count}")
    print(f"\nValidim:  {len(validation_rows)} foto")
    for name in args.validation_splits:
        subset = [row for row in validation_rows if row["split"] == name]
        print(f"    {name:<28} {len(subset):>6}  "
              f"{dict(sorted(Counter(r['label'] for r in subset).items()))}")

    weights = sample_weights(train_rows)
    print(f"\nPeshat: min {weights.min():.4f}, max {weights.max():.4f}, "
          f"shuma {weights.sum():.1f}")

    if args.dry_run:
        print("\n--dry-run: nuk u trajnua asgje.")
        return

    if artifacts.exists() and (artifacts / "model.keras").exists():
        raise SystemExit(
            f"Ekziston {artifacts / 'model.keras'}. "
            "Perdor nje --name tjeter; asnje model nuk u mbishkrua."
        )
    artifacts.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    tf.keras.utils.set_random_seed(42)

    model, extractor, head = build(tf)
    labels = np.asarray(
        [LABELS[row["label"]] for row in train_rows], dtype=np.float32
    )

    print("\nNxjerrja e vecorive (origjinali)...")
    features = [extractor.predict(
        dataset(tf, train_rows, data_root, args.batch_size), verbose=1
    )]
    for view in range(args.views):
        print(f"Nxjerrja e vecorive (pamja e augmentuar {view + 1}/{args.views})...")
        tf.keras.utils.set_random_seed(100 + view)
        features.append(extractor.predict(
            dataset(tf, train_rows, data_root, args.batch_size,
                    augment=augmenter(tf, args.augment_strength)),
            verbose=1,
        ))

    print("Nxjerrja e vecorive (validim)...")
    validation_features = extractor.predict(
        dataset(tf, validation_rows, data_root, args.batch_size), verbose=1
    )

    repeats = len(features)
    train_features = np.concatenate(features)
    train_labels = np.tile(labels, repeats)
    train_weights = np.tile(weights, repeats)
    del features

    baseline = group_metrics(
        validation_rows, head.predict(validation_features, batch_size=256, verbose=0)
    )
    show("Para trajnimit", baseline)

    history = []
    best = {"score": -1.0}

    class Track(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            scores = self.model.predict(validation_features, batch_size=256, verbose=0)
            result = group_metrics(validation_rows, scores)
            record = {
                "epoch": epoch + 1,
                "train_loss": float(logs.get("loss", float("nan"))),
                "validation": result,
            }
            history.append(record)
            improved = result["group_balanced_accuracy"] > best["score"] + 1e-9
            show(f"Epoka {epoch + 1}" + ("  (me e mira)" if improved else ""), result)
            if improved:
                best.update(score=result["group_balanced_accuracy"], record=record)
                model.save(artifacts / "model.keras")
            logs["val_group"] = result["group_balanced_accuracy"]
            (artifacts / "history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    head.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    print(f"\nTrajnimi i kokes ({repeats} pamje x {len(train_rows)} foto)...")
    head.fit(
        train_features, train_labels, sample_weight=train_weights,
        batch_size=128, epochs=args.head_epochs, shuffle=True, verbose=2,
        callbacks=[
            Track(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_group", mode="max", patience=8
            ),
        ],
    )

    if best["score"] < 0:
        raise SystemExit("Asnje epoke nuk perfundoi.")

    (artifacts / "result.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_split": args.train_split,
        "train_count": len(train_rows),
        "validation_splits": list(args.validation_splits),
        "architecture": (
            "MobileNetV2 (ImageNet, i ngrire) -> concat(GlobalAvgPool, GlobalMaxPool) "
            "-> Dropout(0.35) -> Dense(1, sigmoid, L2 1e-4)"
        ),
        "augmentation": (
            f"{args.views} pamje te augmentuara per foto: flip horizontal+vertikal, "
            f"rrotullim 0.08, shkelqim/kontrast {0.25 * args.augment_strength:.2f}"
        ),
        "selection_rule": (
            "Mesatarja e saktesise per klase mbi grupet e validimit, prag 0.5. "
            "test_final nuk u perdor ne asnje moment."
        ),
        "baseline_before_training": baseline,
        "best_epoch": best["record"],
        "limitation": (
            "Development validation is used for checkpoint selection, not an independent "
            "test. Correlated frames and shared recordings limit generalization claims. "
            "Consult the data preparation report for the selected split."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModeli:  {artifacts / 'model.keras'}")
    print(f"Raporti: {artifacts / 'result.json'}")
    print("\ntest_final nuk u prek. Per prove perfundimtare duhet regjistrim i ri.")


if __name__ == "__main__":
    main()
