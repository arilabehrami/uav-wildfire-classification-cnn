import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import DATA_DIR, LABELS, BATCH_SIZE, load_image

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

SOURCE = ARTIFACTS / "models/fire_mobilenet_temporal.keras"
OUTPUT = ARTIFACTS / "models/fire_mobilenet_mixed_v2_finetuned.keras"

HISTORY = ARTIFACTS / "metrics/mobilenet_mixed_v2_history.csv"
REPORT = ARTIFACTS / "metrics/mobilenet_mixed_v2_results.json"


def load_split(name, training=False):
    with (DATA_DIR / "splits" / f"{name}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    counts = Counter(row["label"] for row in rows)

    if not rows or set(counts) != set(LABELS):
        raise ValueError(
            f"Lista {name} duhet t'i kete te dyja klasat."
        )

    paths = [
        str(DATA_DIR / row["path"])
        for row in rows
    ]

    labels = np.asarray(
        [LABELS[row["label"]] for row in rows],
        dtype=np.float32,
    )

    if training:
        weights = []

        for row in rows:
            weight = len(rows) / (2 * counts[row["label"]])

            if row["label"] == "No_Fire":
                if not Path(row["path"]).name.startswith("lake_"):
                    weight *= 2.0

            weights.append(weight)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                paths,
                labels,
                np.asarray(weights, dtype=np.float32),
            )
        )

        dataset = dataset.shuffle(
            len(rows),
            seed=42,
            reshuffle_each_iteration=True,
        )

        def read_sample(path, label, weight):
            image, label = load_image(path, label)
            return image, label, weight

        dataset = dataset.map(
            read_sample,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        dataset = dataset.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    dataset = dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )

    return rows, labels, dataset


class DomainValidation(tf.keras.callbacks.Callback):
    """Validimi i vjeter dhe ai shtese kane peshe te barabarte."""

    def __init__(self, extra_validation):
        super().__init__()
        self.extra_validation = extra_validation
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise RuntimeError("Mungojne metrikat e epokes.")

        extra = self.model.evaluate(
            self.extra_validation,
            verbose=0,
            return_dict=True,
        )

        old_loss = float(logs["val_loss"])
        old_accuracy = float(logs["val_accuracy"])

        score = (
            old_loss + float(extra["loss"])
        ) / 2.0

        if not np.isfinite(score):
            raise FloatingPointError(
                "Validation loss nuk eshte numer i fundem."
            )

        logs["val_extra_loss"] = float(extra["loss"])
        logs["val_extra_accuracy"] = float(extra["accuracy"])
        logs["val_domain_loss"] = score

        self.records.append({
            "epoch": epoch + 1,
            "old_loss": old_loss,
            "old_accuracy": old_accuracy,
            "extra_loss": float(extra["loss"]),
            "extra_accuracy": float(extra["accuracy"]),
            "domain_loss": score,
        })

        print(
            f"\nValidim: i vjeter={old_loss:.4f}, "
            f"shtese={extra['loss']:.4f}, "
            f"mesatarja={score:.4f}"
        )


def main():
    tf.keras.utils.set_random_seed(42)

    if LABELS != {"No_Fire": 0, "Fire": 1}:
        raise ValueError(f"Kontrollo LABELS: {LABELS}")

    for path in (OUTPUT, HISTORY, REPORT):
        if path.exists():
            raise FileExistsError(
                f"Ekziston: {path}. "
                "Asnje rezultat nuk u mbishkrua."
            )

        path.parent.mkdir(parents=True, exist_ok=True)

    train_rows, _, train_data = load_split(
        "train_mixed_v2",
        training=True,
    )

    old_rows, _, old_validation = load_split(
        "validation_temporal"
    )

    extra_rows, _, extra_validation = load_split(
        "validation_extra_v2"
    )

    holdout_rows, holdout_labels, holdout_data = load_split(
        "holdout_extra_v2"
    )

    seen = set()

    for rows in (
        train_rows,
        old_rows,
        extra_rows,
        holdout_rows,
    ):
        paths = [row["path"] for row in rows]

        if (
            len(paths) != len(set(paths))
            or seen.intersection(paths)
        ):
            raise ValueError(
                "Ka foto te perseritura brenda ose mes listave."
            )

        seen.update(paths)

    images, _, _ = next(iter(train_data))

    tf.debugging.assert_all_finite(
        images,
        "Imazhe me NaN ose infinity.",
    )

    if tuple(images.shape[1:]) != (254, 254, 3):
        raise ValueError(
            "load_image duhet te jape (254,254,3): "
            f"{images.shape}"
        )

    low = float(tf.reduce_min(images))
    high = float(tf.reduce_max(images))

    if low < -1e-6 or high > 1.000001:
        raise ValueError(
            "load_image duhet te jape [0,1]; "
            f"u gjet [{low}, {high}]."
        )

    model = tf.keras.models.load_model(
        SOURCE,
        compile=False,
    )

    initial_weights = model.get_weights()

    backbones = [
        layer
        for layer in model.layers
        if layer.name.startswith("mobilenetv2")
    ]

    if len(backbones) != 1:
        raise ValueError("Nuk u identifikua MobileNetV2.")

    base = backbones[0]
    base.trainable = True

    for index, layer in enumerate(base.layers):
        layer.trainable = (
            index >= len(base.layers) - 20
            and not isinstance(
                layer,
                tf.keras.layers.BatchNormalization,
            )
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(
        "\nVleresimi i modelit fillestar ne te dy validimet:"
    )

    baseline_old = {
        key: float(value)
        for key, value in model.evaluate(
            old_validation,
            verbose=0,
            return_dict=True,
        ).items()
    }

    baseline_extra = {
        key: float(value)
        for key, value in model.evaluate(
            extra_validation,
            verbose=0,
            return_dict=True,
        ).items()
    }

    baseline_score = (
        baseline_old["loss"] + baseline_extra["loss"]
    ) / 2.0

    if not np.isfinite(baseline_score):
        raise FloatingPointError(
            "Loss i modelit fillestar nuk eshte i fundem."
        )

    print(
        f"Baseline: i vjeter={baseline_old['loss']:.4f}, "
        f"shtese={baseline_extra['loss']:.4f}, "
        f"mesatarja={baseline_score:.4f}"
    )

    domain_validation = DomainValidation(extra_validation)

    model.fit(
        train_data,
        validation_data=old_validation,
        epochs=3,
        callbacks=[
            domain_validation,

            # Ruhet kandidati pas trajnimit.
            # Modeli fillestar mbetet te SOURCE.
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT),
                monitor="val_domain_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor="val_domain_loss",
                mode="min",
                patience=2,
            ),

            tf.keras.callbacks.CSVLogger(
                str(HISTORY)
            ),
        ],
    )

    best_record = min(
        domain_validation.records,
        key=lambda record: record["domain_loss"],
    )

    candidate = tf.keras.models.load_model(
        OUTPUT,
        compile=False,
    )

    final_weights = candidate.get_weights()

    identical = (
        len(initial_weights) == len(final_weights)
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                initial_weights,
                final_weights,
            )
        )
    )

    print(
        "\nPeshat e kandidatit identike me baseline: "
        f"{identical}"
    )
    print(
        f"Kandidati pas fine-tuning u ruajt: {OUTPUT}"
    )

    # Zgjedhja behet vetem sipas validimit.
    if best_record["domain_loss"] < baseline_score:
        selected_path = OUTPUT
        selected = candidate
    else:
        selected_path = SOURCE
        selected = tf.keras.models.load_model(
            SOURCE,
            compile=False,
        )

    print(
        "Modeli i zgjedhur sipas validimit: "
        f"{selected_path.name}"
    )

    print(
        f"\nKontrolli ne {len(holdout_rows)} "
        "fotot e rezervuara:"
    )

    probabilities = selected.predict(
        holdout_data,
        verbose=1,
    ).reshape(-1)

    if (
        len(probabilities) != len(holdout_labels)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("Dalje te pavlefshme nga modeli.")

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    per_class = {}

    for name, value in LABELS.items():
        mask = holdout_labels == value

        correct = int(
            np.sum(predictions[mask] == value)
        )
        total = int(mask.sum())

        per_class[name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }

        print(
            f"{name}: {correct}/{total} sakte "
            f"({correct / total:.2%})"
        )

    accuracy = float(
        np.mean(predictions == holdout_labels)
    )

    print(f"Gjithsej: {accuracy:.2%}")

    report = {
        "source": str(SOURCE),
        "trained_candidate": str(OUTPUT),
        "candidate_weights_identical_to_source": identical,
        "selected_model": str(selected_path),

        "selection_rule": (
            "mean of old-domain and extra-domain "
            "validation BCE losses, equal weights"
        ),

        "baseline": {
            "old": baseline_old,
            "extra": baseline_extra,
            "domain_loss": baseline_score,
        },

        "best_finetuned_epoch": best_record,
        "history": domain_validation.records,

        "split_counts": {
            "train": len(train_rows),
            "validation_old": len(old_rows),
            "validation_extra": len(extra_rows),
            "holdout": len(holdout_rows),
        },

        "holdout": {
            "accuracy": accuracy,
            "per_class": per_class,
            "threshold": 0.5,
        },

        "limitation": (
            "The retained extra images were inspected "
            "in an earlier experiment. This is a temporal "
            "comparison on known sources, not a new "
            "independent-source final test."
        ),
    }

    with REPORT.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Raporti: {REPORT}")


if __name__ == "__main__":
    main()