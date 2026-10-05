import csv
import json
from collections import Counter
from pathlib import Path

import tensorflow as tf

from dataset_loader import (
    DATA_DIR, LABELS, BATCH_SIZE, load_image, load_dataset,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"


def read_rows(split):
    with (DATA_DIR / "splits" / f"{split}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def build_train_data(rows, base_weights):
    paths, labels, weights = [], [], []

    for row in rows:
        path = DATA_DIR / row["path"]
        label = row["label"]
        weight = base_weights[label]

        if label == "No_Fire" and not path.name.startswith("lake_"):
            weight *= 2.0

        paths.append(str(path))
        labels.append(LABELS[label])
        weights.append(weight)

    def load_sample(path, label, weight):
        image, label = load_image(path, label)
        return image, label, tf.cast(weight, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels, weights)
    )
    dataset = dataset.shuffle(
        len(rows), seed=42, reshuffle_each_iteration=True
    )
    dataset = dataset.map(
        load_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("control", "train_v2"),
        ("extra", "train_v5"),
    ]

    for name, _ in experiments:
        if (MODEL_DIR / f"fire_cnn_compare_{name}.keras").exists():
            raise FileExistsError(
                f"Modeli compare_{name} ekziston. "
                "Mos e mbishkruaj pa e ruajtur."
            )

    base_rows = read_rows("train_v2")
    extra_rows = read_rows("train_v5")

    base_entries = {(r["path"], r["label"]) for r in base_rows}
    extra_entries = {(r["path"], r["label"]) for r in extra_rows}

    if not base_entries.issubset(extra_entries):
        raise ValueError("train_v5 nuk përmban të gjitha fotot e train_v2.")

    added = extra_entries - base_entries
    if not added or any(label != "No_Fire" for _, label in added):
        raise ValueError("Fotot shtesë duhet të jenë vetëm No_Fire.")

    counts = Counter(row["label"] for row in base_rows)
    base_weights = {
        label: len(base_rows) / (2 * counts[label])
        for label in LABELS
    }

    # Merret vetëm arkitektura e v2, jo peshat e mësuara.
    template = tf.keras.models.load_model(
        MODEL_DIR / "fire_cnn_v2.keras", compile=False
    )
    architecture = template.get_config()
    del template

    for name, split in experiments:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        model = tf.keras.Sequential.from_config(architecture)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        rows = read_rows(split)
        train_data = build_train_data(rows, base_weights)
        validation_data = load_dataset("validation_v2")

        model_path = MODEL_DIR / f"fire_cnn_compare_{name}.keras"

        print(f"\nEksperimenti: {name}")
        print(f"Foto: {len(rows)}, parametra: {model.count_params()}")

        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=10,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    str(model_path),
                    monitor="val_loss",
                    save_best_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.CSVLogger(
                    str(REPORT_DIR / f"training_compare_{name}.csv")
                ),
            ],
        )

        with (REPORT_DIR / f"history_compare_{name}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history.history, file, indent=2)

        print(f"Modeli u ruajt te: {model_path}")


if __name__ == "__main__":
    main()