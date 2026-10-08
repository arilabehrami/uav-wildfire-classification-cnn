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

NAME = "brightness_maxpool_temporal"


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
        image = tf.image.random_brightness(
            image, max_delta=0.15, seed=42
        )
        image = tf.clip_by_value(image, 0.0, 1.0)

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

    model_path = MODEL_DIR / f"fire_cnn_{NAME}.keras"

    if model_path.exists():
        raise FileExistsError(f"Modeli ekziston: {model_path}")

    rows = read_rows("train_temporal")
    validation_rows = read_rows("validation_temporal")

    train_paths = {row["path"] for row in rows}
    validation_paths = {row["path"] for row in validation_rows}

    if train_paths & validation_paths:
        raise ValueError("Ka foto të përbashkëta në train dhe validation.")

    counts = Counter(row["label"] for row in rows)

    if any(counts[label] == 0 for label in LABELS):
        raise ValueError("Trajnimi duhet t'i ketë të dyja klasat.")

    base_weights = {
        label: len(rows) / (2 * counts[label])
        for label in LABELS
    }

    # Merret vetëm arkitektura, pa peshat e mësuara.
    template = tf.keras.models.load_model(
        MODEL_DIR / "fire_cnn_v2.keras", compile=False
    )
    architecture = template.get_config()

    changed = 0
    for layer in architecture["layers"]:
        if layer["class_name"] == "GlobalAveragePooling2D":
            layer["class_name"] = "GlobalMaxPooling2D"
            layer["config"]["name"] = "global_max_pooling2d"
            changed += 1

    if changed != 1:
        raise ValueError("Pritej saktësisht një shtresë average pooling.")

    del template

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = tf.keras.Sequential.from_config(architecture)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    train_data = build_train_data(rows, base_weights)
    validation_data = load_dataset("validation_temporal")

    print(f"\nEksperimenti: {NAME}")
    print(f"Train: {len(rows)} foto — {dict(counts)}")
    print(f"Validation: {len(validation_rows)} foto")
    print(f"Parametra: {model.count_params()}")

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
                str(REPORT_DIR / f"training_{NAME}.csv")
            ),
        ],
    )

    with (REPORT_DIR / f"history_{NAME}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history.history, file, indent=2)

    print(f"Modeli u ruajt te: {model_path}")


if __name__ == "__main__":
    main()