import csv
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import DATA_DIR, LABELS, BATCH_SIZE, load_image

MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts/models"
SOURCE = MODEL_DIR / "fire_mobilenet_temporal.keras"
OUTPUT = MODEL_DIR / "fire_mobilenet_mixed_finetuned.keras"


def load_split(name, training=False):
    with (DATA_DIR / "splits" / f"{name}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Lista {name} është bosh.")

    paths = [str(DATA_DIR / row["path"]) for row in rows]
    labels = [LABELS[row["label"]] for row in rows]

    if training:
        counts = Counter(row["label"] for row in rows)
        weights = []

        for row in rows:
            label = row["label"]
            weight = len(rows) / (2 * counts[label])

            if label == "No_Fire":
                if not Path(row["path"]).name.startswith("lake_"):
                    weight *= 2.0

            weights.append(weight)

        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels, weights)
        )
        dataset = dataset.shuffle(
            len(rows), seed=42, reshuffle_each_iteration=True
        )

        def read_sample(path, label, weight):
            image, label = load_image(path, label)
            return image, label, tf.cast(weight, tf.float32)

        dataset = dataset.map(
            read_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    tf.keras.utils.set_random_seed(42)

    if OUTPUT.exists():
        raise FileExistsError(f"Modeli ekziston: {OUTPUT}")

    model = tf.keras.models.load_model(SOURCE, compile=False)

    backbones = [
        layer for layer in model.layers
        if layer.name.startswith("mobilenetv2")
    ]
    if len(backbones) != 1:
        raise ValueError("Nuk u identifikua MobileNetV2.")

    base = backbones[0]
    base.trainable = True

    # Përshtaten vetëm shtresat e fundit.
    for layer in base.layers[:-20]:
        layer.trainable = False

    # Ruhen statistikat e BatchNormalization.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    train_data = load_split("train_mixed", training=True)
    validation_data = load_split("validation_mixed")

    print("\nKontrolli para fine-tuning:")
    baseline = model.evaluate(
        validation_data, return_dict=True
    )

    # Ruhet edhe pika fillestare, nëse trajnimi nuk ndihmon.
    model.save(OUTPUT)

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=3,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT),
                monitor="val_loss",
                save_best_only=True,
                initial_value_threshold=baseline["loss"],
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
            ),
        ],
    )

    best_model = tf.keras.models.load_model(
        OUTPUT, compile=False
    )

    print(f"\nModeli më i mirë u ruajt te: {OUTPUT}")

    print("\nKontrolli në 202 fotot shtesë:")
    extra_data = load_split("validation_extra")

    labels = np.concatenate([
        batch_labels.numpy() for _, batch_labels in extra_data
    ])
    probabilities = best_model.predict(extra_data).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    for name, value in LABELS.items():
        mask = labels == value
        correct = int(np.sum(predictions[mask] == value))
        total = int(np.sum(mask))

        print(f"{name}: {correct}/{total} saktë ({correct / total:.2%})")


if __name__ == "__main__":
    main()