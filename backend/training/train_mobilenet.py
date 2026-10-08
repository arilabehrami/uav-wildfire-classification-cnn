import csv
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import DATA_DIR, LABELS, BATCH_SIZE, load_image

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_DIR = ARTIFACTS / "models"

MODEL_PATH = MODEL_DIR / "fire_mobilenet_temporal.keras"
HEAD_PATH = MODEL_DIR / "mobilenet_head_temporal.weights.h5"


def read_split(name):
    with (DATA_DIR / "splits" / f"{name}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Lista {name} është bosh.")

    paths = [str(DATA_DIR / row["path"]) for row in rows]
    labels = np.array(
        [LABELS[row["label"]] for row in rows],
        dtype=np.float32,
    )

    # Pa shuffle: veçoritë duhet të përputhen me etiketat.
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return rows, labels, dataset


def main():
    tf.keras.utils.set_random_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        raise FileExistsError(f"Modeli ekziston: {MODEL_PATH}")

    train_rows, train_labels, train_data = read_split("train_temporal")
    val_rows, val_labels, val_data = read_split("validation_temporal")

    train_paths = {row["path"] for row in train_rows}
    val_paths = {row["path"] for row in val_rows}

    if train_paths & val_paths:
        raise ValueError("Ka foto të përbashkëta në train dhe validation.")

    counts = Counter(row["label"] for row in train_rows)

    if any(counts[label] == 0 for label in LABELS):
        raise ValueError("Trajnimi duhet t'i ketë të dyja klasat.")

    weights = []

    for row in train_rows:
        label = row["label"]
        weight = len(train_rows) / (2 * counts[label])

        # I njëjti theks si në eksperimentin temporal.
        if label == "No_Fire":
            if not Path(row["path"]).name.startswith("lake_"):
                weight *= 2.0

        weights.append(weight)

    weights = np.array(weights, dtype=np.float32)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(254, 254, 3))

    x = tf.keras.layers.Resizing(224, 224)(inputs)

    # Loader-i jep [0, 1]; MobileNetV2 pret [-1, 1].
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)

    features = base(x, training=False)
    extractor = tf.keras.Model(inputs, features)

    print("\nNxjerrja e veçorive të trajnimit:")
    train_features = extractor.predict(train_data)

    print("\nNxjerrja e veçorive të validimit:")
    val_features = extractor.predict(val_data)

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\nTrajnimi i klasifikuesit Fire/No_Fire:")

    head.fit(
        train_features,
        train_labels,
        sample_weight=weights,
        validation_data=(val_features, val_labels),
        batch_size=32,
        epochs=30,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(HEAD_PATH),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
        ],
    )

    head.load_weights(HEAD_PATH)

    # Ruhet modeli i plotë: foto → MobileNetV2 → Fire/No_Fire.
    outputs = head(extractor.output)
    model = tf.keras.Model(extractor.input, outputs)
    model.save(MODEL_PATH)

    print(f"\nModeli u ruajt te: {MODEL_PATH}")


if __name__ == "__main__":
    main()