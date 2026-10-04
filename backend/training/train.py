from pathlib import Path
from collections import Counter
import csv
import json

import tensorflow as tf

from cnn import build_cnn
from dataset_loader import load_dataset, DATA_DIR, LABELS
from dataset_loader import (
    load_dataset, load_weighted_train, DATA_DIR, LABELS
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"


def main():
    tf.keras.utils.set_random_seed(42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = load_weighted_train()
    validation_data = load_dataset("validation_v2")

    # Peshat llogariten vetëm nga pjesa train.
    with (DATA_DIR / "splits" / "train_v2.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        counts = Counter(row["label"] for row in csv.DictReader(file))

    total = sum(counts.values())
    class_weights = {
        LABELS[label]: total / (2 * counts[label])
        for label in LABELS
    }

    model = build_cnn()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "fire_cnn_v4.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            str(REPORT_DIR / "training_log_v4.csv")
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=10,
        callbacks=callbacks,
    )

    with (REPORT_DIR / "history_v4.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history.history, file, indent=2)

    print(f"Modeli më i mirë u ruajt te: {MODEL_DIR / 'fire_cnn_v4.keras'}")


if __name__ == "__main__":
    main()