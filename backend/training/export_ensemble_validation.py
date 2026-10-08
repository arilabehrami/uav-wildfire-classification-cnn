import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import DATA_DIR, LABELS, load_dataset

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    with (DATA_DIR / "splits/validation_v2.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Lista e validimit është bosh.")

    labels = np.array([LABELS[row["label"]] for row in rows])
    paths = np.array([row["path"] for row in rows])

    model_names = {
        "brightness": "fire_cnn_compare_brightness.keras",
        "maxpool": "fire_cnn_compare_brightness_maxpool.keras",
    }

    scores = {}

    for name, filename in model_names.items():
        model = tf.keras.models.load_model(
            ARTIFACTS / "models" / filename,
            compile=False,
        )

        probabilities = model.predict(
            load_dataset("validation_v2")
        ).reshape(-1)

        if len(probabilities) != len(labels):
            raise ValueError(f"Numër i gabuar rezultatesh: {name}")

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Pikë të pavlefshme: {name}")

        scores[name] = probabilities
        del model
        tf.keras.backend.clear_session()

    output_dir = ARTIFACTS / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ensemble_validation_scores.npz"

    np.savez_compressed(
        output_path,
        paths=paths,
        labels=labels,
        brightness=scores["brightness"],
        maxpool=scores["maxpool"],
    )

    print(f"U ruajtën pikët për {len(labels)} foto.")
    print(f"Fajlli: {output_path}")


if __name__ == "__main__":
    main()