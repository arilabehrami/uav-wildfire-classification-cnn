from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from dataset_loader import DATA_DIR, LABELS, load_dataset

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/models/fire_cnn_v5.keras"
)


def main():
    with (DATA_DIR / "splits/validation_v2.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    labels = np.array([LABELS[row["label"]] for row in rows])

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(
        load_dataset("validation_v2")
    ).reshape(-1)

    predictions = (probabilities >= 0.5).astype(int)

    print("\nRezultati në validation:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["No_Fire", "Fire"],
        zero_division=0,
    ))

    groups = {
        "Fire": [],
        "No_Fire — liqen": [],
        "No_Fire — të tjera": [],
    }

    for index, row in enumerate(rows):
        if row["label"] == "Fire":
            group = "Fire"
        elif Path(row["path"]).name.startswith("lake_"):
            group = "No_Fire — liqen"
        else:
            group = "No_Fire — të tjera"

        groups[group].append(index)

    for group, indices in groups.items():
        if not indices:
            continue

        correct = np.sum(predictions[indices] == labels[indices])
        total = len(indices)
        print(
            f"{group}: {correct}/{total} saktë "
            f"({correct / total:.2%})"
        )

        print("\nKrahasimi i kufijve në validation:")
    print("Kufiri | Fire saktë | No_Fire të tjera saktë | Liqen saktë")

    for threshold in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted = (probabilities >= threshold).astype(int)
        scores = []

        for group in ["Fire", "No_Fire — të tjera", "No_Fire — liqen"]:
            indices = groups[group]
            score = np.mean(predicted[indices] == labels[indices])
            scores.append(score)

        print(
            f"{threshold:.1f}    | "
            f"{scores[0]:.2%}     | "
            f"{scores[1]:.2%}                 | "
            f"{scores[2]:.2%}"
        )


if __name__ == "__main__":
    main()