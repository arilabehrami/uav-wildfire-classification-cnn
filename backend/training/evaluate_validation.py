from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import DATA_DIR, LABELS, load_dataset

SPLIT = "validation_temporal"
THRESHOLD = 0.5

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "models"
    / "fire_cnn_brightness_maxpool_temporal.keras"
)


def main():
    with (DATA_DIR / "splits" / f"{SPLIT}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Lista e validimit është bosh.")

    labels = np.array([LABELS[row["label"]] for row in rows])

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(
        load_dataset(SPLIT)
    ).reshape(-1)

    if len(probabilities) != len(labels):
        raise ValueError("Numri i parashikimeve nuk përputhet me etiketat.")

    predictions = (probabilities >= THRESHOLD).astype(int)

    print(f"\nModeli: {MODEL_PATH.name}")
    print(f"Validimi: {SPLIT}, {len(rows)} foto")
    print(f"Pragu: {THRESHOLD}")

    print("\nRezultati në validation:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["No_Fire", "Fire"],
        zero_division=0,
    ))

    print("Confusion matrix — rendi: No_Fire, Fire")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

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

    print("\nRezultatet sipas grupeve:")

    for group, indices in groups.items():
        if not indices:
            print(f"{group}: nuk ka foto")
            continue

        correct = int(np.sum(predictions[indices] == labels[indices]))
        total = len(indices)

        print(
            f"{group}: {correct}/{total} saktë "
            f"({correct / total:.2%})"
        )

    print("\nKrahasimi i pragjeve në validation:")
    print("Pragu | Fire saktë | No_Fire të tjera saktë | Liqen saktë")

    for threshold in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted = (probabilities >= threshold).astype(int)
        scores = []

        for group in ["Fire", "No_Fire — të tjera", "No_Fire — liqen"]:
            indices = groups[group]

            if not indices:
                scores.append("N/A")
            else:
                score = np.mean(predicted[indices] == labels[indices])
                scores.append(f"{score:.2%}")

        print(
            f"{threshold:.1f}   | "
            f"{scores[0]:>10} | "
            f"{scores[1]:>23} | "
            f"{scores[2]:>11}"
        )


if __name__ == "__main__":
    main()