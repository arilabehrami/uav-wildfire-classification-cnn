import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import DATA_DIR, LABELS, load_image, BATCH_SIZE


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

MODEL_PATH = (
    ARTIFACTS
    / "models"
    / "fire_mobilenet_mixed_v3_finetuned.keras"
)

SPLIT_PATH = (
    DATA_DIR
    / "splits"
    / "holdout_extra_v3.csv"
)

OUTPUT_PATH = (
    ARTIFACTS
    / "inspection"
    / "holdout_v3_predictions.csv"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nuk u gjet modeli: {MODEL_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Nuk u gjet split-i: {SPLIT_PATH}"
        )

    with SPLIT_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    paths = [
        str(DATA_DIR / row["path"])
        for row in rows
    ]

    true_labels = np.asarray(
        [
            LABELS[row["label"]]
            for row in rows
        ],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            true_labels,
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    probabilities = model.predict(
        dataset,
        verbose=1,
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    reverse_labels = {
        value: key
        for key, value in LABELS.items()
    }

    results = []

    false_positive = []
    false_negative = []

    for row, true_value, predicted_value, probability in zip(
        rows,
        true_labels,
        predictions,
        probabilities,
    ):
        correct = (
            true_value == predicted_value
        )

        result = {
            "path": row["path"],
            "true_label": reverse_labels[
                int(true_value)
            ],
            "predicted_label": reverse_labels[
                int(predicted_value)
            ],
            "fire_probability": float(
                probability
            ),
            "correct": correct,
        }

        results.append(result)

        # No_Fire -> Fire
        if (
            true_value == LABELS["No_Fire"]
            and predicted_value == LABELS["Fire"]
        ):
            false_positive.append(result)

        # Fire -> No_Fire
        if (
            true_value == LABELS["Fire"]
            and predicted_value == LABELS["No_Fire"]
        ):
            false_negative.append(result)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "true_label",
                "predicted_label",
                "fire_probability",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n============================")
    print("REZULTATET E HOLDOUT V3")
    print("============================")

    print(
        f"\nFalse Positives "
        f"(No_Fire -> Fire): "
        f"{len(false_positive)}"
    )

    for item in false_positive:
        print(
            f"{item['path']} | "
            f"Fire probability: "
            f"{item['fire_probability']:.4f}"
        )

    print(
        f"\nFalse Negatives "
        f"(Fire -> No_Fire): "
        f"{len(false_negative)}"
    )

    for item in false_negative:
        print(
            f"{item['path']} | "
            f"Fire probability: "
            f"{item['fire_probability']:.4f}"
        )

    print(
        f"\nCSV u ruajt te:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()