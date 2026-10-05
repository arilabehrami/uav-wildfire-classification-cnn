from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset_loader import (
    DATA_DIR, LABELS, BATCH_SIZE, load_image,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    paths, labels = [], []

    for label, value in LABELS.items():
        for path in sorted((DATA_DIR / "Test" / label).glob("*.jpg")):
            paths.append(str(path))
            labels.append(value)

    if not paths:
        raise ValueError("Nuk u gjetën imazhe te Test.")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(
        ARTIFACTS_DIR / "models" / "fire_cnn_compare_brightness_maxpool.keras"
    )

    probabilities = model.predict(dataset).reshape(-1)
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["No_Fire", "Fire"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels, predictions, labels=[0, 1]
    )

    print(f"\nTest accuracy: {accuracy_score(labels, predictions):.2%}")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["No_Fire", "Fire"],
        zero_division=0,
    ))
    print("Confusion matrix — rendi: No_Fire, Fire")
    print(matrix)

    output_dir = ARTIFACTS_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "test_results_brightness_maxpool.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "threshold": threshold,
            "report": report,
            "confusion_matrix": matrix.tolist(),
        }, file, indent=2)

    np.savez(
        output_dir / "test_predictions_brightness_maxpool.npz",
        paths=np.array(paths),
        labels=np.array(labels),
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()