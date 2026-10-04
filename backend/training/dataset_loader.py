import csv
from pathlib import Path
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
IMAGE_SIZE = (254, 254)
BATCH_SIZE = 16
LABELS = {"No_Fire": 0, "Fire": 1}


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, tf.cast(label, tf.float32)


def load_dataset(split):
    paths, labels = [], []

    with (DATA_DIR / "splits" / f"{split}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            paths.append(str(DATA_DIR / row["path"]))
            labels.append(LABELS[row["label"]])

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split in ["train", "train_experiment", "train_v2"]:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    for split in ["train", "validation"]:
        dataset = load_dataset(split)
        images, labels = next(iter(dataset))
        print(f"{split}: images={images.shape}, labels={labels.shape}")

def load_weighted_train():
    from collections import Counter

    with (DATA_DIR / "splits/train_v2.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    counts = Counter(row["label"] for row in rows)
    total = len(rows)

    paths, labels, weights = [], [], []

    for row in rows:
        label = row["label"]
        path = DATA_DIR / row["path"]

        # Balancimi mes Fire dhe No_Fire.
        weight = total / (2 * counts[label])

        # Theks shtesë për pamjet pa zjarr që nuk janë liqen.
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
        len(paths), seed=42, reshuffle_each_iteration=True
    )
    dataset = dataset.map(
        load_sample, num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)