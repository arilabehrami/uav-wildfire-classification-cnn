from pathlib import Path
import tensorflow as tf

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts/models/fire_cnn_v2.keras"
)

IMAGE_SIZE = (254, 254)
THRESHOLD = 0.8


class FireClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            MODEL_PATH, compile=False
        )

    def classify(self, image_bytes: bytes) -> dict:
        try:
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )
        except (tf.errors.InvalidArgumentError, ValueError) as error:
            raise ValueError(
                "Fajlli nuk mund të lexohet si imazh."
            ) from error

        # I njëjti përpunim si gjatë trajnimit.
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        batch = tf.expand_dims(image, axis=0)

        fire_score = float(self.model(batch, training=False)[0, 0])

        return {
            "label": "Fire" if fire_score >= THRESHOLD else "No_Fire",
            "fire_score": fire_score,
            "threshold": THRESHOLD,
            "model_version": "v2",
        }