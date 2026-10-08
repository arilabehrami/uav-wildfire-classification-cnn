from pathlib import Path
import tensorflow as tf

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
)

IMAGE_SIZE = (254, 254)
# Development calibration: >=95% recall in each Fire source group.
# This is a sigmoid operating point, not a calibrated probability.
THRESHOLD = 0.02003161795437336
MODEL_VERSION = "mobilenet_v3_smoke_development_v1"


class FireClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            MODEL_DIR / "fire_mobilenet_mixed_v3_finetuned.keras",
            compile=False,
        )
        if tuple(self.model.input_shape[1:]) != (*IMAGE_SIZE, 3) or self.model.output_shape[-1] != 1:
            raise ValueError("Modeli duhet të pranojë RGB 254x254 dhe të japë një score sigmoid.")

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

        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        batch = tf.expand_dims(image, axis=0)

        fire_score = float(self.model(batch, training=False)[0, 0])
        label = "Fire" if fire_score >= THRESHOLD else "No_Fire"

        return {
            "label": label,
            "fire_score": fire_score,
            "threshold": THRESHOLD,
            "model_version": MODEL_VERSION,
        }
