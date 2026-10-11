"""Kontrata e hyrjes, e lexuar nga inference.json dhe e zbatuar ne nje vend.

Trajnimi, vleresimi dhe sherbimi duhet ta perpunojne imazhin njesoj. Çdo
ndryshim ketu e prish krahasueshmerine me rezultatet e raportuara.
"""

import tensorflow as tf


def decode(image_bytes: bytes) -> tf.Tensor:
    """Bytes -> tensor RGB. Ngre ValueError nese fajlli nuk eshte imazh."""
    if not image_bytes:
        raise ValueError("Imazhi është bosh.")

    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
    except (tf.errors.InvalidArgumentError, ValueError) as error:
        raise ValueError("Fajlli nuk mund të lexohet si imazh.") from error

    image.set_shape([None, None, 3])
    return image


def prepare_batch(image_bytes: bytes, input_size: tuple[int, int]) -> tf.Tensor:
    """Nje imazh -> batch (1, H, W, 3) ne [0,1], bilinear, njesoj si ne trajnim."""
    image = tf.cast(decode(image_bytes), tf.float32)
    image = tf.image.resize(image, input_size) / 255.0
    return tf.expand_dims(image, axis=0)
