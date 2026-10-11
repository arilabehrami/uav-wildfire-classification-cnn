"""Offline checks: preprocessing, fine-tuning and ensemble serialization."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fire_final as core


class XceptionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = core.tensorflow()

    def test_xception_and_saved_ensemble(self):
        tf = self.tf
        # No downloads and no training images needed for this contract check.
        xception, extractor, head = core.build_models(tf, weights=None, architecture="xception")
        self.assertEqual(extractor.output_shape[-1], 2048)
        self.assertEqual(head.input_shape[-1], 2048)
        probe = tf.random.uniform((1, 254, 254, 3), seed=42)
        preprocessing = tf.keras.Model(xception.input, xception.get_layer("xception_scaling").output)
        scaled = preprocessing(probe).numpy()
        self.assertEqual(scaled.shape, (1, 299, 299, 3))
        self.assertGreaterEqual(scaled.min(), -1)
        self.assertLessEqual(scaled.max(), 1)
        core.enable_finetuning(tf, xception, architecture="xception")
        base = xception.get_layer("xception")
        self.assertFalse(base.get_layer("block12_sepconv1").trainable)
        self.assertTrue(base.get_layer("block13_sepconv1").trainable)
        self.assertTrue(all(not layer.trainable for layer in base.layers
                            if isinstance(layer, tf.keras.layers.BatchNormalization)))
        mobile, _, _ = core.build_models(tf, weights=None)
        inputs = tf.keras.Input(shape=(254, 254, 3))
        branches = []
        for name, model in (("mobilenet", mobile), ("xception", xception)):
            branch_input = tf.keras.Input(shape=(254, 254, 3))
            branch = tf.keras.Model(branch_input, model(branch_input, training=False), name=f"{name}_branch")
            branches.append(branch(inputs, training=False))
        ensemble = tf.keras.Model(inputs, tf.keras.layers.Average()(branches))
        expected = (mobile(probe, training=False).numpy() + xception(probe, training=False).numpy()) / 2
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "ensemble.keras"
            ensemble.save(destination)
            restored = tf.keras.models.load_model(destination, compile=False)
            np.testing.assert_allclose(restored(probe, training=False).numpy(), expected,
                                       rtol=1e-5, atol=1e-6)
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
