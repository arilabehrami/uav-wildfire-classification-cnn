"""Shtresa e sherbimit: klasifikim plus shpjegim, me shpjegimin si opsional.

Nese Explainer nuk arrin te ndaje grafin, klasifikimi vazhdon pa harte.
Arsyeja logohet njehere, ne nisje, qe mungesa e hartes te mos mbetet e heshtur.
"""

from __future__ import annotations

import logging

from ..ml.classifier import FireClassifier
from ..ml.explain import Explainer

logger = logging.getLogger("fire.api")


class ClassificationService:
    def __init__(self, classifier: FireClassifier):
        self.classifier = classifier
        # Ndarja e grafit behet nje here, ne nisje.
        self.explainer = Explainer(classifier.model)
        if self.explainer.available:
            logger.info(
                "Grad-CAM aktiv dhe i verifikuar: shtresa=%s rrjet=%s pooling=%s",
                self.explainer.layer,
                self.explainer.shape,
                self.explainer.pooling,
            )
        else:
            logger.warning(
                "Grad-CAM i padisponueshem, sherbimi vazhdon pa harte: %s",
                self.explainer.reason,
            )

    def classify(self, image_bytes: bytes) -> dict:
        if not image_bytes:
            raise ValueError("Imazhi është bosh.")

        result = self.classifier.classify(image_bytes)

        explanation = self.explainer.heatmap(
            image_bytes, self.classifier.input_size
        )
        if explanation is not None:
            result["explanation"] = explanation

        return result
