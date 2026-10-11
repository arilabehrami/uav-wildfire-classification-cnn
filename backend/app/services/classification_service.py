from ..ml.classifier import FireClassifier
from ..ml.explain import Explainer


class ClassificationService:
    def __init__(self, classifier: FireClassifier):
        self.classifier = classifier
        # Ndarja e grafit behet nje here, ne nisje. Nese deshton, Explainer
        # mbetet i padisponueshem dhe klasifikimi vazhdon pa harte.
        self.explainer = Explainer(classifier.model)

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
