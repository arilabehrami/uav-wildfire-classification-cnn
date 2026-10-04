from ..ml.classifier import FireClassifier


class ClassificationService:
    def __init__(self, classifier: FireClassifier):
        self.classifier = classifier

    def classify(self, image_bytes: bytes) -> dict:
        if not image_bytes:
            raise ValueError("Imazhi është bosh.")

        return self.classifier.classify(image_bytes)