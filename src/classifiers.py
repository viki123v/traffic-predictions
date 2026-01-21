import numpy as np

class WeightedAccidentsClassifier:
    def __init__(self, pipeline, mapping):
        self.pipeline = pipeline
        self.mapping = mapping

    def predict(self, x: np.ndarray):
        return [self.mapping[p] for p in self.pipeline.predict(x)]
