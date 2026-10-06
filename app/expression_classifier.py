'''Live expression classification.

Wraps the trained expression model so the app sees one call: feature vector
in, Prediction out. The feature layout is owned by app.features and checked
by training.model.load_model, so train/serve skew can't creep in here.

Pipeline position: Features -> [ExpressionClassifier] -> Smoother
'''

from pathlib import Path

import numpy as np
import torch

from app.prediction import Prediction
from training.model import DEFAULT_MODEL_PATH, load_model


class ExpressionClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self._model, self._labels = load_model(model_path)

    def predict(self, features: np.ndarray) -> Prediction:
        '''Classify one feature vector of shape (52,).'''
        with torch.no_grad():
            logits = self._model(torch.from_numpy(features).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
        index = int(probs.argmax())
        return Prediction(self._labels[index], float(probs[index]))
