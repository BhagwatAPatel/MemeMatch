'''Tests for app.expression_classifier. Tiny saved models, no camera.'''

import numpy as np
import pytest
import torch

from app.expression_classifier import ExpressionClassifier
from app.features import NUM_FEATURES
from training.model import ExpressionMLP, save_model


def _model_that_always_says(label_index: int, labels: list[str]) -> ExpressionMLP:
    '''An MLP whose output ignores the input and strongly favours one label.'''
    model = ExpressionMLP(len(labels))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias[label_index] = 10.0  # logit gap of 10 -> softmax ~1.0
    return model


def _save(tmp_path, model, labels):
    path = tmp_path / "model.pt"
    save_model(model, labels, path, test_session="s3", val_session="s2")
    return path


def test_predict_returns_the_models_label_with_its_confidence(tmp_path):
    labels = ["angry", "happy", "sad"]
    path = _save(tmp_path, _model_that_always_says(1, labels), labels)

    prediction = ExpressionClassifier(path).predict(np.zeros(NUM_FEATURES, dtype=np.float32))

    assert prediction.label == "happy"
    assert prediction.confidence == pytest.approx(1.0, abs=1e-3)


def test_model_trained_on_a_different_feature_layout_is_rejected(tmp_path):
    labels = ["angry", "happy"]
    path = _save(tmp_path, _model_that_always_says(0, labels), labels)
    bundle = torch.load(path, weights_only=False)
    bundle["feature_names"] = bundle["feature_names"][::-1]
    torch.save(bundle, path)

    with pytest.raises(ValueError, match="feature layout"):
        ExpressionClassifier(path)


def test_a_corrupt_model_file_says_to_retrain(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a torch file")

    with pytest.raises(ValueError, match="Retrain"):
        ExpressionClassifier(path)


def test_a_model_file_missing_its_weights_says_to_retrain(tmp_path):
    labels = ["angry", "happy"]
    path = _save(tmp_path, _model_that_always_says(0, labels), labels)
    bundle = torch.load(path, weights_only=False)
    del bundle["state_dict"]
    torch.save(bundle, path)

    with pytest.raises(ValueError, match="Retrain"):
        ExpressionClassifier(path)
