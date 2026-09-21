'''Tests for training.train.train, on small synthetic datasets. No camera,
no CSV: a Dataset is built directly so the expected outcome is known.'''

import numpy as np
import torch

from app.features import NUM_FEATURES
from training.dataset import Dataset
from training.train import train


def _blobs(n_per_class, seed, flip=False):
    '''Two well-separated classes: features near 0.2 (class 0) or 0.8 (class 1).
    flip=True swaps the labels, so the data contradicts an unflipped set.'''
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_per_class)
    X = 0.2 + 0.6 * y[:, None] + 0.05 * rng.standard_normal((len(y), NUM_FEATURES))
    if flip:
        y = 1 - y
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))


def _dataset(val_flip=False):
    X_train, y_train = _blobs(100, seed=1)
    X_val, y_val = _blobs(30, seed=2, flip=val_flip)
    X_test, y_test = _blobs(30, seed=3)
    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test, ["a", "b"], "s3", "s2")


def _accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def test_learns_a_separable_dataset():
    data = _dataset()

    result = train(data)

    assert _accuracy(result.model, data.X_val, data.y_val) >= 0.95
    assert not result.model.training  # returned in eval mode, ready for inference


def _weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_same_seed_gives_identical_weights():
    data = _dataset()

    first = train(data, epochs=5, seed=7).model
    second = train(data, epochs=5, seed=7).model
    other = train(data, epochs=5, seed=8).model

    assert torch.equal(_weights(first), _weights(second))
    assert not torch.equal(_weights(first), _weights(other))


def test_returns_the_best_validation_epoch_not_the_last():
    # Validation labels contradict training, so validation accuracy gets
    # worse the better the model learns: the last epoch is not the best one.
    data = _dataset(val_flip=True)

    result = train(data, epochs=20)
    model, history = result.model, result.history

    best_accuracy = max(h.val_accuracy for h in history)
    assert best_accuracy > history[-1].val_accuracy  # premise of the scenario
    assert _accuracy(model, data.X_val, data.y_val) == best_accuracy
    assert result.best in history
    assert result.best.val_accuracy == best_accuracy

    # Among epochs tied on accuracy, the returned one has the lowest val loss.
    with torch.no_grad():
        returned_loss = torch.nn.functional.cross_entropy(model(data.X_val), data.y_val).item()
    tied = [h.val_loss for h in history if h.val_accuracy == best_accuracy]
    assert returned_loss == min(tied)
    assert result.best.val_loss == min(tied)
