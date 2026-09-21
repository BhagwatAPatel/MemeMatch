'''Tests for training.dataset.load_dataset, through a real SampleStore file.'''

import numpy as np
import pytest

from app.features import NUM_FEATURES
from training.dataset import load_dataset
from training.samples import SampleStore


def _fill(path, per_label):
    store = SampleStore(path)
    rng = np.random.default_rng(0)
    for label, n in per_label.items():
        for _ in range(n):
            store.append(label, rng.random(NUM_FEATURES))


def test_labels_sorted_and_split_is_stratified(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"sad": 10, "happy": 10})

    ds = load_dataset(path, test_size=0.2)

    assert ds.labels == ["happy", "sad"]
    assert ds.X_train.shape == (16, NUM_FEATURES) and ds.X_test.shape == (4, NUM_FEATURES)
    assert sorted(ds.y_test.tolist()) == [0, 0, 1, 1]
    assert ds.y_train.dtype.is_floating_point is False


def test_label_with_single_sample_is_rejected(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"happy": 10, "sad": 1})

    with pytest.raises(ValueError, match="sad"):
        load_dataset(path)
