'''Tests for training.dataset.load_dataset, through a real SampleStore file.'''

import numpy as np
import pytest

from app.features import NUM_FEATURES
from training.dataset import load_dataset
from training.samples import SampleStore

LABELS = {"happy": 3, "sad": 3}


def _fill(path, sessions):
    '''Write sessions in the given order. Every feature of a session's rows
    equals its 1-based position / 10 (0.1, 0.2, ...), so a row's value shows
    which session it came from.'''
    store = SampleStore(path)
    for i, (session, per_label) in enumerate(sessions.items(), start=1):
        for label, n in per_label.items():
            for _ in range(n):
                store.append(label, np.full(NUM_FEATURES, i / 10), session)


def _values(X):
    return set(np.round(X[:, 0].numpy().astype(np.float64), 1).tolist())


def test_last_session_is_test_and_the_one_before_is_validation(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"a": LABELS, "b": LABELS, "c": LABELS, "d": LABELS})

    data = load_dataset(path)

    assert data.labels == ["happy", "sad"]
    assert (data.test_session, data.val_session) == ("d", "c")
    assert _values(data.X_test) == {0.4}
    assert _values(data.X_val) == {0.3}
    assert _values(data.X_train) == {0.1, 0.2}
    assert len(data.y_test) == len(data.X_test) == 6
    assert sorted(data.y_val.tolist()) == [0, 0, 0, 1, 1, 1]


def test_sessions_can_be_chosen_explicitly(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"a": LABELS, "b": LABELS, "c": LABELS, "d": LABELS})

    data = load_dataset(path, test_session="a", val_session="c")

    assert (data.test_session, data.val_session) == ("a", "c")
    assert _values(data.X_test) == {0.1}
    assert _values(data.X_val) == {0.3}
    assert _values(data.X_train) == {0.2, 0.4}


def test_fewer_than_three_sessions_is_rejected(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"a": LABELS, "b": LABELS})

    with pytest.raises(ValueError, match=r"at least 3 sessions.*found 2"):
        load_dataset(path)


def test_label_missing_from_a_session_is_rejected_with_a_count_table(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"a": LABELS, "b": {"happy": 3}, "c": LABELS})

    with pytest.raises(ValueError) as err:
        load_dataset(path)

    message = str(err.value)
    assert "every label" in message
    assert "b" in message and "sad" in message
    assert "happy=3" in message and "sad=0" in message


def test_bad_session_overrides_are_rejected(tmp_path):
    path = tmp_path / "samples.csv"
    _fill(path, {"a": LABELS, "b": LABELS, "c": LABELS})

    with pytest.raises(ValueError, match="nope.*not a recorded session"):
        load_dataset(path, test_session="nope")
    with pytest.raises(ValueError, match="different sessions"):
        load_dataset(path, test_session="a", val_session="a")
