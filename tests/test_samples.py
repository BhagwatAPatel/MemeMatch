'''Tests for training.samples.SampleStore. No camera involved.'''

import csv

import numpy as np
import pytest

from app.features import FEATURE_NAMES, NUM_FEATURES
from training.samples import SampleStore


def test_append_then_load_round_trips(tmp_path):
    store = SampleStore(tmp_path / "samples.csv")
    happy = np.linspace(0, 1, NUM_FEATURES, dtype=np.float32)
    store.append("happy", happy)
    store.append("sad", np.zeros(NUM_FEATURES, dtype=np.float32))

    X, labels = store.load()

    assert labels == ["happy", "sad"]
    assert X.shape == (2, NUM_FEATURES)
    assert X.dtype == np.float32
    np.testing.assert_allclose(X[0], happy, rtol=1e-6)


def test_header_written_once_across_sessions(tmp_path):
    path = tmp_path / "samples.csv"
    SampleStore(path).append("happy", np.zeros(NUM_FEATURES))
    SampleStore(path).append("angry", np.zeros(NUM_FEATURES))  # second "session"

    rows = list(csv.reader(path.open()))
    assert sum(1 for r in rows if r[0] == "label") == 1
    assert rows[0] == ["label", *FEATURE_NAMES]
    assert SampleStore(path).counts() == {"happy": 1, "angry": 1}


def test_counts_on_missing_file_is_empty(tmp_path):
    assert SampleStore(tmp_path / "nope.csv").counts() == {}


def test_clear_removes_samples(tmp_path):
    store = SampleStore(tmp_path / "samples.csv")
    store.append("happy", np.zeros(NUM_FEATURES))
    store.clear()
    assert store.counts() == {}
    store.clear()  # clearing an already-empty store is fine


def test_load_missing_file_hints_at_collect(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.collect"):
        SampleStore(tmp_path / "nope.csv").load()


def test_load_rejects_header_from_other_feature_layout(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("label,a,b\nhappy,0.1,0.2\n")
    with pytest.raises(ValueError, match="feature layout"):
        SampleStore(path).load()


def test_append_rejects_stale_header(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("label,a,b\nhappy,0.1,0.2\n")
    with pytest.raises(ValueError, match="feature layout"):
        SampleStore(path).append("sad", np.zeros(NUM_FEATURES))


def test_append_rejects_wrong_length_features(tmp_path):
    with pytest.raises(ValueError, match=str(NUM_FEATURES)):
        SampleStore(tmp_path / "samples.csv").append("happy", np.zeros(3))


def test_load_reports_line_number_of_malformed_row(tmp_path):
    path = tmp_path / "samples.csv"
    store = SampleStore(path)
    store.append("happy", np.zeros(NUM_FEATURES))
    with path.open("a") as f:
        f.write("sad,not-a-number\n")

    with pytest.raises(ValueError, match="line 3"):
        store.load()


def test_load_rejects_row_with_wrong_column_count(tmp_path):
    path = tmp_path / "samples.csv"
    store = SampleStore(path)
    store.append("happy", np.zeros(NUM_FEATURES))
    with path.open("a") as f:
        f.write("sad,0.1,0.2\n")

    with pytest.raises(ValueError, match=r"line 3.*columns"):
        store.load()


def test_empty_file_is_an_empty_store(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("")
    store = SampleStore(path)

    X, labels = store.load()

    assert X.shape == (0, NUM_FEATURES) and labels == []
    assert store.counts() == {}


def test_append_to_empty_file_writes_header(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("")
    SampleStore(path).append("happy", np.zeros(NUM_FEATURES))

    assert next(csv.reader(path.open())) == ["label", *FEATURE_NAMES]


def test_counts_rejects_stale_header(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("label,a,b\nhappy,0.1,0.2\n")
    with pytest.raises(ValueError, match="feature layout"):
        SampleStore(path).counts()
