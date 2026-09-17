'''Tests for the CSV side of training.collect. No camera involved.'''

import csv

import numpy as np

from app.features import NUM_FEATURES
from training.collect import SampleWriter, count_existing

def test_writer_creates_header_then_rows(tmp_path):
    # tmp_path is a pytest fixture: a fresh temporary folder per test,
    # deleted afterwards, so tests never touch real data / folder.

    out = tmp_path / "samples.csv"
    with SampleWriter(out) as w:
        w.write("happy", np.ones(NUM_FEATURES, dtype=np.float32))
        w.write("sad", np.zeros(NUM_FEATURES, dtype=np.float32))

    rows = list(csv.reader(out.open()))
    assert rows[0][0] == "label"
    assert len(rows[0]) == 1 + NUM_FEATURES
    assert rows[1][0] == "happy" and rows[2][0] == "sad"
    assert len(rows) == 3

def test_reopening_does_not_duplicate_header(tmp_path):
    out = tmp_path / "samples.csv"
    with SampleWriter(out) as w:
        w.write("happy", np.zeros(NUM_FEATURES))
    with SampleWriter(out) as w: # second "session"
        w.write("angry", np.zeros(NUM_FEATURES))

    rows = list(csv.reader(out.open()))
    assert sum(1 for r in rows if r[0] == "label") == 1
    assert count_existing(out) == {"happy": 1, "angry": 1}