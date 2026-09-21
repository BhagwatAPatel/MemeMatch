'''Tests for training.evaluate. A stub model with hand-picked logits gives
outputs whose metrics can be worked out on paper.'''

import math
import re
import sys

import numpy as np
import pytest
import torch
from torch import nn

from app.features import NUM_FEATURES
from training.dataset import Dataset
from training import evaluate, train as train_module
from training.evaluate import evaluate_model
from training.model import ExpressionMLP, save_model
from training.samples import SampleStore
from training.train import train
from training.dataset import load_dataset


class FixedLogits(nn.Module):
    '''Ignores its input and returns the given logits, one row per sample.'''

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self.logits


def _data_with_test_labels(y_test):
    empty_X, empty_y = torch.zeros((0, NUM_FEATURES)), torch.zeros(0, dtype=torch.long)
    return Dataset(
        empty_X, empty_y, empty_X, empty_y,
        torch.zeros((len(y_test), NUM_FEATURES)),
        torch.tensor(y_test, dtype=torch.long),
        ["calm", "loud"], "s3", "s2",
    )


L3, L9, L4 = math.log(3), math.log(9), math.log(4)  # softmax top-prob .75, .9, .8

# true label, logits            -> predicts, confidence
#   0     [ln3, 0]              -> 0 (right), .75
#   0     [ln9, 0]              -> 0 (right), .90
#   0     [0, ln3]              -> 1 (WRONG), .75
#   1     [0, ln3]              -> 1 (right), .75
#   1     [0, ln9]              -> 1 (right), .90
#   1     [ln4, 0]              -> 0 (WRONG), .80
TRUE = [0, 0, 0, 1, 1, 1]
LOGITS = [[L3, 0], [L9, 0], [0, L3], [0, L3], [0, L9], [L4, 0]]


def test_accuracy_recall_and_confusion_matrix():
    report = evaluate_model(FixedLogits(LOGITS), _data_with_test_labels(TRUE))

    assert report.accuracy == pytest.approx(4 / 6)
    assert report.recall == {"calm": pytest.approx(2 / 3), "loud": pytest.approx(2 / 3)}
    # rows = true label, columns = predicted label
    np.testing.assert_array_equal(report.confusion, [[2, 1], [1, 2]])
    assert report.precision == {"calm": pytest.approx(2 / 3), "loud": pytest.approx(2 / 3)}


def test_precision_is_none_for_a_label_the_model_never_predicts():
    always_calm = [[L3, 0]] * 3
    report = evaluate_model(FixedLogits(always_calm), _data_with_test_labels([0, 0, 1]))

    assert report.recall == {"calm": pytest.approx(1.0), "loud": 0.0}
    assert report.precision == {"calm": pytest.approx(2 / 3), "loud": None}


def test_mean_confidence_of_correct_vs_wrong_predictions():
    report = evaluate_model(FixedLogits(LOGITS), _data_with_test_labels(TRUE))

    assert report.confidence_correct == pytest.approx(0.825)
    assert report.confidence_wrong == pytest.approx(0.775)


def test_confidence_wrong_is_none_when_nothing_is_wrong():
    right = [[L3, 0], [0, L3]]
    report = evaluate_model(FixedLogits(right), _data_with_test_labels([0, 1]))

    assert report.confidence_wrong is None
    assert report.confidence_correct == pytest.approx(0.75)


def _write_samples(path, test_counts):
    '''Three sessions; the last is the test session with the given per-label
    counts. Label "calm" sits near 0.2 on every feature, "loud" near 0.8.'''
    store = SampleStore(path)
    rng = np.random.default_rng(0)
    plan = {"s1": {"calm": 50, "loud": 50}, "s2": {"calm": 20, "loud": 20}, "s3": test_counts}
    for session, counts in plan.items():
        for label, n in counts.items():
            centre = 0.2 if label == "calm" else 0.8
            for _ in range(n):
                store.append(label, centre + 0.05 * rng.standard_normal(NUM_FEATURES), session)


def _always_calm_model():
    model = ExpressionMLP(num_classes=2)
    for p in model.parameters():
        nn.init.zeros_(p)
    model.net[-1].bias.data = torch.tensor([5.0, 0.0])  # "calm" always wins
    return model


def _save(model, path):
    save_model(model, ["calm", "loud"], path, test_session="s3", val_session="s2")


def _run_main(monkeypatch, csv_path, model_path):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--data", str(csv_path), "--model", str(model_path)])
    return evaluate.main()


def test_exit_zero_when_both_bars_are_met(tmp_path, monkeypatch, capsys):
    csv_path, model_path = tmp_path / "samples.csv", tmp_path / "model.pt"
    _write_samples(csv_path, {"calm": 20, "loud": 20})
    model = train(load_dataset(csv_path)).model
    _save(model, model_path)

    assert _run_main(monkeypatch, csv_path, model_path) == 0
    assert "accuracy" in capsys.readouterr().out.lower()


def test_exit_nonzero_when_accuracy_is_below_85_percent(tmp_path, monkeypatch, capsys):
    csv_path, model_path = tmp_path / "samples.csv", tmp_path / "model.pt"
    _write_samples(csv_path, {"calm": 20, "loud": 20})  # always-calm gets 50%
    _save(_always_calm_model(), model_path)

    assert _run_main(monkeypatch, csv_path, model_path) == 1
    assert "accuracy" in capsys.readouterr().out.lower()


def test_exit_nonzero_when_one_class_recall_is_below_70_percent(tmp_path, monkeypatch, capsys):
    csv_path, model_path = tmp_path / "samples.csv", tmp_path / "model.pt"
    _write_samples(csv_path, {"calm": 90, "loud": 10})  # always-calm: 90% accuracy, 0% loud recall
    _save(_always_calm_model(), model_path)

    assert _run_main(monkeypatch, csv_path, model_path) == 1
    out = capsys.readouterr().out
    assert "loud" in out and "recall" in out.lower()


def test_evaluate_scores_the_sessions_the_model_was_trained_to_hold_out(
    tmp_path, monkeypatch, capsys
):
    csv_path, model_path = tmp_path / "samples.csv", tmp_path / "model.pt"
    store = SampleStore(csv_path)
    rng = np.random.default_rng(0)
    plan = {"s1": {"calm": 13, "loud": 7}, "s2": {"calm": 20, "loud": 20}, "s3": {"calm": 30, "loud": 30}}
    for session, counts in plan.items():
        for label, n in counts.items():
            centre = 0.2 if label == "calm" else 0.8
            for _ in range(n):
                store.append(label, centre + 0.05 * rng.standard_normal(NUM_FEATURES), session)

    # Train holding out s1 (not the newest session), then evaluate with no flags.
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(csv_path), "--out", str(model_path),
                                      "--test-session", "s1", "--val-session", "s2"])
    assert train_module.main() == 0
    capsys.readouterr()
    _run_main(monkeypatch, csv_path, model_path)
    out = capsys.readouterr().out

    row_sums = {
        label: sum(int(n) for n in re.search(rf"^\s*{label}\s+((?:\d+\s*)+)$", out, re.M).group(1).split())
        for label in ("calm", "loud")
    }
    assert row_sums == {"calm": 13, "loud": 7}  # s1's rows, not the newest session's 30/30
