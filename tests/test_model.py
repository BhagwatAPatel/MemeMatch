'''Tests for training.model's save/load helpers. No training involved.'''

import torch
import pytest

from training.model import ExpressionMLP, load_model, load_split_sessions, save_model


def test_held_out_sessions_round_trip_through_the_model_file(tmp_path):
    path = tmp_path / "model.pt"
    save_model(ExpressionMLP(2), ["a", "b"], path, test_session="s3", val_session="s2")

    assert load_split_sessions(path) == ("s3", "s2")
    _, labels = load_model(path)  # existing contract is unchanged
    assert labels == ["a", "b"]


def test_model_file_without_sessions_says_to_retrain(tmp_path):
    path = tmp_path / "old.pt"
    save_model(ExpressionMLP(2), ["a", "b"], path, test_session="s3", val_session="s2")
    bundle = torch.load(path, weights_only=False)
    del bundle["test_session"], bundle["val_session"]
    torch.save(bundle, path)

    with pytest.raises(ValueError, match="Retrain"):
        load_split_sessions(path)
