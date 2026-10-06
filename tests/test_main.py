'''Tests for app.main's startup error handling. No camera is opened.'''

import sys

import pytest

from app import main


@pytest.mark.parametrize("error", [FileNotFoundError("No trained model"), ValueError("Retrain")])
def test_a_missing_or_stale_model_prints_the_message_and_exits_1(error, monkeypatch, capsys):
    def failing_classifier():
        raise error

    monkeypatch.setattr(main, "ExpressionClassifier", failing_classifier)
    monkeypatch.setattr(sys, "argv", ["app.main"])

    assert main.main() == 1
    assert str(error) in capsys.readouterr().err
