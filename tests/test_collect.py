'''Tests for training.collect.main's startup flow. No camera involved:
Camera is replaced with one that fails, so main() returns right after
the reset / validation steps it is being tested on.'''

import re
import sys
from datetime import datetime

import pytest

from training import collect
from training.samples import HEADER

STALE = "label,a,b\nhappy,0.1,0.2\n"


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    def fail(*_args, **_kwargs):
        raise collect.CameraError("no camera in tests")
    monkeypatch.setattr(collect, "Camera", fail)


def run(monkeypatch, path, *flags, answer="y"):
    monkeypatch.setattr(sys, "argv", ["collect", "--out", str(path), *flags])
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    return collect.main()


def test_stale_header_fails_fast_with_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "samples.csv"
    path.write_text(STALE)

    assert run(monkeypatch, path) == 1
    assert "feature layout" in capsys.readouterr().out


def test_reset_clears_stale_header_file(tmp_path, monkeypatch):
    path = tmp_path / "samples.csv"
    path.write_text(STALE)

    run(monkeypatch, path, "--reset")

    assert not path.exists()


def test_reset_clears_header_only_file(tmp_path, monkeypatch):
    path = tmp_path / "samples.csv"
    path.write_text(",".join(HEADER) + "\n")  # exists but has zero samples

    run(monkeypatch, path, "--reset")

    assert not path.exists()


def test_reset_declined_keeps_file(tmp_path, monkeypatch):
    path = tmp_path / "samples.csv"
    path.write_text(STALE)

    assert run(monkeypatch, path, "--reset", answer="n") == 0
    assert path.exists()


def test_default_session_id_is_a_readable_sortable_timestamp():
    early = collect.default_session_id(datetime(2026, 9, 21, 14, 30, 5))
    late = collect.default_session_id(datetime(2026, 9, 21, 14, 31, 0))

    assert early == "20260921-143005"
    assert early < late


def test_session_flag_overrides_the_timestamp_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["collect", "--session", "desk-evening"])
    assert collect.parse_args().session == "desk-evening"

    monkeypatch.setattr(sys, "argv", ["collect"])
    assert re.fullmatch(r"\d{8}-\d{6}", collect.parse_args().session)
