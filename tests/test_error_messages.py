'''Tests for the startup error messages. No camera or model is opened.'''

from pathlib import Path

import pytest

from app import camera
from app.camera import Camera, CameraError
from app.face_tracker import FaceTracker


class _ClosedCapture:
    '''Stands in for cv2.VideoCapture when the camera cannot be opened.'''

    def __init__(self, *_args):
        pass

    def isOpened(self):
        return False


def test_an_unopenable_camera_gives_a_readable_message(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", _ClosedCapture)

    with pytest.raises(CameraError) as error:
        Camera(3)

    assert str(error.value) == (
        "Could not open camera 3. Check that it is connected and not in use "
        "by another app, and that Terminal has Camera permissions "
        "(System Settings > Privacy & Security > Camera)."
    )


def test_a_missing_face_model_gives_a_readable_message(tmp_path):
    missing = tmp_path / "face_landmarker.task"

    with pytest.raises(FileNotFoundError) as error:
        FaceTracker(missing)

    assert str(error.value) == (
        f"Face landmarker model not found at {missing}. "
        "Download it as described in README.md."
    )
