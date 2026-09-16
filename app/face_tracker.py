'''
Face detection and landmark extraction.

Wraps MediaPipe's Face Landmarker so the rest of MemeMatch never touches 
MediaPipe directly. Each frame in -> one FaceResult out (or None).

Pipeline position: Camera -> [FaceTracker] -> Features -> Classifer
'''

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np 
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Resolved relative to this, so it works no matter which folder
# you launch the app from.

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent/ "models" / "face_landmarker.task"

@dataclass
class FaceResult:
    '''Everything we know about the one face in a frame.

    landmarks: (478, 3) array of x, y, z. x and y are NORMALISED(0.0-1.0
                fractions of frame width/height), z is relative depth. 
    blendshapes: (52,) array of expression score, each 0.0-1.0 
                (e.g. jawOpen, mouthSmileLeft, browDownRight).
    blendshape_names: the 52 names in the same order as 'blendshapes'.
    box:            (x, y, w, h) bounding box in PIXELS, derived from landmarks.
    '''

    landmarks: np.ndarray
    blendshapes: np.ndarray
    blendshape_names: list[str]
    box: tuple[int, int, int, int]

class FaceTracker:
    '''Runs MediaPipe Face Landmarker on a stream of video frames.'''

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}."
                "Download it as described in README.md"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode = frames arrive in order with timestamps, so MediaPipe
            # can track between frames instead of re-detecting from scratch.
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        # VIDEO mode demands strictly increasing millisecond timestamps.
        self._start = time.monotonic()
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray) -> FaceResult | None:
        '''Detect the face in one frame. Returns None when no face is found.'''
        # OpenCV gives BGR, MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((time.monotonic() - self._start) * 1000)
        if ts_ms <= self._last_ts_ms:
            # Two frames inside the same millisecond would make MediaPipe 
            # raise an error, so nudge the timestamp forward.
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None

        # Convert MediaPipe's list of landmark objects into a plain array
        landmarks = np.array(
            [[p.x, p.y, p.z] for p in result.face_landmarks[0]],
            dtype=np.float32,
        )

        categories = result.face_blendshapes[0]
        blendshapes = np.array([c.score for c in categories], dtype=np.float32)
        names = [c.category_name for c in categories]

        return FaceResult(
            landmarks=landmarks,
            blendshapes=blendshapes,
            blendshape_names=names,
            box=self._bounding_box(landmarks, frame_bgr.shape),
        )

    @staticmethod
    def _bounding_box(landmarks: np.ndarray, frame_shape) -> tuple[int, int, int, int]:
        '''Pixel box around all landmarks (used later to place the meme).'''
        h, w = frame_shape[:2]
        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1 - x0, y1 - y0

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

def draw_face_debug(frame_bgr: np.ndarray, face: FaceResult) -> None:
    '''Draw landmarks and bounding box onto frame, in place.'''
    h, w = frame_bgr.shape[:2]
    for x, y, _ in face.landmarks:
        cv2.circle(frame_bgr, (int(x * w), int(y * h)), 1, (0, 255, 255), -1)

    x, y, bw, bh = face.box 
    cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (255, 0, 0), 1)

    

    