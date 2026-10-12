'''One frame through the whole pipeline: camera frame in, annotated frame out.

Wires the stages together and nothing else. The frame is mirrored first, so
the preview behaves like a mirror and the landmarks line up with what is
drawn. No face means the smoother hears "no prediction" and the meme display
hears "no face box". The meme display is driven by the confirmed expression,
never by the raw prediction.

Pipeline position: Camera -> [FramePipeline] -> HUD + display
'''

from dataclasses import dataclass

import cv2
import numpy as np

from app.expression_classifier import ExpressionClassifier
from app.face_tracker import FaceResult, FaceTracker
from app.features import extract_features
from app.meme_display import MemeDisplay
from app.prediction import Prediction
from app.rules import label_by_rules
from app.smoothing import Smoother


@dataclass(frozen=True)
class FrameResult:
    '''What one frame produced. `frame` is mirrored and has the meme drawn on it.'''

    frame: np.ndarray
    face: FaceResult | None
    prediction: Prediction | None
    rule_label: str | None
    confirmed: str | None


class FramePipeline:
    def __init__(
        self,
        tracker: FaceTracker,
        classifier: ExpressionClassifier,
        smoother: Smoother,
        meme_display: MemeDisplay,
    ):
        self._tracker = tracker
        self._classifier = classifier
        self._smoother = smoother
        self._meme_display = meme_display

    def process(self, frame: np.ndarray, now: float) -> FrameResult:
        '''Run one camera frame through every stage.

        now is a monotonic timestamp in seconds, passed in so the smoother's
        hold time can be tested without a clock.
        '''
        # Must happen BEFORE tracking so landmarks line up with what is drawn.
        frame = cv2.flip(frame, 1)

        face = self._tracker.process(frame)
        prediction = rule_label = None
        if face is not None:
            features = extract_features(face)
            prediction = self._classifier.predict(features)
            rule_label = label_by_rules(features)
        confirmed = self._smoother.update(prediction, now)
        self._meme_display.draw(frame, confirmed, face.box if face else None)
        return FrameResult(frame, face, prediction, rule_label, confirmed)
