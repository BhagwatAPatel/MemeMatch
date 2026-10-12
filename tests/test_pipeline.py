'''Tests for app.pipeline. Fake tracker, classifier, smoother and meme
display: no camera, no MediaPipe model, no trained model.'''

import numpy as np

from app.face_tracker import FaceResult
from app.features import FEATURE_NAMES, NUM_FEATURES
from app.pipeline import FramePipeline
from app.prediction import Prediction

BOX = (100, 200, 80, 90)
HAPPY = Prediction("happy", 0.9)


def _face(**scores):
    '''A FaceResult with all blendshapes at 0 except the named ones.'''
    blendshapes = np.zeros(NUM_FEATURES, dtype=np.float32)
    for name, score in scores.items():
        blendshapes[FEATURE_NAMES.index(name)] = score
    return FaceResult(
        landmarks=np.zeros((478, 3), dtype=np.float32),
        blendshapes=blendshapes,
        blendshape_names=list(FEATURE_NAMES),
        box=BOX,
    )


class FakeTracker:
    def __init__(self, face):
        self._face = face
        self.frames = []

    def process(self, frame):
        self.frames.append(frame.copy())
        return self._face


class FakeClassifier:
    def __init__(self, prediction=HAPPY):
        self._prediction = prediction
        self.inputs = []

    def predict(self, features):
        self.inputs.append(features)
        return self._prediction


class FakeSmoother:
    def __init__(self, confirmed):
        self._confirmed = confirmed
        self.updates = []

    def update(self, prediction, now):
        self.updates.append((prediction, now))
        return self._confirmed


class FakeMemeDisplay:
    def __init__(self):
        self.draws = []

    def draw(self, frame, confirmed, face_box):
        self.draws.append((frame, confirmed, face_box))


def _pipeline(face, confirmed=None):
    tracker, classifier = FakeTracker(face), FakeClassifier()
    smoother, display = FakeSmoother(confirmed), FakeMemeDisplay()
    pipeline = FramePipeline(tracker, classifier, smoother, display)
    return pipeline, tracker, classifier, smoother, display


def _frame():
    '''A 4x6 frame whose left column is white, so mirroring is visible.'''
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0] = 255
    return frame


def test_the_frame_is_mirrored_before_the_tracker_sees_it():
    pipeline, tracker, *_ = _pipeline(face=None)

    result = pipeline.process(_frame(), now=0.0)

    seen = tracker.frames[0]
    assert (seen[:, -1] == 255).all() and (seen[:, 0] == 0).all()
    assert (result.frame == seen).all()


def test_no_face_tells_the_smoother_and_the_meme_display_there_is_none():
    pipeline, _, classifier, smoother, display = _pipeline(face=None)

    result = pipeline.process(_frame(), now=5.0)

    assert classifier.inputs == []
    assert smoother.updates == [(None, 5.0)]
    assert [(confirmed, box) for _, confirmed, box in display.draws] == [(None, None)]
    assert (result.face, result.prediction, result.rule_label, result.confirmed) == (
        None, None, None, None,
    )


def test_a_face_is_classified_from_its_feature_vector_and_fed_to_the_smoother():
    face = _face(mouthSmileLeft=0.8, mouthSmileRight=0.8)
    pipeline, _, classifier, smoother, _ = _pipeline(face)

    result = pipeline.process(_frame(), now=7.5)

    assert classifier.inputs[0].shape == (NUM_FEATURES,)
    assert smoother.updates == [(HAPPY, 7.5)]
    assert result.face is face
    assert result.prediction == HAPPY
    assert result.rule_label == "happy"  # app.rules: smile blendshapes above 0.4


def test_the_meme_display_gets_the_confirmed_expression_not_the_raw_prediction():
    pipeline, _, _, _, display = _pipeline(_face(), confirmed="sad")

    result = pipeline.process(_frame(), now=0.0)

    frame, confirmed, box = display.draws[0]
    assert confirmed == "sad"  # the classifier said "happy"
    assert box == BOX
    assert result.confirmed == "sad"


def test_the_meme_is_drawn_onto_the_frame_that_is_returned():
    pipeline, _, _, _, display = _pipeline(_face(), confirmed="happy")

    result = pipeline.process(_frame(), now=0.0)

    assert display.draws[0][0] is result.frame
