'''Unit tests for app.features.

No camera, no MediaPipe inference: we build FaceResult objects by hand.
That's the point of unit test - it runs anywhere in milliseconds.
'''

import numpy as np
import pytest

from app.face_tracker import FaceResult
from app.features import FEATURE_NAMES, NUM_FEATURES, extract_features, top_features

def make_face(blendshapes, names=None) -> FaceResult:
    '''Build a FaceResult with dummy landmarks/box; only blendshapes matter here.'''
    return FaceResult(
        landmarks=np.zeros((478, 3), dtype=np.float32),
        blendshapes=np.asarray(blendshapes),
        blendshape_names=FEATURE_NAMES if names is None else names,
        box=(0, 0, 10, 10),
    )

def test_feature_name_list_has_52_entries():
    assert NUM_FEATURES == 52
    # no duplicates: a duplicate name would mean two columns with one label.
    assert len(set(FEATURE_NAMES)) == 52

def test_extract_returns_float32_vecor_of_52():
    face = make_face(np.linspace(0.0, 1.0, 52))
    feats = extract_features(face)
    assert feats.shape == (52,)
    assert feats.dtype == np.float32

def test_extract_clips_out_of_range_values():
    scores = np.zeros(52)
    scores[0] = -0.01 # slightly below 0
    scores[1] = 1.02 # slightly above 1
    feats = extract_features(make_face(scores))
    assert feats[0] == 0.0
    assert feats[1] == 1.0

def test_wrong_length_raises():
    with pytest.raises(ValueError):
        extract_features(make_face(np.zeros(51)))

def test_wrong_name_order_raises():
    swapped = FEATURE_NAMES.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        extract_features(make_face(np.zeros(52), names=swapped))

def test_top_features_orders_strongest_first():
    scores = np.zeros(52)
    scores[FEATURE_NAMES.index("jawOpen")] = 0.9
    scores[FEATURE_NAMES.index("mouthSmileLeft")] = 0.7
    scores[FEATURE_NAMES.index("browInnerUp")] = 0.4
    top = top_features(make_face(scores), n=3)
    assert [name for name, _ in top] == ["jawOpen", "mouthSmileLeft", "browInnerUp"]

def test_extract_returns_a_copy():
    face = make_face(np.zeros(52))
    feats = extract_features(face)
    assert feats is not face.blendshapes