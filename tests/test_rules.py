'''Tests for app.rules, the hand-written baseline the model is compared with.'''

import numpy as np
import pytest

from app.features import FEATURE_NAMES, NUM_FEATURES
from app.rules import label_by_rules


def _features(**scores: float) -> np.ndarray:
    vector = np.zeros(NUM_FEATURES, dtype=np.float32)
    for name, score in scores.items():
        vector[FEATURE_NAMES.index(name)] = score
    return vector


@pytest.mark.parametrize("label, scores", [
    ("happy", dict(mouthSmileLeft=0.8, mouthSmileRight=0.8)),
    ("surprised", dict(jawOpen=0.7, browInnerUp=0.7)),
    ("angry", dict(browDownLeft=0.7, browDownRight=0.7)),
    ("sad", dict(mouthFrownLeft=0.7, mouthFrownRight=0.7)),
    ("neutral", dict()),
])
def test_each_expression_is_picked_from_its_blendshape_signature(label, scores):
    assert label_by_rules(_features(**scores)) == label


def test_an_open_mouth_alone_is_not_surprise():
    assert label_by_rules(_features(jawOpen=0.9)) == "neutral"
