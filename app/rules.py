'''A hand-written rule baseline for the five expressions (SPEC section 7).

The expression model is compared against this on screen: if a handful of
blendshape thresholds do as well, the model isn't earning its keep. The
thresholds are first guesses, not tuned.

Pipeline position: Features -> [rules] -> HUD (comparison only)
'''

import numpy as np

from app.features import FEATURE_NAMES

_RULE_THRESHOLD = 0.4


def _score(features: np.ndarray, *names: str) -> float:
    return float(np.mean([features[FEATURE_NAMES.index(n)] for n in names]))


def label_by_rules(features: np.ndarray) -> str:
    '''Label a feature vector by blendshape thresholds. First matching rule wins.'''
    # Surprise needs the mouth AND the brows: an open mouth alone is just talking.
    if min(_score(features, "jawOpen"), _score(features, "browInnerUp")) > _RULE_THRESHOLD:
        return "surprised"
    if _score(features, "mouthSmileLeft", "mouthSmileRight") > _RULE_THRESHOLD:
        return "happy"
    if _score(features, "browDownLeft", "browDownRight") > _RULE_THRESHOLD:
        return "angry"
    if _score(features, "mouthFrownLeft", "mouthFrownRight") > _RULE_THRESHOLD:
        return "sad"
    return "neutral"
