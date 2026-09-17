'''Turn a FaceResult into the feature vector the expression model uses

This file is imported by BOTH the live app and the training scripts.
That is deliberate: if both sides call the same function, the model can
never be trained on features laid out differently from the ones it sees
at runtime (a bug known as train/serve skew).

Pipeline position: FaceTracker -> [features] -> Classifier / CSV
'''

import numpy as np

from app.face_tracker import FaceResult

# The 52 blendshape names MediaPipe's Face Landmarker outputs, in ITS order.
# Hard-coded on purpose. This list is the contract: CSV column order, model
# input order, and live infernce all follow it. If MediaPipe ever changes
# the order, extract_features() will raise instead of feeding the model
# scrambled inputs.

FEATURE_NAMES: list[str] = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

NUM_FEATURES = len(FEATURE_NAMES) # 52

def extract_features(face: FaceResult) -> np.ndarray:
    '''Return the model input for one face: shape (52,), float32, values 0-1.

    Raise ValueError if the tracker's output doesn't match FEATURE_NAMES,
    because a silent mismatch here would poison every prediction downstream.
    '''

    scores = np.asarray(face.blendshapes, dtype=np.float32)

    if scores.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Expected {NUM_FEATURES} blendshapes, got shape {scores.shape}"
        )

    if list(face.blendshape_names) != FEATURE_NAMES:
        raise ValueError(
            "Blendshape names/order from the tracker differ from FEATURE_NAMES. "
            "MediaPipe may have changed; update FEATURE_NAMES and retain."
        )

    # MediaPipe documents scores as 0-1 but float noise can nudge them a hair 
    # outside. Clipping keeps the model's input range exactly as promised.
    return np.clip(scores, 0.0, 1.0)

def top_features(face: FaceResult, n: int = 3) -> list[tuple[str, float]]:
    '''The n strongest blendshapes as (name, score), strongest first.

    Debug helper for the HUD; not used by the model
    '''

    feats = extract_features(face)
    # argsort is ascending, so [::-1] reverses it, then [:n] takes the top n.
    order = np.argsort(feats)[::-1][:n]
    return [(FEATURE_NAMES[i], float(feats[i])) for i in order]

