'''Temporal smoothing: per-frame predictions -> a stable confirmed expression.

A single frame's prediction is jittery. The Smoother only confirms a label
once it dominates the recent window, so the meme doesn't flicker.

Pipeline position: ExpressionClassifier -> [Smoother] -> meme engine
'''

from collections import Counter, deque

from app.expression_classifier import Prediction


class Smoother:
    def __init__(self, window: int = 10, threshold: float = 0.70, hold: float = 1.5):
        self._window = window
        self._threshold = threshold
        self._hold = hold
        self._recent: deque[Prediction] = deque(maxlen=window)
        self._confirmed: str | None = None
        self._confirmed_at = 0.0

    def update(self, prediction: Prediction | None, now: float) -> str | None:
        '''Feed one frame's prediction (None = no face); return the confirmed expression.'''
        if prediction is None:
            self._recent.clear()
            self._confirmed = None
            return None
        self._recent.append(prediction)
        label, votes = Counter(p.label for p in self._recent).most_common(1)[0]
        mean_confidence = sum(p.confidence for p in self._recent if p.label == label) / votes
        if votes > self._window / 2 and mean_confidence >= self._threshold:
            if label != self._confirmed and self._may_switch(now):
                self._confirmed = label
                self._confirmed_at = now
        return self._confirmed

    def _may_switch(self, now: float) -> bool:
        return self._confirmed is None or now - self._confirmed_at >= self._hold
