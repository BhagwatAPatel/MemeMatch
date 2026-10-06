'''Tests for app.smoothing. Pure logic: timestamps are passed in, no clock.'''

from app.expression_classifier import Prediction
from app.smoothing import Smoother


def _feed(smoother, label, confidence, count, start=0.0, step=0.05):
    '''Feed `count` identical predictions; return the last confirmed expression.'''
    confirmed = None
    for i in range(count):
        confirmed = smoother.update(Prediction(label, confidence), start + i * step)
    return confirmed


def test_expression_is_confirmed_once_it_wins_a_strict_majority_of_the_window():
    smoother = Smoother(window=10)

    assert _feed(smoother, "happy", 0.9, 5) is None  # 5 of 10 is not a strict majority
    assert smoother.update(Prediction("happy", 0.9), 0.25) == "happy"  # 6 of 10


def test_a_majority_with_low_mean_confidence_is_not_confirmed():
    smoother = Smoother(window=10, threshold=0.70)

    assert _feed(smoother, "happy", 0.6, 10) is None


def test_losing_the_face_clears_the_expression_and_forgets_old_votes():
    smoother = Smoother(window=10)
    assert _feed(smoother, "happy", 0.9, 10) == "happy"

    assert smoother.update(None, 1.0) is None
    assert _feed(smoother, "happy", 0.9, 5, start=1.05) is None  # old votes are gone
    assert smoother.update(Prediction("happy", 0.9), 1.3) == "happy"


def test_switching_to_another_expression_waits_for_the_hold_time():
    smoother = Smoother(window=10, hold=1.5)
    assert _feed(smoother, "happy", 0.9, 6) == "happy"  # confirmed at t=0.25

    assert _feed(smoother, "angry", 0.9, 10, start=0.3) == "happy"  # t=0.3-0.75, too soon
    assert smoother.update(Prediction("angry", 0.9), 1.8) == "angry"  # held 1.55 s
