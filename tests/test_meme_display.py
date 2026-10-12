'''Tests for app.meme_display. Fake picker and overlay: no files, no pixels.'''

import numpy as np

from app.meme_picker import Meme
from app.meme_display import MemeDisplay
from app.placement import PlacementTracker

FACE = (100, 200, 100, 100)


class FakePicker:
    def __init__(self, memes_by_label):
        self._memes = memes_by_label
        self.picks = []

    def pick(self, label):
        self.picks.append(label)
        return self._memes.get(label)


class FakeOverlay:
    def __init__(self):
        self.draws = []

    def draw(self, frame, image, placement):
        self.draws.append((image, placement))


def _meme(name, size=(10, 10)):
    return Meme(path=name, image=np.zeros((*size, 4), dtype=np.uint8))


def _display(**memes):
    picker, overlay = FakePicker(memes), FakeOverlay()
    return MemeDisplay(picker, PlacementTracker(), overlay), picker, overlay


def test_a_meme_is_drawn_exactly_when_an_expression_is_confirmed():
    happy = _meme("happy")
    display, _, overlay = _display(happy=happy)

    display.draw(None, None, FACE)
    assert overlay.draws == []

    display.draw(None, "happy", FACE)
    assert [image for image, _ in overlay.draws] == [happy.image]


def test_the_meme_is_chosen_once_per_confirmation_and_held():
    display, picker, _ = _display(happy=_meme("happy"))

    for _ in range(30):
        display.draw(None, "happy", FACE)

    assert picker.picks == ["happy"]


def test_a_new_confirmed_expression_picks_a_new_meme():
    happy, sad = _meme("happy"), _meme("sad")
    display, picker, overlay = _display(happy=happy, sad=sad)

    display.draw(None, "happy", FACE)
    display.draw(None, "sad", FACE)

    assert picker.picks == ["happy", "sad"]
    assert overlay.draws[-1][0] is sad.image


def test_clearing_and_reconfirming_the_same_label_picks_again():
    display, picker, _ = _display(happy=_meme("happy"))

    display.draw(None, "happy", FACE)
    display.draw(None, None, FACE)  # the vote failed
    display.draw(None, "happy", FACE)

    assert picker.picks == ["happy", "happy"]


def test_a_label_with_no_meme_draws_nothing_and_is_not_asked_again():
    display, picker, overlay = _display()

    for _ in range(5):
        display.draw(None, "neutral", FACE)

    assert overlay.draws == []
    assert picker.picks == ["neutral"]


def test_losing_the_face_hides_the_meme_and_the_placement_snaps_back():
    happy = _meme("happy")
    display, _, overlay = _display(happy=happy)
    display.draw(None, "happy", (0, 200, 100, 100))
    display.draw(None, "happy", (0, 200, 100, 100))

    display.draw(None, None, None)  # no face: the smoother also reports no expression
    display.draw(None, "happy", (400, 300, 100, 100))

    assert len(overlay.draws) == 3
    snapped = overlay.draws[-1][1]
    assert snapped == PlacementTracker().update((400, 300, 100, 100), 1.0)


def test_the_placement_uses_the_memes_aspect_ratio():
    wide = _meme("wide", size=(10, 20))  # height 10, width 20 -> aspect 0.5
    display, _, overlay = _display(happy=wide)

    display.draw(None, "happy", FACE)

    placement = overlay.draws[0][1]
    assert placement.height == placement.width // 2
