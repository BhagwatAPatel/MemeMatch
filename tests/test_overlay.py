'''Tests for app.overlay. Synthetic images only: no camera, no files.'''

import cv2
import numpy as np
import pytest

from app.overlay import Overlay
from app.placement import Placement

FRAME_H, FRAME_W = 60, 80


def _frame(value=0):
    return np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)


def _meme(bgr=(255, 0, 0), alpha=255, size=(16, 16)):
    '''A flat-colour BGRA meme of (height, width).'''
    image = np.zeros((*size, 4), dtype=np.uint8)
    image[:, :] = (*bgr, alpha)
    return image


def test_an_opaque_meme_replaces_the_pixels_under_it_and_nothing_else():
    frame = _frame(10)

    Overlay().draw(frame, _meme(), Placement(x=20, y=10, width=16, height=16))

    assert (frame[10:26, 20:36] == (255, 0, 0)).all()
    assert (frame[:10] == 10).all() and (frame[:, :20] == 10).all()
    assert (frame[26:] == 10).all() and (frame[:, 36:] == 10).all()


def test_a_fully_transparent_meme_leaves_the_frame_untouched():
    frame = _frame(10)

    Overlay().draw(frame, _meme(alpha=0), Placement(x=20, y=10, width=16, height=16))

    assert (frame == 10).all()


def test_half_transparent_pixels_blend_with_the_frame():
    frame = _frame(100)

    Overlay().draw(frame, _meme(bgr=(200, 200, 200), alpha=128), Placement(20, 10, 16, 16))

    blended = frame[18, 28]
    assert (abs(blended.astype(int) - 150) <= 2).all()


def test_the_meme_is_scaled_to_the_placement_width_keeping_its_aspect():
    frame = _frame()

    # A 2:1 (tall) meme; the placement is what PlacementTracker would give it.
    Overlay().draw(frame, _meme(size=(16, 8)), Placement(x=10, y=20, width=16, height=32))

    ys, xs = np.nonzero((frame != 0).any(axis=2))
    assert (xs.min(), xs.max()) == (10, 25)  # 16 px wide, bucket-aligned
    assert (ys.min(), ys.max()) == (20, 51)  # 32 px tall, bottom edge at the placement's bottom


@pytest.mark.parametrize(
    "placement",
    [
        Placement(x=-8, y=10, width=16, height=16),   # off the left edge
        Placement(x=FRAME_W - 8, y=10, width=16, height=16),  # off the right edge
        Placement(x=20, y=-8, width=16, height=16),   # off the top edge
        Placement(x=20, y=FRAME_H - 8, width=16, height=16),  # off the bottom edge
        Placement(x=-8, y=-8, width=16, height=16),   # off a corner
        Placement(x=-10, y=-10, width=200, height=200),  # larger than the frame
    ],
)
def test_overflow_is_clipped_and_the_visible_part_is_drawn(placement):
    frame = _frame()

    Overlay().draw(frame, _meme(), placement)

    assert frame.shape == (FRAME_H, FRAME_W, 3)
    assert (frame == (255, 0, 0)).all(axis=2).any()


@pytest.mark.parametrize(
    "placement",
    [
        Placement(x=-100, y=10, width=16, height=16),
        Placement(x=FRAME_W, y=10, width=16, height=16),
        Placement(x=10, y=-100, width=16, height=16),
        Placement(x=10, y=FRAME_H, width=16, height=16),
        Placement(x=10, y=10, width=0, height=16),
        Placement(x=10, y=10, width=16, height=0),
        Placement(x=10, y=10, width=-5, height=-5),
    ],
)
def test_fully_off_frame_or_empty_placements_draw_nothing_and_do_not_crash(placement):
    frame = _frame(10)

    Overlay().draw(frame, _meme(), placement)

    assert (frame == 10).all()


def test_a_tiny_placement_does_not_crash():
    Overlay().draw(_frame(), _meme(), Placement(x=10, y=10, width=1, height=1))


def test_nearby_sizes_share_one_resized_copy(monkeypatch):
    calls = []
    real_resize = cv2.resize
    monkeypatch.setattr(cv2, "resize", lambda *a, **k: calls.append(1) or real_resize(*a, **k))
    overlay, meme = Overlay(), _meme()

    for width in (40, 41, 42, 39):
        overlay.draw(_frame(), meme, Placement(x=10, y=5, width=width, height=width))

    assert len(calls) == 1


def test_a_different_meme_is_not_served_from_another_memes_cache():
    overlay, frame = Overlay(), _frame()
    overlay.draw(frame, _meme(bgr=(255, 0, 0)), Placement(10, 10, 16, 16))

    overlay.draw(frame, _meme(bgr=(0, 255, 0)), Placement(10, 10, 16, 16))

    assert tuple(frame[18, 18]) == (0, 255, 0)


def test_edges_of_a_cut_out_meme_do_not_pick_up_a_dark_fringe():
    meme = np.zeros((16, 16, 4), dtype=np.uint8)  # transparent black everywhere...
    meme[4:12, 4:12] = (255, 255, 255, 255)  # ...except an opaque white square
    frame = _frame(255)

    Overlay().draw(frame, meme, Placement(x=10, y=10, width=32, height=32))

    assert frame.min() >= 250  # white on white must stay white
