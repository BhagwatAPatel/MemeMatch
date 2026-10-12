'''Tests for app.placement. Pure geometry: face boxes in, rectangles out.'''

from app.placement import Placement, PlacementTracker

SQUARE = 1.0  # meme height / width


def test_meme_is_scaled_to_face_width_and_centred_above_the_head():
    tracker = PlacementTracker(face_scale=1.2, gap=0.1)

    placement = tracker.update((100, 200, 100, 120), SQUARE)

    # width 1.2 * 100 = 120, centred on face centre x = 150
    assert (placement.x, placement.width, placement.height) == (90, 120, 120)
    # bottom = face top 200 - 0.1 * face height 120 = 188; y = 188 - 120
    assert placement.y == 68


def test_meme_height_follows_the_memes_aspect_ratio():
    tracker = PlacementTracker(face_scale=1.0, gap=0.0)

    placement = tracker.update((0, 300, 100, 100), 2.0)

    assert (placement.width, placement.height, placement.y) == (100, 200, 100)


def test_placement_may_extend_past_the_frame_edge():
    tracker = PlacementTracker()

    placement = tracker.update((0, 10, 100, 100), SQUARE)

    assert placement.x < 0 and placement.y < 0


def test_the_face_box_is_smoothed_with_an_ema():
    tracker = PlacementTracker(face_scale=1.0, gap=0.0, alpha=0.5)
    tracker.update((0, 200, 100, 100), SQUARE)

    placement = tracker.update((100, 200, 100, 100), SQUARE)

    # smoothed face x = 0.5 * 100 + 0.5 * 0 = 50
    assert placement.x == 50


def test_smoothing_converges_on_a_steady_face():
    tracker = PlacementTracker(alpha=0.3)
    tracker.update((0, 200, 100, 100), SQUARE)

    for _ in range(40):
        placement = tracker.update((300, 250, 140, 140), SQUARE)

    assert placement == PlacementTracker().update((300, 250, 140, 140), SQUARE)


def test_losing_the_face_returns_none_and_the_next_face_snaps():
    tracker = PlacementTracker(alpha=0.3)
    tracker.update((0, 200, 100, 100), SQUARE)

    assert tracker.update(None, SQUARE) is None
    placement = tracker.update((400, 300, 100, 100), SQUARE)

    assert placement == PlacementTracker().update((400, 300, 100, 100), SQUARE)


def test_swapping_memes_is_a_hard_cut_and_does_not_affect_the_smoothed_face():
    boxes = [(0, 400, 100, 100), (50, 400, 100, 100), (80, 380, 100, 100)]
    kept, swapped = PlacementTracker(face_scale=1.0, gap=0.0), PlacementTracker(face_scale=1.0, gap=0.0)
    for box in boxes[:-1]:
        kept.update(box, SQUARE)
        swapped.update(box, SQUARE)

    before = kept.update(boxes[-1], SQUARE)
    after = swapped.update(boxes[-1], 2.0)  # taller meme dropped in on the same frame

    assert after.x == before.x and after.width == before.width
    assert after.height == 2 * before.height
    assert after.bottom == before.bottom


def test_a_placement_reports_its_bottom_right_corner():
    assert Placement(x=10, y=20, width=30, height=40).right == 40
    assert Placement(x=10, y=20, width=30, height=40).bottom == 60
