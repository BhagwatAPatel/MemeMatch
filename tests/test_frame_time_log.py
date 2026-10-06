'''Tests for app.camera.FrameTimeLog. Timestamps are passed in, no clock.'''

from app.camera import FrameTimeLog


def test_reports_the_average_frame_time_once_per_period_then_starts_over():
    log = FrameTimeLog(period=30.0)

    assert log.record(0.020, now=0.0) is None
    assert log.record(0.040, now=15.0) is None
    report = log.record(0.030, now=30.0)

    assert report is not None and "30.0 ms" in report  # (20 + 40 + 30) / 3
    assert log.record(0.100, now=31.0) is None  # a fresh period has begun
