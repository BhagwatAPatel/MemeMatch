'''
MemeMatch entry point.

Run with: 
    python -m app.main (defualt camera)
    python -m app.main --camera 1 (a different camera)

    Press q in the video window to quit.
'''

import argparse
import sys
import time

import cv2

from app.camera import Camera, CameraError, FPSCounter, FrameTimeLog
from app.face_tracker import FaceTracker, draw_face_debug 
from app.expression_classifier import ExpressionClassifier
from app.features import top_features
from app.meme_display import MemeDisplay
from app.meme_picker import MemePicker
from app.pipeline import FramePipeline
from app.prediction import Prediction
from app.smoothing import Smoother


WINDOW_NAME = "MemeMatch"

def parse_args():
    parser = argparse.ArgumentParser(description="MemeMatch live preview")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="camera index (0 = build-in camera on most Macs)",
    )
    return parser.parse_args()

def draw_hud(
    frame, fps: float, prediction: Prediction | None, confirmed: str | None,
    rule_label: str | None, features: list[tuple[str, float]],
):
    '''Draw FPS, the raw prediction, the confirmed expression, the rule
    baseline and the strongest blendshapes, stacked so no two lines overlap.'''
    raw = f"{prediction.label} {prediction.confidence:.2f}" if prediction else "-"
    lines = [
        f"FPS: {fps:4.1f}    Face: {'yes' if prediction else 'no'}",
        f"Raw: {raw}    Confirmed: {confirmed or '-'}",
        f"Rules: {rule_label or '-'}",
    ]
    y = 25
    for text in lines:
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 30
    for name, score in features:
        cv2.putText(
            frame, f"{name}: {score:.2f}", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y += 25

def main() -> int:
    args = parse_args()
    fps_counter = FPSCounter()
    frame_times = FrameTimeLog()
    show_landmarks = True

    try:
        classifier = ExpressionClassifier()
    except (FileNotFoundError, ValueError) as err:  # no model, or a stale one
        print(f"Error: {err}", file=sys.stderr)
        return 1
    smoother = Smoother()
    picker = MemePicker()
    if warning := picker.startup_warning():
        print(f"Warning: {warning}", file=sys.stderr)
    meme_display = MemeDisplay(picker)

    try:
        with Camera(args.camera) as cam, FaceTracker() as tracker:
            pipeline = FramePipeline(tracker, classifier, smoother, meme_display)
            print(f"Camera {args.camera} opened. Press q to quit, 1 to toggle landmarks.")
            while True:
                frame_started = time.perf_counter()
                result = pipeline.process(cam.read(), time.monotonic())
                frame, face = result.frame, result.face

                if face is not None and show_landmarks:
                    draw_face_debug(frame, face)

                draw_hud(
                    frame, fps_counter.tick(), result.prediction, result.confirmed,
                    result.rule_label, top_features(face) if face is not None else [],
                )
                cv2.imshow(WINDOW_NAME, frame)

                # waitKey(1) waits 1ms for a key AND lets the window redraw.
                # Without it, imshow never actually paints anything.
                key = cv2.waitKey(1) & 0xFF
                report = frame_times.record(
                    time.perf_counter() - frame_started, time.monotonic()
                )
                if report:
                    print(report)
                if key == ord("q"):
                    break
                if key == ord("1"):
                    show_landmarks = not show_landmarks

    except (CameraError, FileNotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

