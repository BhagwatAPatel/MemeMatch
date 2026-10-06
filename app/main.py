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
from app.features import extract_features, top_features
from app.prediction import Prediction
from app.rules import label_by_rules
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
    rule_label: str | None,
):
    '''Draw FPS, the raw prediction, the confirmed expression and the rule baseline.'''
    raw = f"{prediction.label} {prediction.confidence:.2f}" if prediction else "-"
    lines = [
        f"FPS: {fps:4.1f}    Face: {'yes' if prediction else 'no'}",
        f"Raw: {raw}    Confirmed: {confirmed or '-'}",
        f"Rules: {rule_label or '-'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (10, 25 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

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

    try:
        with Camera(args.camera) as cam, FaceTracker() as tracker:
            print(f"Camera {args.camera} opened. Press q to quit, 1 to toggle landmarks.")
            while True:
                frame_started = time.perf_counter()
                frame = cam.read()

                # Mirror so the preview behaves like a mirror (move right,
                # you move right). Video-call apps do the same.
                # Must happen BEFORE tracking so landmarks line up with 
                # what is drawn on screen
                frame = cv2.flip(frame, 1)

                face = tracker.process(frame)
                prediction = rule_label = None
                if face is not None:
                    features = extract_features(face)
                    prediction = classifier.predict(features)
                    rule_label = label_by_rules(features)
                confirmed = smoother.update(prediction, time.monotonic())
                if face is not None:
                    if show_landmarks:
                        draw_face_debug(frame, face)

                    for i, (name, score) in enumerate(top_features(face)):
                        cv2.putText(frame, f"{name}: {score:.2f}", (10, 90 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                draw_hud(frame, fps_counter.tick(), prediction, confirmed, rule_label)
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

