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

from app.camera import Camera, CameraError, FPSCounter
from app.face_tracker import FaceTracker, draw_face_debug 
from app.expression_classifier import ExpressionClassifier, Prediction
from app.features import extract_features, top_features
from app.smoothing import Smoother


WINDOW_NAME = "MemeMatch"

def parse_args():
    parser = argparse.ArgumentParser(description="MemeMatch live preview")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="camera index (0 = build-in camera on most Macs)",
    )
    return parser.parse_args()

def draw_hud(frame, fps: float, face_found: bool = False):
    '''Draw the heads-up display (FPS and face status) onto the frame.'''
    text = f"FPS: {fps:4.1f}    Face: {'yes' if face_found else 'no'}"
    cv2.putText(
        frame, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

def draw_expression_hud(frame, prediction: Prediction | None, confirmed: str | None):
    '''Draw the raw per-frame prediction and the confirmed expression.'''
    raw = f"{prediction.label} {prediction.confidence:.2f}" if prediction else "-"
    cv2.putText(
        frame, f"Raw: {raw}    Confirmed: {confirmed or '-'}", (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

def main() -> int:
    args = parse_args()
    fps_counter = FPSCounter()
    show_landmarks = True

    try:
        classifier = ExpressionClassifier()
        smoother = Smoother()
        with Camera(args.camera) as cam, FaceTracker() as tracker:
            print(f"Camera {args.camera} opened. Press q to quit, 1 to toggle landmarks.")
            while True:
                frame = cam.read()

                # Mirror so the preview behaves like a mirror (move right,
                # you move right). Video-call apps do the same.
                # Must happen BEFORE tracking so landmarks line up with 
                # what is drawn on screen
                frame = cv2.flip(frame, 1)

                face = tracker.process(frame)
                prediction = classifier.predict(extract_features(face)) if face is not None else None
                confirmed = smoother.update(prediction, time.monotonic())
                if face is not None:
                    if show_landmarks:
                        draw_face_debug(frame, face)

                    for i, (name, score) in enumerate(top_features(face)):
                        cv2.putText(frame, f"{name}: {score:.2f}", (10, 90 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                draw_hud(frame, fps_counter.tick(), face is not None)
                draw_expression_hud(frame, prediction, confirmed)
                cv2.imshow(WINDOW_NAME, frame)

                # waitKey(1) waits 1ms for a key AND lets the window redraw.
                # Without it, imshow never actually paints anything.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("1"):
                    show_landmarks = not show_landmarks

    except (CameraError, FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

