'''Record labelled expression samples from the webcam to a CSV file.

Run: python -m training.collect
Keys: 1-5 picks a label, r starts/stops recording, q quits.

Each recorded sample is appended through training.samples.SampleStore,
which owns the file format. No images are stored - only the numbers - so
the dataset is tiny and private. The file is appended to, so you can
record in several sessions (different lighting, distance, angle) and it
all goes in one place.

Pipeline position: Camera -> FaceTracker -> features -> [collect] -> SampleStore
'''

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2 

from app.camera import Camera, CameraError
from app.face_tracker import FaceTracker, draw_face_debug
from app.features import extract_features
from training.samples import DEFAULT_PATH, SampleStore

# MVP classes. Order here is only for the key mapping; the model learns
# labels by name from the CSV, so the order never matters downstream
LABELS = ["neutral", "happy", "surprised", "angry", "sad"]

# Keys 1..5 map onto LABELS by index.
KEY_TO_LABEL = {ord(str(i + 1)): label for i, label in enumerate(LABELS)}

WINDOW_NAME = "MemeMatch - collect"

def default_session_id(now: datetime | None = None) -> str:
    '''A sortable timestamp naming this recording run, e.g. 20260921-143005.'''
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")

def draw_hud(frame, label: str, recording: bool, counts: Counter, face_found: bool) -> None:
    '''Overlay the collection status on the frame.'''
    h = frame.shape[0]
    status = "REC" if recording else "paused"
    colour = (0, 0, 255) if recording else (200, 200, 200) # BGR: red when recording

    cv2.putText(frame, f"Label: {label} [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(frame, f"Face: {'yes' if face_found else 'NO'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # One line per class with its running count, bottom-left.
    for i, name in enumerate(LABELS):
        y = h - 20 - (len(LABELS) - 1 - i) * 22
        marker = ">" if name == label else " "
        cv2.putText(frame, f"{marker} {i + 1}:{name} {counts[name]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Record expression samples to CSV")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH, help="CSV to append to")
    parser.add_argument("--session", default=default_session_id(),
                        help="name for this recording run (default: a timestamp)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="samples per second while recording")
    parser.add_argument("--reset", action="store_true",
                        help="delete the existing CSV before recording")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    store = SampleStore(args.out)
    if args.reset and args.out.exists():
        try:
            described = f"{sum(store.counts().values())} samples"
        except ValueError:
            # A stale-layout file is exactly what --reset must be able to clear.
            described = "unreadable: different feature layout"
        answer = input(f"Delete {args.out} ({described})? [y/N]")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return 0
        store.clear()
        print("Deleted.")

    label = LABELS[0]
    recording = False
    interval = 1.0 / args.rate # seconds between saved samples
    last_saved = 0.0

    try:
        # Reading the counts also validates the file's header, so a stale
        # layout fails here, before the camera opens, not mid-recording.
        counts = store.counts()
        with Camera(args.camera) as cam, FaceTracker() as tracker:
            print(f"Appending to {args.out} as session {args.session}. Existing samples: {dict(counts)}")
            while True:
                frame = cam.read()
                frame = cv2.flip(frame, 1) # same mirror as the app, so it feels natural
                face = tracker.process(frame)

                if face is not None:
                    draw_face_debug(frame, face)
                    now = time.monotonic()
                    # Rate-limit so 30 fps doesn't dump 30 near-identical samples/s.
                    if recording and now - last_saved >= interval:
                        store.append(label, extract_features(face), args.session)
                        counts[label] += 1
                        last_saved = now

                draw_hud(frame, label, recording, counts, face is not None)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                if key in KEY_TO_LABEL:
                    label = KEY_TO_LABEL[key]
                    recording = False # never carry recording across a label change 

    except (CameraError, FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1
    finally:
        cv2.destroyAllWindows()

    print(f"Done. Samples per label: {dict(counts)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

