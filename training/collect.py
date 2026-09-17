'''Record labelled expression samples from the webcam to a CSV file.

Run: python -m training.collect
Keys: 1-5 picks a label, r starts/stops recording, q quits.

Each row is: label, then the 52 blendshape scores from app.features. 
No images are stored - only the numbers - so the dataset is tiny and
private. The CSV is appended to, so you can record in several sessions
(different lighting, distance, angle) and it all goes in one file.

Pipeline position: Camera -> FaceTracker -> features -> [collect] -> CSV
'''

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2 
import numpy as np

from app.camera import Camera, CameraError
from app.face_tracker import FaceTracker, draw_face_debug
from app.features import FEATURE_NAMES, extract_features

# MVP classes. Order here is only for the key mapping; the model learns
# labels by name from the CSV, so the order never matters downstream
LABELS = ["neutral", "happy", "surprised", "angry", "sad"]

# Keys 1..5 map onto LABELS by index.
KEY_TO_LABEL = {ord(str(i + 1)): label for i, label in enumerate(LABELS)}

DEFAULT_OUT = Path("data") / "samples.csv"
WINDOW_NAME = "MemeMatch - collect"

class SampleWriter:
    '''Appends labelled feature rows to a CSV, creating the header if needed.'''

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Header goes in only when the file is new or empty. Otherwise we'd 
        # write a second header mid-file and corrupt the dataset.
        needs_header = not path.exists() or path.stat().st_size == 0

        # newline"" is required by csv module on all platforms, otherwise
        # you can get blank lines between rows.
        self._file = path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if needs_header:
            self._writer.writerow(["label", *FEATURE_NAMES])
            self._file.flush()

    def write(self, label: str, features: np.ndarray) -> None:
        self._writer.writerow([label, *features.tolist()])
        # Flush every row so a crash (or Ctrl-C) never loses more than one.
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    # Context-manager support so callers can use 'with SampleWriter(...) as w:`.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def count_existing(path: Path) -> Counter:
    '''Return {label: n} for rows already in the CSV, so the HUD shows totals.'''
    counts: Counter = Counter()

    if not path.exists():
        return counts
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None) # skip header
        for row in reader:
            if row: # tolerate a stray blank line
                counts[row[0]] += 1
    return counts

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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV to append to")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="samples per second while recording")
    parser.add_argument("--reset", action="store_true",
                        help="delete the existing CSV before recording")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.reset and args.out.exists():
        existing = sum(count_existing(args.out).values())
        answer = input(f"Delete {args.out} ({existing} rows)? [y/N]")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return 0
        args.out.unlink() # Path's delete-a-file method
        print("Deleted.")

    counts = count_existing(args.out)
    label = LABELS[0]
    recording = False
    interval = 1.0 / args.rate # seconds between saved rows
    last_saved = 0.0

    try:
        with Camera(args.camera) as cam, FaceTracker() as tracker, SampleWriter(args.out) as writer:
            print(f"Appending to {args.out}. Existing rows: {dict(counts)}")
            while True:
                frame = cam.read()
                frame = cv2.flip(frame, 1) # same mirror as the app, so it feels natural
                face = tracker.process(frame)

                if face is not None:
                    draw_face_debug(frame, face)
                    now = time.monotonic()
                    # Rate-limit so 30 fps doesn't dump 30 near-identical rows/s.
                    if recording and now - last_saved >= interval:
                        writer.write(label, extract_features(face))
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

    except (CameraError, FileNotFoundError) as err:
        print(f"Error: {err}")
        return 1
    finally:
        cv2.destroyAllWindows()

    print(f"Done. Rows per label: {dict(counts)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

