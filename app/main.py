'''
MemeMatch entry point.

Run with: 
    python -m app.main (defualt camera)
    python -m app.main --camera 1 (a different camera)

    Press q in the video window to quit.
'''

import argparse
import sys 

import cv2

from app.camera import Camera, CameraError, FPSCounter

WINDOW_NAME = "MemeMatch"

def parse_args():
    parser = argparse.ArgumentParser(description="MemeMatch live preview")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="camera index (0 = build-in camera on most Macs)",
    )
    return parser.parse_args()

def draw_hud(frame, fps: float):
    '''Draw the heads-up display (for now: just FPS) onto the frame.'''
    text = f"FPS: {fps:4.1f}"
    cv2.putText(
        frame, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

def main() -> int:
    args = parse_args()
    fps_counter = FPSCounter()

    try:
        with Camera(args.camera) as cam:
            print(f"Camera {args.camera} opened. Press q to quit.")
            while True:
                frame = cam.read()

                # Mirror so the preview behaves like a mirror (move right,
                # you move right). Video-call apps do the same.
                frame = cv2.flip(frame, 1)

                draw_hud(frame, fps_counter.tick())
                cv2.imshow(WINDOW_NAME, frame)

                # waitKey(1) waits 1ms for a key AND lets the window redraw.
                # Without it, imshow never actually paints anything.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except CameraError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

