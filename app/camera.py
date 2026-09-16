# Camera capture for MemeMatch

'''
Responsibilites:
- open a webcam by index
- hand back one frame at a time as a NumPy array (H x W x 3, BGR)
- report a readable error instead of crashing when something is wrong 
- measure frames per second
'''

import time 

import cv2

class CameraError(Exception):
    '''Raised when the camera cannot be opened or stops delivering frames.'''

class Camera: 
    '''A thin wrapper around cv2.VideoCapture with error handling.'''

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index 

        # CAP_AVFOUNDATION is the native macOS camera backend, Being explicit avoids
        # OpenCV probing other backedns and printing warnings.
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened():
            raise CameraError( f"Count not open camera {index}. Check that it is connected and"
                              "not in use by another app, and that Terminal has Camera permissions"
                              "(System Settings > Privacy & security > Camera)"
                            )

        # These are requests not guarantees: the camera picks the closest
        # size it supports. 640 x 480 keeps every later stage fast.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        '''Return the next frame. Raises CameraError if the camera died.'''
        ok, frame = self.cap.read()

        if not ok or frame is None:
            raise CameraError(f"Camera {self.index} stopped returning frames.")
        return frame

    def release(self):
        '''Give the camera back to the operating system.'''
        self.cap.release()

    # The two methods below let us write 'with Camera() as cam:' so the 
    # camera is always released, even if an error happens mid-loop.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

class FPSCounter:
    # Exponentially smoothed frames-per-second estimate

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.perf_counter()

    def tick(self) -> float:
        '''Call once per frame. Returns the current smoothed FPS.'''
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            # Blend the new reading with the old value so the number 
            # doesn't jitter every frame.
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps
        