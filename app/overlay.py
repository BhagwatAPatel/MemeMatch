'''Composites a meme onto a camera frame at a placement.

The only code that touches pixels for the meme. Anything the placement puts
outside the frame is clipped, never clamped or hidden. Resized memes are
cached per size bucket (nearby sizes share one copy), so the per-frame cost
is a slice and a blend rather than a resize.

Pipeline position: PlacementTracker -> [Overlay] -> display
'''

import cv2
import numpy as np

from app.placement import Placement

_BUCKET = 8  # px: memes are resized to multiples of this width
_MAX_CACHED = 64


class Overlay:
    def __init__(self) -> None:
        # (id(meme), bucketed width) -> (meme, premultiplied float32 BGR, alpha 0-1).
        # Holding `meme` keeps its id from being reused by another array.
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def draw(self, frame: np.ndarray, meme: np.ndarray, placement: Placement) -> None:
        '''Alpha-blend a BGRA meme onto a BGR frame, in place.

        The meme is scaled to the placement's width (rounded to a size bucket,
        aspect preserved) and anchored at the placement's bottom centre.
        '''
        if placement.width < 1 or placement.height < 1:
            return
        colour, alpha = self._resized(meme, placement.width)
        h, w = alpha.shape
        left = placement.x + placement.width // 2 - w // 2
        top = placement.bottom - h

        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + w, frame.shape[1]), min(top + h, frame.shape[0])
        if x0 >= x1 or y0 >= y1:
            return

        roi = frame[y0:y1, x0:x1]
        fg = colour[y0 - top:y1 - top, x0 - left:x1 - left]
        a = alpha[y0 - top:y1 - top, x0 - left:x1 - left, None]
        roi[:] = np.clip(fg + roi * (1.0 - a), 0, 255).astype(np.uint8)

    def _resized(self, meme: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_w = max(1, round(width / _BUCKET) * _BUCKET)
        key = (id(meme), bucket_w)
        if key not in self._cache:
            if len(self._cache) >= _MAX_CACHED:
                self._cache.clear()
            self._cache[key] = (meme, *_premultiplied_resize(meme, bucket_w))
        _, colour, alpha = self._cache[key]
        return colour, alpha


def _premultiplied_resize(meme: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    '''Resize to `width` keeping aspect, interpolating colour premultiplied by alpha.

    Interpolating straight BGRA blends the colour of fully transparent pixels
    into the edges, which shows up as a dark fringe around cut-out memes.
    '''
    src_h, src_w = meme.shape[:2]
    height = max(1, round(width * src_h / src_w))
    alpha = meme[:, :, 3:4].astype(np.float32) / 255.0
    premultiplied = np.concatenate([meme[:, :, :3].astype(np.float32) * alpha, alpha], axis=2)
    interpolation = cv2.INTER_AREA if width < src_w else cv2.INTER_LINEAR
    resized = cv2.resize(premultiplied, (width, height), interpolation=interpolation)
    return resized[:, :, :3], resized[:, :, 3]
