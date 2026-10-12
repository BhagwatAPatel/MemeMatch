'''Where and how large the meme sits for one frame.

Pure geometry, no pixels: a face box in, a rectangle out. The face box is
smoothed over time with an EMA; the meme never is, so swapping memes is a hard
cut dropped into the current placement. The rectangle may extend past the
frame edge; clipping is the overlay's job.

Pipeline position: FaceTracker (face box) -> [PlacementTracker] -> Overlay
'''

from dataclasses import dataclass

FaceBox = tuple[int, int, int, int]  # x, y, w, h in pixels


@dataclass(frozen=True)
class Placement:
    '''A meme rectangle in frame pixels. x and y may be negative.'''

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class PlacementTracker:
    '''Turns a stream of face boxes (or their absence) into meme placements.'''

    def __init__(self, face_scale: float = 1.2, gap: float = 0.1, alpha: float = 0.3):
        '''face_scale: meme width as a multiple of face width.
        gap: space between meme bottom and face top, as a fraction of face height.
        alpha: EMA weight of the newest face box (1.0 = no smoothing).
        '''
        self._face_scale = face_scale
        self._gap = gap
        self._alpha = alpha
        self._smoothed: tuple[float, float, float, float] | None = None

    def update(self, face_box: FaceBox | None, meme_aspect: float) -> Placement | None:
        '''Feed one frame's face box (None = no face); return the meme placement.

        meme_aspect is the meme's height / width. Losing the face clears the
        smoothing state, so a returning face snaps rather than glides.
        '''
        if face_box is None:
            self._smoothed = None
            return None

        if self._smoothed is None:
            self._smoothed = tuple(float(v) for v in face_box)
        else:
            a = self._alpha
            self._smoothed = tuple(a * new + (1 - a) * old for new, old in zip(face_box, self._smoothed))

        x, y, w, h = self._smoothed
        width = self._face_scale * w
        height = width * meme_aspect
        bottom = y - self._gap * h
        return Placement(
            x=round(x + w / 2 - width / 2),
            y=round(bottom - height),
            width=round(width),
            height=round(height),
        )
