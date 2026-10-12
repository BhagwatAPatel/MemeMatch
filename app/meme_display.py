'''Shows the confirmed expression's meme on the frame.

Glue between the meme picker, placement and overlay. A meme is shown exactly
when an expression is confirmed: the smoother owns the confidence threshold,
so there is no second gate here. The meme is chosen once when a label becomes
confirmed and kept until the confirmed expression changes or clears.

Pipeline position: Smoother + FaceTracker -> [MemeDisplay] -> display
'''

import numpy as np

from app.meme_picker import Meme, MemePicker
from app.overlay import Overlay
from app.placement import FaceBox, PlacementTracker

_NO_MEME_ASPECT = 1.0  # keeps the placement smoothing warm while nothing is shown


class MemeDisplay:
    def __init__(
        self,
        picker: MemePicker,
        tracker: PlacementTracker | None = None,
        overlay: Overlay | None = None,
    ):
        self._picker = picker
        self._tracker = tracker or PlacementTracker()
        self._overlay = overlay or Overlay()
        self._label: str | None = None
        self._meme: Meme | None = None

    def draw(self, frame: np.ndarray, confirmed: str | None, face_box: FaceBox | None) -> None:
        '''Draw the meme for the confirmed expression onto the frame, in place.

        confirmed is the smoother's output (None = nothing confirmed or no
        face); face_box is None when no face was found.
        '''
        if confirmed != self._label:
            self._label = confirmed
            self._meme = self._picker.pick(confirmed) if confirmed else None

        aspect = _NO_MEME_ASPECT
        if self._meme is not None:
            height, width = self._meme.image.shape[:2]
            aspect = height / width
        placement = self._tracker.update(face_box, aspect)

        if self._meme is not None and placement is not None:
            self._overlay.draw(frame, self._meme.image, placement)
