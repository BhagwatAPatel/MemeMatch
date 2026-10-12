'''The meme library: loads memes/<label>/ and chooses a meme for a label.

The only code that knows the library's layout on disk. Images are decoded
once at construction; picking is a dictionary lookup. A category may be
empty, and a label with no meme simply yields none: nothing is shown and
nothing falls back to another label.

Pipeline position: Smoother (confirmed expression) -> [MemePicker] -> overlay
'''

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path("memes")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}


@dataclass(frozen=True)
class Meme:
    '''One loaded meme: its file and its BGRA pixels.'''

    path: Path
    image: np.ndarray


class MemePicker:
    '''Loads a meme library and picks a meme per label, never repeating itself.'''

    def __init__(self, root: Path = DEFAULT_ROOT, rng: random.Random | None = None):
        self._root = Path(root)
        self._rng = rng or random.Random()
        self._memes: dict[str, list[Meme]] = {}
        self._skipped: list[Path] = []
        self._last: dict[str, Path] = {}
        self._library_found = self._root.is_dir()
        if self._library_found:
            for category in sorted(p for p in self._root.iterdir() if p.is_dir()):
                self._memes[category.name] = self._load_category(category)

    def pick(self, label: str) -> Meme | None:
        '''A random meme for the label other than the last one picked, or None.'''
        candidates = self._memes.get(label, [])
        if len(candidates) > 1:
            candidates = [m for m in candidates if m.path != self._last.get(label)]
        if not candidates:
            return None
        meme = self._rng.choice(candidates)
        self._last[label] = meme.path
        return meme

    def startup_warning(self) -> str | None:
        '''One message naming empty categories and unreadable files, or None.'''
        if not self._library_found:
            return f"Meme library folder not found: {self._root}"
        parts = []
        empty = [label for label, memes in self._memes.items() if not memes]
        if empty:
            parts.append(f"No memes for: {', '.join(empty)}")
        if self._skipped:
            parts.append(f"Skipped unreadable files: {', '.join(p.name for p in self._skipped)}")
        return "; ".join(parts) or None

    def _load_category(self, folder: Path) -> list[Meme]:
        memes = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            image = _load_bgra(path)
            if image is None:
                self._skipped.append(path)
            else:
                memes.append(Meme(path, image))
        return memes


def _load_bgra(path: Path) -> np.ndarray | None:
    '''Decode an image as uint8 BGRA (opaque when the file has no alpha), or None.'''
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image
