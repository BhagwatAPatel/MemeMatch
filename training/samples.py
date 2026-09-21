'''The labelled-sample CSV: the one place that knows its format.

File layout: a header row ["label", "session", *FEATURE_NAMES], then one
row per sample: the label string, the session id, then the 52 feature values. Collection
appends to it, training loads from it, and nothing else touches the file,
so the writer and reader can never drift apart.

Pipeline position: features -> collect -> [SampleStore] -> dataset -> train
'''

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from app.features import FEATURE_NAMES, NUM_FEATURES

DEFAULT_PATH = Path("data") / "samples.csv"
HEADER = ["label", "session", *FEATURE_NAMES]

_PRE_SESSION_HEADER = ["label", *FEATURE_NAMES]

_PRE_SESSION_ERROR = (
    "{path} has no session column (it was collected before sessions "
    "existed). Clear it and re-record: python -m training.collect --reset"
)

_LAYOUT_ERROR = (
    "{path} was collected with a different feature layout than "
    "app.features defines. Re-collect, or restore the matching version."
)


class SampleStore:
    '''Labelled feature rows persisted in a CSV file.'''

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)

    def append(self, label: str, features: np.ndarray, session: str) -> None:
        '''Add one sample, creating the file and header if needed.

        Each call opens, writes and closes the file, so a crash or Ctrl-C
        never loses more than the row in flight.
        '''
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (NUM_FEATURES,):
            raise ValueError(
                f"Expected {NUM_FEATURES} features, got shape {values.shape}"
            )

        is_new = not self.path.exists() or self.path.stat().st_size == 0
        if is_new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Appending new-layout rows under an old header would silently
            # corrupt the dataset, so check before writing.
            self._check_header()

        # newline="" is required by the csv module, otherwise you can get
        # blank lines between rows on some platforms.
        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(HEADER)
            writer.writerow([label, session, *values.tolist()])

    def counts(self) -> Counter:
        '''Return {label: n} for the samples on disk; empty if there is no file.

        Raises ValueError if the file's layout doesn't match, like load().
        '''
        if not self.path.exists():
            return Counter()
        return Counter(label for label, _, _ in self._read_samples())

    def load(self) -> tuple[np.ndarray, list[str], list[str]]:
        '''Return (X, labels, sessions): X is float32 (rows, 52); labels and
        sessions each hold one entry per sample, in file order.

        Raises FileNotFoundError if nothing has been collected, and
        ValueError if the header or any row doesn't match the layout.
        '''
        if not self.path.exists():
            raise FileNotFoundError(
                f"No dataset at {self.path}. Run: python -m training.collect"
            )

        samples = list(self._read_samples())
        labels = [label for label, _, _ in samples]
        sessions = [session for _, session, _ in samples]
        X = np.array([values for _, _, values in samples], dtype=np.float32)
        return X.reshape(len(samples), NUM_FEATURES), labels, sessions

    def clear(self) -> None:
        '''Delete all samples. Safe to call when there are none.'''
        self.path.unlink(missing_ok=True)

    def _check_header(self) -> None:
        with self.path.open(newline="") as f:
            self._verify_header(next(csv.reader(f), None))

    def _verify_header(self, header: list[str] | None) -> None:
        if header == _PRE_SESSION_HEADER:
            raise ValueError(_PRE_SESSION_ERROR.format(path=self.path))
        if header != HEADER:
            raise ValueError(_LAYOUT_ERROR.format(path=self.path))

    def _read_samples(self):
        '''Yield (label, session, values) per sample, validating as it goes.

        The one reader behind counts() and load(): it owns the header
        check, blank-line skipping and per-row validation. An empty
        (0-byte) file yields nothing.
        '''
        with self.path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return
            self._verify_header(header)
            for row in reader:
                if not row:  # tolerate a stray blank line
                    continue
                if len(row) != len(HEADER):
                    raise ValueError(
                        f"Bad row at line {reader.line_num} of {self.path}: "
                        f"expected {len(HEADER)} columns, got {len(row)}"
                    )
                try:
                    values = [float(v) for v in row[2:]]
                except ValueError as err:
                    raise ValueError(
                        f"Bad row at line {reader.line_num} of {self.path}: {err}"
                    ) from err
                yield row[0], row[1], values
