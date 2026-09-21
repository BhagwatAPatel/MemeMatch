'''Turn the collected samples into tensors ready for training.

Responsibilities: turn label strings into integer class ids, and divide
whole sessions between train, validation and test so every held-out score
is measured on a recording the model never saw. Reading the CSV itself is
training.samples' job.
'''

from dataclasses import dataclass
from pathlib import Path

from collections import Counter

import numpy as np
import torch

from training.samples import DEFAULT_PATH, SampleStore

MIN_SESSIONS = 3

@dataclass
class Dataset:
    '''Train/validation/test tensors plus the label vocabulary.

    X_*: float32 tensors of shape (rows, 52).
    y_*: int64 tensors of shape (rows,), values are indices into 'labels'.
    labels: sorted class names; index i <-> y value i.
    test_session, val_session: the sessions those splits were taken from.
    '''

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    labels: list[str]
    test_session: str
    val_session: str

def load_dataset(
    path: Path = DEFAULT_PATH,
    test_session: str | None = None,
    val_session: str | None = None,
) -> Dataset:
    '''Load the samples and divide whole sessions between the three splits.

    By default the last session in file order is test, the one before it is
    validation, and every earlier session is train. Name test_session and/or
    val_session to hold out specific sessions instead; a val_session left
    unnamed is the newest session that isn't the test session.
    '''
    X, label_per_row, session_per_row = SampleStore(path).load()

    labels = sorted(set(label_per_row))
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.array([label_to_id[name] for name in label_per_row], dtype=np.int64)

    order = list(dict.fromkeys(session_per_row))  # file order, no repeats
    if len(order) < MIN_SESSIONS:
        raise ValueError(
            f"Need at least {MIN_SESSIONS} sessions (train, validation, test); "
            f"found {len(order)}. Record more with: python -m training.collect"
        )
    _require_every_label_in_every_session(labels, order, label_per_row, session_per_row)
    for name in (test_session, val_session):
        if name is not None and name not in order:
            raise ValueError(f"{name!r} is not a recorded session; have: {order}")
    if test_session is not None and test_session == val_session:
        raise ValueError("test_session and val_session must be different sessions")
    test_session = test_session or order[-1]
    val_session = val_session or [s for s in order if s != test_session][-1]
    sessions = np.array(session_per_row)
    test = sessions == test_session
    val = sessions == val_session
    train = ~(test | val)

    def split(mask):
        return torch.from_numpy(X[mask]), torch.from_numpy(y[mask])

    X_train, y_train = split(train)
    X_val, y_val = split(val)
    X_test, y_test = split(test)
    return Dataset(
        X_train, y_train, X_val, y_val, X_test, y_test, labels, test_session, val_session
    )


def _require_every_label_in_every_session(labels, order, label_per_row, session_per_row):
    '''Raise, with a count table, if any session lacks any label.

    A label absent from the test session has no recall to measure; absent
    from train it can't be learned.
    '''
    counts = {session: Counter() for session in order}
    for label, session in zip(label_per_row, session_per_row):
        counts[session][label] += 1

    if all(counts[s][label] for s in order for label in labels):
        return
    table = "\n".join(
        f"  {session}: " + ", ".join(f"{label}={counts[session][label]}" for label in labels)
        for session in order
    )
    raise ValueError(f"Every session must contain every label. Samples per session:\n{table}")
