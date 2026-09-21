'''Turn the collected samples into tensors ready for training.

Responsibilities: turn label strings into integer class ids, and split
train/test so we measure accuracy on rows the model never saw. Reading
the CSV itself is training.samples' job.
'''

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from training.samples import DEFAULT_PATH, SampleStore

@dataclass
class Dataset:
    '''Train/test tensors plus the label vocabulary.

    X_*: float32 tensors of shape (rows, 52).
    y_*: int64 tensors of shape (rows,), values are indices into 'labels'.
    labels: sorted class names; index i <-> y value i.
    '''

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    labels: list[str]

def load_dataset(
    path: Path = DEFAULT_PATH, test_size: float = 0.2, seed: int = 42
) -> Dataset:
    '''Load the samples and split them into stratified train/test tensors.

    Stratified means each label appears in train and test in the same
    proportion as in the full data, so a rare class isn't lost from the
    test set. Raises ValueError if any label has fewer than 2 samples.
    '''
    X, label_per_row = SampleStore(path).load()

    labels = sorted(set(label_per_row))
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = [label_to_id[name] for name in label_per_row]

    too_few = [name for name in labels if label_per_row.count(name) < 2]
    if too_few:
        raise ValueError(f"Need at least 2 samples per label; too few for: {too_few}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(
        X_train=torch.from_numpy(X_train),
        y_train=torch.tensor(y_train, dtype=torch.long),
        X_test=torch.from_numpy(X_test),
        y_test=torch.tensor(y_test, dtype=torch.long),
        labels=labels,
    )
