'''The expression classifier network, plus save/load helpers

Imported by BOTH training (train.py) and the live app (Step 7), so the
architecture is defined in exactly one place.

Architecture (from SPEC.md section 7)
    Linear(52, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> ReLU -> Linear(32, N)

'''
from pathlib import Path

import torch
from torch import nn

from app.features import FEATURE_NAMES, NUM_FEATURES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "expression_model.pt"

class ExpressionMLP(nn.Module):
    def __init__(self, num_classes: int, num_features: int = NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            # Dropout randomly zeroes 20% of activations during training only.
            # It stops the net memorising your exact samples, which matters
            # when the dataset is small and all from one face.
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw scores ('logits"), NOT probabilities. CrossEntropyLoss
        # applies softmax itself, and applying it twice hurts training.
        return self.net(x)

def save_model(
    model: ExpressionMLP,
    labels: list[str],
    path: Path = DEFAULT_MODEL_PATH,
    *,
    test_session: str,
    val_session: str,
) -> None:
    '''Save weights AND the metadata needed to use them correctly later,
    including which sessions were held out, so evaluate scores the same split.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "feature_names": FEATURE_NAMES,
            "test_session": test_session,
            "val_session": val_session,
        },
        path,
    )

def load_model(path: Path = DEFAULT_MODEL_PATH) -> tuple[ExpressionMLP, list[str]]:
    '''Load a saved model, verify its feature contract, return (model, labels).

    The returned model is in eval mode (dropout off), ready for inference.
    '''

    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python -m training.train"
        )
    # weights_only=False because we stored plain Python lists alongside the
    # tensors. Fine for a file we produced ourselves.

    bundle = torch.load(path, map_location="cpu", weights_only=False)

    if bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError(
            "Model was trained on a different feature layout than app.features "
            "currently defines. Retrain with: python -m training.train"
        )

    labels = bundle["labels"]
    model = ExpressionMLP(num_classes=len(labels))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, labels


def load_split_sessions(path: Path = DEFAULT_MODEL_PATH) -> tuple[str, str]:
    '''Return (test_session, val_session): the sessions held out in training.'''
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return bundle["test_session"], bundle["val_session"]
    except KeyError:
        raise ValueError(
            "Model file doesn't record its held-out sessions (saved by an older "
            "version). Retrain with: python -m training.train"
        ) from None
