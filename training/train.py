'''Train the expression model on a Dataset.

Adam, cross-entropy, batch 64, fixed seed. Pipeline position:
SampleStore -> dataset -> [train] -> expression model file -> evaluate
'''

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from training.dataset import DEFAULT_PATH, Dataset, load_dataset
from training.model import DEFAULT_MODEL_PATH, ExpressionMLP, save_model

BATCH_SIZE = 64
LEARNING_RATE = 1e-3


@dataclass
class EpochStats:
    '''Validation numbers after one epoch (epochs count from 1).'''

    epoch: int
    val_loss: float
    val_accuracy: float


@dataclass
class TrainResult:
    '''model holds the weights of epoch `best`: highest validation accuracy,
    ties broken by lower validation loss. history has every epoch.'''

    model: ExpressionMLP
    history: list[EpochStats]
    best: EpochStats


def train(data: Dataset, epochs: int = 50, seed: int = 42) -> TrainResult:
    '''Train on data.X_train, choosing the best epoch on data.X_val.

    The model comes back in eval mode, ready for inference.
    '''
    torch.manual_seed(seed)
    model = ExpressionMLP(num_classes=len(data.labels))
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best, best_weights = None, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(data.X_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimiser.zero_grad()
            loss_fn(model(data.X_train[batch]), data.y_train[batch]).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.X_val)
            stats = EpochStats(
                epoch=epoch,
                val_loss=loss_fn(logits, data.y_val).item(),
                val_accuracy=(logits.argmax(dim=1) == data.y_val).float().mean().item(),
            )
        history.append(stats)
        if best is None or (stats.val_accuracy, -stats.val_loss) > (best.val_accuracy, -best.val_loss):
            best, best_weights = stats, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return TrainResult(model, history, best)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the expression model")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH, help="sample CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH, help="model file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test-session", help="session to hold out as test")
    parser.add_argument("--val-session", help="session to use as validation")
    args = parser.parse_args()

    try:
        data = load_dataset(args.data, args.test_session, args.val_session)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    result = train(data, epochs=args.epochs)
    save_model(result.model, data.labels, args.out,
               test_session=data.test_session, val_session=data.val_session)
    best = result.best
    print(f"Best epoch {best.epoch}: val accuracy {best.val_accuracy:.1%}, "
          f"val loss {best.val_loss:.3f}. Held out: test={data.test_session}, "
          f"val={data.val_session}. Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
