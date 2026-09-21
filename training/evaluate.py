'''Score an expression model on the test session it was trained to hold out.

Pipeline position: SampleStore -> dataset -> train -> [evaluate]
'''

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from training.dataset import DEFAULT_PATH, Dataset, load_dataset
from training.model import DEFAULT_MODEL_PATH, load_model, load_split_sessions

MIN_ACCURACY = 0.85
MIN_CLASS_RECALL = 0.70


@dataclass
class Report:
    '''Test-split results. confusion[i][j] counts samples whose true label
    is labels[i] and predicted label is labels[j]. precision is None for a
    label the model never predicted. confidence_* are mean
    softmax top-probabilities; confidence_wrong is None with no mistakes.'''

    labels: list[str]
    accuracy: float
    recall: dict[str, float]
    precision: dict[str, float | None]
    confusion: np.ndarray
    confidence_correct: float | None
    confidence_wrong: float | None


def evaluate_model(model, data: Dataset) -> Report:
    '''Run the model over data.X_test and summarise how it did.'''
    model.eval()
    with torch.no_grad():
        top_probability, predicted = model(data.X_test).softmax(dim=1).max(dim=1)
    predicted, top_probability = predicted.numpy(), top_probability.numpy()
    true = data.y_test.numpy()
    right = predicted == true

    n = len(data.labels)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(true, predicted):
        confusion[t, p] += 1

    return Report(
        labels=data.labels,
        accuracy=float(right.mean()),
        recall={
            label: float(confusion[i, i] / confusion[i].sum())
            for i, label in enumerate(data.labels)
        },
        precision={
            label: float(confusion[i, i] / confusion[:, i].sum()) if confusion[:, i].sum() else None
            for i, label in enumerate(data.labels)
        },
        confusion=confusion,
        confidence_correct=_mean_or_none(top_probability[right]),
        confidence_wrong=_mean_or_none(top_probability[~right]),
    )


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def failed_bars(report: Report) -> list[str]:
    '''One message per acceptance bar the report misses (empty when it passes).'''
    failures = []
    if report.accuracy < MIN_ACCURACY:
        failures.append(f"accuracy {report.accuracy:.1%} is below {MIN_ACCURACY:.0%}")
    for label, recall in report.recall.items():
        if recall < MIN_CLASS_RECALL:
            failures.append(f"recall for {label!r} is {recall:.1%}, below {MIN_CLASS_RECALL:.0%}")
    return failures


def print_report(report: Report) -> None:
    width = max(len(label) for label in report.labels)
    print(f"Test accuracy: {report.accuracy:.1%}")
    print("Recall per label: " + ", ".join(f"{k} {v:.1%}" for k, v in report.recall.items()))
    print("Precision per label: " + ", ".join(
        f"{k} " + ("n/a" if v is None else f"{v:.1%}") for k, v in report.precision.items()))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(" " * (width + 1) + " ".join(f"{label[:7]:>7}" for label in report.labels))
    for label, row in zip(report.labels, report.confusion):
        print(f"{label:>{width}} " + " ".join(f"{n:>7}" for n in row))
    for name, value in (("correct", report.confidence_correct), ("wrong", report.confidence_wrong)):
        print(f"Mean confidence when {name}: " + ("n/a" if value is None else f"{value:.2f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the expression model on the test session")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH, help="sample CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="model file")
    args = parser.parse_args()

    try:
        model, labels = load_model(args.model)
        test_session, val_session = load_split_sessions(args.model)
        data = load_dataset(args.data, test_session, val_session)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1
    if labels != data.labels:
        print(f"Error: model labels {labels} differ from the data's labels {data.labels}. Retrain.")
        return 1

    report = evaluate_model(model, data)
    print_report(report)

    baseline = LogisticRegression(max_iter=1000).fit(data.X_train.numpy(), data.y_train.numpy())
    print(f"Baseline (logistic regression) test accuracy: "
          f"{baseline.score(data.X_test.numpy(), data.y_test.numpy()):.1%}")

    failures = failed_bars(report)
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
