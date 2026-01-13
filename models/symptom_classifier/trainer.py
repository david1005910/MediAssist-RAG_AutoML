"""Training utilities for symptom classifier."""

from typing import Dict, List, Tuple
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from .classifier import SymptomClassifier


def load_training_data(data_path: str) -> Tuple[List[Dict], List[str]]:
    """Load training data from JSON file.

    Args:
        data_path: Path to training data JSON file.

    Returns:
        Tuple of (samples, labels).
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    y = []

    for item in data:
        X.append({
            "symptoms": item["symptoms"],
            "patient_info": item.get("patient_info"),
        })
        y.append(item["disease"])

    return X, y


def train_and_evaluate(
    data_path: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Train and evaluate the symptom classifier.

    Args:
        data_path: Path to training data.
        output_path: Path to save trained model.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Evaluation metrics.
    """
    # Load data
    X, y = load_training_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    classifier = SymptomClassifier()
    train_metrics = classifier.train(X_train, y_train)

    # Evaluate on test set
    correct = 0
    for sample, true_label in zip(X_test, y_test):
        predictions = classifier.predict(
            sample["symptoms"],
            sample.get("patient_info"),
            top_k=1,
        )
        if predictions and predictions[0]["disease"] == true_label:
            correct += 1

    test_accuracy = correct / len(X_test)

    # Save model
    Path(output_path).mkdir(parents=True, exist_ok=True)
    classifier.save(output_path)

    return {
        "train_accuracy": train_metrics["accuracy_mean"],
        "train_accuracy_std": train_metrics["accuracy_std"],
        "test_accuracy": test_accuracy,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train symptom classifier")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", required=True, help="Output directory for model")
    args = parser.parse_args()

    metrics = train_and_evaluate(args.data, args.output)
    print(f"Training complete: {metrics}")
