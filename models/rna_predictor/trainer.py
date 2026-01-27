"""Training pipeline for RNA disease prediction model."""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import RNAEncoderConfig, RNAClassifierConfig, DISEASE_CLASSES, RNA_TYPES
from .model import RNAPredictionModel
from .tokenizer import HybridNGramTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "f1_macro"

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    max_seq_length: int = 512

    # Logging
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

    # Loss weights for multi-task learning
    rna_type_weight: float = 0.2
    disease_weight: float = 0.5
    pathogenicity_weight: float = 0.2
    risk_weight: float = 0.1


class RNADataset(Dataset):
    """Dataset for RNA sequence disease prediction."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: HybridNGramTokenizer,
        max_length: int = 512,
    ):
        """Initialize dataset.

        Args:
            data: List of sample dictionaries with keys:
                - sequence: RNA sequence
                - rna_type: RNA type label (optional)
                - disease: Disease label (optional)
                - pathogenicity: Pathogenicity label (optional)
                - risk_score: Risk score 0-100 (optional)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build label mappings
        self.rna_type_to_idx = {t: i for i, t in enumerate(RNA_TYPES)}
        self.disease_to_idx = {d[0]: i for i, d in enumerate(DISEASE_CLASSES)}
        self.pathogenicity_to_idx = {
            "benign": 0, "likely_benign": 1, "uncertain": 2,
            "likely_pathogenic": 3, "pathogenic": 4,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Encode sequence
        encoded = self.tokenizer.encode(
            sample["sequence"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        item = {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
        }

        # Add labels if present
        if "rna_type" in sample:
            item["rna_type_label"] = torch.tensor(
                self.rna_type_to_idx.get(sample["rna_type"], 0)
            )

        if "disease" in sample:
            item["disease_label"] = torch.tensor(
                self.disease_to_idx.get(sample["disease"], 0)
            )

        if "pathogenicity" in sample:
            item["pathogenicity_label"] = torch.tensor(
                self.pathogenicity_to_idx.get(sample["pathogenicity"], 2)
            )

        if "risk_score" in sample:
            item["risk_label"] = torch.tensor(sample["risk_score"] / 100.0)

        return item


class RNATrainer:
    """Trainer for RNA disease prediction model."""

    def __init__(
        self,
        model: RNAPredictionModel,
        tokenizer: HybridNGramTokenizer,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer instance
            config: Training configuration
            device: Device for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.training_history: List[Dict] = []

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss.

        Args:
            outputs: Model outputs
            labels: Ground truth labels

        Returns:
            Tuple of total loss and loss breakdown
        """
        losses = {}
        total_loss = 0.0

        if "rna_type_label" in labels:
            rna_loss = self.ce_loss(
                outputs["rna_type_logits"],
                labels["rna_type_label"],
            )
            losses["rna_type_loss"] = rna_loss.item()
            total_loss += self.config.rna_type_weight * rna_loss

        if "disease_label" in labels:
            disease_loss = self.ce_loss(
                outputs["disease_logits"],
                labels["disease_label"],
            )
            losses["disease_loss"] = disease_loss.item()
            total_loss += self.config.disease_weight * disease_loss

        if "pathogenicity_label" in labels:
            pathogenicity_loss = self.ce_loss(
                outputs["pathogenicity_logits"],
                labels["pathogenicity_label"],
            )
            losses["pathogenicity_loss"] = pathogenicity_loss.item()
            total_loss += self.config.pathogenicity_weight * pathogenicity_loss

        if "risk_label" in labels:
            risk_loss = self.mse_loss(
                outputs["risk_score"],
                labels["risk_label"],
            )
            losses["risk_loss"] = risk_loss.item()
            total_loss += self.config.risk_weight * risk_loss

        losses["total_loss"] = total_loss.item()
        return total_loss, losses

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses: Dict[str, List[float]] = {}

        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k.endswith("_label")
            }

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)

            # Compute loss
            loss, loss_dict = self.compute_loss(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Track losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)

            self.global_step += 1

        # Average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        all_preds = {
            "rna_type": [], "disease": [], "pathogenicity": [], "risk": [],
        }
        all_labels = {
            "rna_type": [], "disease": [], "pathogenicity": [], "risk": [],
        }
        all_losses: Dict[str, List[float]] = {}

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k.endswith("_label")
            }

            outputs = self.model(input_ids, attention_mask)
            _, loss_dict = self.compute_loss(outputs, labels)

            # Track losses
            for k, v in loss_dict.items():
                if k not in all_losses:
                    all_losses[k] = []
                all_losses[k].append(v)

            # Collect predictions
            if "rna_type_label" in labels:
                all_preds["rna_type"].extend(
                    outputs["rna_type_logits"].argmax(dim=1).cpu().numpy()
                )
                all_labels["rna_type"].extend(
                    labels["rna_type_label"].cpu().numpy()
                )

            if "disease_label" in labels:
                all_preds["disease"].extend(
                    outputs["disease_logits"].argmax(dim=1).cpu().numpy()
                )
                all_labels["disease"].extend(
                    labels["disease_label"].cpu().numpy()
                )

            if "pathogenicity_label" in labels:
                all_preds["pathogenicity"].extend(
                    outputs["pathogenicity_logits"].argmax(dim=1).cpu().numpy()
                )
                all_labels["pathogenicity"].extend(
                    labels["pathogenicity_label"].cpu().numpy()
                )

            if "risk_label" in labels:
                all_preds["risk"].extend(
                    outputs["risk_score"].cpu().numpy()
                )
                all_labels["risk"].extend(
                    labels["risk_label"].cpu().numpy()
                )

        # Compute metrics
        metrics = {k: np.mean(v) for k, v in all_losses.items()}

        if all_labels["rna_type"]:
            metrics["rna_type_accuracy"] = accuracy_score(
                all_labels["rna_type"], all_preds["rna_type"]
            )

        if all_labels["disease"]:
            metrics["disease_accuracy"] = accuracy_score(
                all_labels["disease"], all_preds["disease"]
            )
            metrics["disease_f1_macro"] = f1_score(
                all_labels["disease"], all_preds["disease"], average="macro"
            )
            metrics["disease_f1_weighted"] = f1_score(
                all_labels["disease"], all_preds["disease"], average="weighted"
            )

        if all_labels["pathogenicity"]:
            metrics["pathogenicity_accuracy"] = accuracy_score(
                all_labels["pathogenicity"], all_preds["pathogenicity"]
            )

        if all_labels["risk"]:
            metrics["risk_mae"] = np.mean(
                np.abs(np.array(all_labels["risk"]) - np.array(all_preds["risk"]))
            )

        # Overall metric for early stopping
        metrics["f1_macro"] = metrics.get("disease_f1_macro", 0.0)

        return metrics

    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        output_path: str,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_data: Training samples
            val_data: Validation samples
            output_path: Path to save model and logs

        Returns:
            Dictionary with training results
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets
        train_dataset = RNADataset(train_data, self.tokenizer, self.config.max_seq_length)
        val_dataset = RNADataset(val_data, self.tokenizer, self.config.max_seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * self.config.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )

        # Training loop
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        for epoch in range(self.config.epochs):
            # Train epoch
            train_losses = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_losses['total_loss']:.4f}")

            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"F1: {val_metrics.get('f1_macro', 0):.4f}"
            )

            # Save history
            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_losses,
                "val": val_metrics,
            })

            # Early stopping check
            current_metric = val_metrics.get(self.config.early_stopping_metric, 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(output_dir / "best_model")
                logger.info(f"New best {self.config.early_stopping_metric}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

        # Save final model
        self.save_checkpoint(output_dir / "final_model")

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        return {
            "best_metric": self.best_metric,
            "epochs_trained": epoch + 1,
            "final_metrics": val_metrics,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save directory
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "model.pt")
        self.tokenizer.save(str(save_dir / "tokenizer"))

        logger.info(f"Saved checkpoint to {save_dir}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to save directory
        """
        load_dir = Path(path)
        self.model.load_state_dict(
            torch.load(load_dir / "model.pt", map_location=self.device)
        )
        self.tokenizer = HybridNGramTokenizer.load(str(load_dir / "tokenizer"))

        logger.info(f"Loaded checkpoint from {load_dir}")


def load_training_data(path: str) -> List[Dict]:
    """Load training data from JSON file.

    Expected format:
    [
        {
            "sequence": "AUGCUAGCUA...",
            "rna_type": "mRNA",
            "disease": "정상/저위험",
            "pathogenicity": "benign",
            "risk_score": 15.0
        },
        ...
    ]

    Args:
        path: Path to JSON file

    Returns:
        List of sample dictionaries
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Handle both direct list and wrapped format
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    return data


def train_rna_model(
    data_path: str,
    output_path: str,
    encoder_config: Optional[RNAEncoderConfig] = None,
    classifier_config: Optional[RNAClassifierConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """Main training entry point.

    Args:
        data_path: Path to training data JSON
        output_path: Path for saving model and logs
        encoder_config: Encoder configuration
        classifier_config: Classifier configuration
        training_config: Training configuration

    Returns:
        Training results dictionary
    """
    # Load data
    data = load_training_data(data_path)

    # Split data
    config = training_config or TrainingConfig()
    n_train = int(len(data) * config.train_split)
    n_val = int(len(data) * config.val_split)

    np.random.shuffle(data)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]

    # Create model and tokenizer
    enc_config = encoder_config or RNAEncoderConfig()
    cls_config = classifier_config or RNAClassifierConfig()

    model = RNAPredictionModel(enc_config, cls_config)
    tokenizer = HybridNGramTokenizer(
        n_gram_sizes=enc_config.n_gram_sizes,
        max_length=enc_config.max_position_embeddings,
    )

    # Train
    trainer = RNATrainer(model, tokenizer, config)
    results = trainer.train(train_data, val_data, output_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RNA disease prediction model")
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    results = train_rna_model(
        args.data,
        args.output,
        training_config=training_config,
    )

    print(f"Training complete. Best {training_config.early_stopping_metric}: {results['best_metric']:.4f}")
