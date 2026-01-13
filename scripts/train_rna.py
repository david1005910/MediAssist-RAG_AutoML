#!/usr/bin/env python3
"""Training script for RNA disease prediction model."""

import logging
import json
import torch
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    from models.rna_predictor.config import RNAEncoderConfig, RNAClassifierConfig
    from models.rna_predictor.model import RNAPredictionModel
    from models.rna_predictor.tokenizer import HybridNGramTokenizer
    from models.rna_predictor.trainer import (
        RNATrainer, TrainingConfig, RNADataset, load_training_data
    )
    from torch.utils.data import DataLoader

    # Load data
    logger.info("Loading training data...")
    data = load_training_data("data/sample_rna_training_data.json")
    logger.info(f"Loaded {len(data)} samples")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(data)
    n_train = int(len(data) * 0.8)
    n_val = int(len(data) * 0.1)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Use smaller model for faster training
    logger.info("Creating model...")
    encoder_config = RNAEncoderConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    classifier_config = RNAClassifierConfig(hidden_size=128)

    model = RNAPredictionModel(encoder_config, classifier_config)
    tokenizer = HybridNGramTokenizer(
        n_gram_sizes=encoder_config.n_gram_sizes,
        max_length=encoder_config.max_position_embeddings,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model = model.to(device)

    # Training config
    training_config = TrainingConfig(
        epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        early_stopping_patience=5,
    )

    # Create datasets with num_workers=0 for compatibility
    train_dataset = RNADataset(train_data, tokenizer, training_config.max_seq_length)
    val_dataset = RNADataset(val_data, tokenizer, training_config.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    total_steps = len(train_loader) * training_config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config.learning_rate,
        total_steps=total_steps,
        pct_start=training_config.warmup_ratio,
    )

    # Training loop
    logger.info(f"Starting training for {training_config.epochs} epochs")

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    best_metric = 0.0
    patience_counter = 0
    training_history = []

    output_dir = Path("models/rna_predictor/weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config.epochs):
        # Train
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Compute loss
            total_loss = 0.0

            if "rna_type_label" in batch:
                rna_loss = ce_loss(outputs["rna_type_logits"], batch["rna_type_label"].to(device))
                total_loss += 0.2 * rna_loss

            if "disease_label" in batch:
                disease_loss = ce_loss(outputs["disease_logits"], batch["disease_label"].to(device))
                total_loss += 0.5 * disease_loss

            if "pathogenicity_label" in batch:
                patho_loss = ce_loss(outputs["pathogenicity_logits"], batch["pathogenicity_label"].to(device))
                total_loss += 0.2 * patho_loss

            if "risk_label" in batch:
                risk_loss = mse_loss(outputs["risk_score"], batch["risk_label"].to(device))
                total_loss += 0.1 * risk_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(total_loss.item())

        avg_train_loss = np.mean(epoch_losses)

        # Evaluate
        model.eval()
        val_losses = []
        all_disease_preds = []
        all_disease_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)

                total_loss = 0.0
                if "disease_label" in batch:
                    disease_loss = ce_loss(outputs["disease_logits"], batch["disease_label"].to(device))
                    total_loss += disease_loss
                    all_disease_preds.extend(outputs["disease_logits"].argmax(dim=1).cpu().numpy())
                    all_disease_labels.extend(batch["disease_label"].numpy())

                val_losses.append(total_loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else 0.0

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        if all_disease_labels:
            accuracy = accuracy_score(all_disease_labels, all_disease_preds)
            f1 = f1_score(all_disease_labels, all_disease_preds, average="macro", zero_division=0)
        else:
            accuracy = 0.0
            f1 = 0.0

        logger.info(
            f"Epoch {epoch + 1}/{training_config.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "f1_macro": f1,
        })

        # Early stopping
        if f1 > best_metric:
            best_metric = f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save(str(output_dir / "tokenizer"))
            logger.info(f"  -> New best F1: {best_metric:.4f}, model saved!")
        else:
            patience_counter += 1
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best F1-macro: {best_metric:.4f}")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
