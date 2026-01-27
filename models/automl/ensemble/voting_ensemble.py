"""Voting Ensemble for RNA Disease Prediction."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class VotingEnsemble:
    """Weighted voting ensemble of RNA prediction models.

    Combines predictions from multiple models using weighted averaging
    of prediction probabilities.
    """

    def __init__(self, voting: str = "soft"):
        """Initialize voting ensemble.

        Args:
            voting: Voting type ('soft' for probability averaging, 'hard' for majority)
        """
        self.voting = voting
        self.models: List[Dict[str, Any]] = []
        self.weights: List[float] = []
        self._total_weight: float = 0.0

    def add_model(
        self,
        model_path: str,
        weight: float = 1.0,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Add model to ensemble.

        Args:
            model_path: Path to model checkpoint
            weight: Weight for this model's predictions
            model: Optional pre-loaded model
        """
        self.models.append({
            "path": model_path,
            "weight": weight,
            "model": model,
        })
        self.weights.append(weight)
        self._total_weight += weight

    def load_models(self, device: torch.device) -> None:
        """Load all models into memory.

        Args:
            device: Device to load models on
        """
        for model_info in self.models:
            if model_info["model"] is None:
                model_path = Path(model_info["path"])
                if (model_path / "model.pt").exists():
                    # Load model state dict
                    state_dict = torch.load(
                        model_path / "model.pt",
                        map_location=device,
                    )
                    # Note: Actual model instantiation requires config
                    model_info["state_dict"] = state_dict
                    logger.debug(f"Loaded model from {model_path}")

    def _load_single_model(
        self,
        model_info: Dict[str, Any],
        device: torch.device,
        model_factory: callable,
    ) -> nn.Module:
        """Load a single model.

        Args:
            model_info: Model information dictionary
            device: Device to load on
            model_factory: Factory function to create model

        Returns:
            Loaded model
        """
        if model_info.get("model") is not None:
            return model_info["model"].to(device)

        # Create model and load weights
        model = model_factory()
        model_path = Path(model_info["path"]) / "model.pt"
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        model_factory: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """Make ensemble predictions.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            device: Device for inference
            model_factory: Optional factory to create models

        Returns:
            Dictionary with predictions:
                - rna_type_probs: RNA type probabilities
                - disease_probs: Disease probabilities
                - pathogenicity_probs: Pathogenicity probabilities
                - risk_scores: Risk scores
        """
        all_rna_probs = []
        all_disease_probs = []
        all_pathogenicity_probs = []
        all_risk_scores = []

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        for i, model_info in enumerate(self.models):
            weight = self.weights[i] / self._total_weight

            # Load model
            if model_factory:
                model = self._load_single_model(model_info, device, model_factory)
            else:
                model = model_info.get("model")
                if model is None:
                    logger.warning(f"Model {i} not loaded and no factory provided")
                    continue
                model = model.to(device)
                model.eval()

            # Get predictions
            outputs = model(input_ids, attention_mask)

            # Apply softmax and weight
            rna_probs = torch.softmax(outputs["rna_type_logits"], dim=-1).cpu().numpy()
            disease_probs = torch.softmax(outputs["disease_logits"], dim=-1).cpu().numpy()
            pathogenicity_probs = torch.softmax(
                outputs["pathogenicity_logits"], dim=-1
            ).cpu().numpy()
            risk_scores = outputs["risk_score"].cpu().numpy()

            all_rna_probs.append(rna_probs * weight)
            all_disease_probs.append(disease_probs * weight)
            all_pathogenicity_probs.append(pathogenicity_probs * weight)
            all_risk_scores.append(risk_scores * weight)

        # Combine predictions
        rna_ensemble = np.sum(all_rna_probs, axis=0)
        disease_ensemble = np.sum(all_disease_probs, axis=0)
        pathogenicity_ensemble = np.sum(all_pathogenicity_probs, axis=0)
        risk_ensemble = np.sum(all_risk_scores, axis=0)

        return {
            "rna_type_probs": rna_ensemble,
            "rna_type_pred": np.argmax(rna_ensemble, axis=-1),
            "disease_probs": disease_ensemble,
            "disease_pred": np.argmax(disease_ensemble, axis=-1),
            "pathogenicity_probs": pathogenicity_ensemble,
            "pathogenicity_pred": np.argmax(pathogenicity_ensemble, axis=-1),
            "risk_scores": risk_ensemble,
        }

    def evaluate(
        self,
        dataloader: "torch.utils.data.DataLoader",
        device: torch.device,
        model_factory: Optional[callable] = None,
    ) -> Dict[str, float]:
        """Evaluate ensemble on dataset.

        Args:
            dataloader: Data loader
            device: Device for inference
            model_factory: Optional factory to create models

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, f1_score

        all_preds = {"rna_type": [], "disease": [], "pathogenicity": [], "risk": []}
        all_labels = {"rna_type": [], "disease": [], "pathogenicity": [], "risk": []}

        for batch in dataloader:
            predictions = self.predict(
                batch["input_ids"],
                batch["attention_mask"],
                device,
                model_factory,
            )

            all_preds["rna_type"].extend(predictions["rna_type_pred"])
            all_preds["disease"].extend(predictions["disease_pred"])
            all_preds["pathogenicity"].extend(predictions["pathogenicity_pred"])
            all_preds["risk"].extend(predictions["risk_scores"])

            if "rna_type_label" in batch:
                all_labels["rna_type"].extend(batch["rna_type_label"].numpy())
            if "disease_label" in batch:
                all_labels["disease"].extend(batch["disease_label"].numpy())
            if "pathogenicity_label" in batch:
                all_labels["pathogenicity"].extend(batch["pathogenicity_label"].numpy())
            if "risk_label" in batch:
                all_labels["risk"].extend(batch["risk_label"].numpy())

        metrics = {}

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
            metrics["f1_macro"] = metrics["disease_f1_macro"]

        if all_labels["pathogenicity"]:
            metrics["pathogenicity_accuracy"] = accuracy_score(
                all_labels["pathogenicity"], all_preds["pathogenicity"]
            )

        if all_labels["risk"]:
            metrics["risk_mae"] = np.mean(
                np.abs(np.array(all_labels["risk"]) - np.array(all_preds["risk"]))
            )

        return metrics

    def save(self, path: str) -> None:
        """Save ensemble configuration.

        Args:
            path: Path to save directory
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "voting": self.voting,
            "models": [
                {"path": m["path"], "weight": m["weight"]}
                for m in self.models
            ],
            "weights": self.weights,
        }

        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved voting ensemble to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "VotingEnsemble":
        """Load ensemble configuration.

        Args:
            path: Path to save directory

        Returns:
            Loaded ensemble
        """
        load_dir = Path(path)

        with open(load_dir / "ensemble_config.json", "r") as f:
            config = json.load(f)

        ensemble = cls(voting=config["voting"])
        for model_info in config["models"]:
            ensemble.add_model(model_info["path"], model_info["weight"])

        return ensemble
