"""Model Registry for AutoML experiments."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import logging
import shutil

import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for trained models from AutoML experiments.

    Manages model artifacts, versioning, and metadata for trained models.
    """

    def __init__(self, base_dir: str = "./checkpoints"):
        """Initialize model registry.

        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": {}, "experiments": {}}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def get_trial_path(self, experiment_name: str, trial_id: int) -> str:
        """Get path for trial model.

        Args:
            experiment_name: Experiment name
            trial_id: Trial ID

        Returns:
            Path to trial model directory
        """
        return str(self.models_dir / experiment_name / f"trial_{trial_id}")

    def save_trial_model(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        trial_id: int,
        metrics: Dict[str, float],
        experiment_name: str,
    ) -> str:
        """Save model from a trial.

        Args:
            model: Trained model
            tokenizer: Tokenizer instance
            trial_id: Trial ID
            metrics: Evaluation metrics
            experiment_name: Experiment name

        Returns:
            Path to saved model
        """
        model_dir = Path(self.get_trial_path(experiment_name, trial_id))
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), model_dir / "model.pt")

        # Save tokenizer
        tokenizer.save(str(model_dir / "tokenizer"))

        # Save metadata
        metadata = {
            "trial_id": trial_id,
            "experiment_name": experiment_name,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        model_key = f"{experiment_name}/trial_{trial_id}"
        self.registry["models"][model_key] = {
            "path": str(model_dir),
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
        }

        if experiment_name not in self.registry["experiments"]:
            self.registry["experiments"][experiment_name] = []
        self.registry["experiments"][experiment_name].append(trial_id)

        self._save_registry()

        logger.debug(f"Saved trial model to {model_dir}")
        return str(model_dir)

    def load_trial_model(
        self,
        experiment_name: str,
        trial_id: int,
        model_class: type,
        device: torch.device,
    ) -> torch.nn.Module:
        """Load model from a trial.

        Args:
            experiment_name: Experiment name
            trial_id: Trial ID
            model_class: Model class to instantiate
            device: Device to load model on

        Returns:
            Loaded model
        """
        model_dir = Path(self.get_trial_path(experiment_name, trial_id))

        # Load metadata for config
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load model weights
        model_path = model_dir / "model.pt"
        state_dict = torch.load(model_path, map_location=device)

        # Note: Caller needs to provide configured model
        # This is a placeholder - actual implementation would reconstruct config
        raise NotImplementedError(
            "Model loading requires reconstructing config from trial params"
        )

    def get_model_path(self, experiment_name: str, trial_id: int) -> str:
        """Get path to saved model.

        Args:
            experiment_name: Experiment name
            trial_id: Trial ID

        Returns:
            Path to model directory
        """
        return self.get_trial_path(experiment_name, trial_id)

    def list_models(
        self, experiment_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models.

        Args:
            experiment_name: Optional filter by experiment

        Returns:
            List of model metadata
        """
        models = []
        for key, data in self.registry["models"].items():
            if experiment_name:
                if key.startswith(f"{experiment_name}/"):
                    models.append({"key": key, **data})
            else:
                models.append({"key": key, **data})

        return models

    def get_best_model(
        self,
        experiment_name: str,
        metric: str = "f1_macro",
    ) -> Optional[Dict[str, Any]]:
        """Get best model from an experiment.

        Args:
            experiment_name: Experiment name
            metric: Metric to compare

        Returns:
            Best model metadata or None
        """
        models = self.list_models(experiment_name)
        if not models:
            return None

        best = max(
            models,
            key=lambda m: m.get("metrics", {}).get(metric, 0),
        )
        return best

    def delete_model(self, experiment_name: str, trial_id: int) -> bool:
        """Delete a model.

        Args:
            experiment_name: Experiment name
            trial_id: Trial ID

        Returns:
            True if deleted, False if not found
        """
        model_key = f"{experiment_name}/trial_{trial_id}"
        model_dir = Path(self.get_trial_path(experiment_name, trial_id))

        if model_key in self.registry["models"]:
            del self.registry["models"][model_key]

            if experiment_name in self.registry["experiments"]:
                trials = self.registry["experiments"][experiment_name]
                if trial_id in trials:
                    trials.remove(trial_id)

            self._save_registry()

        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True

        return False

    def cleanup_experiment(
        self,
        experiment_name: str,
        keep_top_k: int = 5,
        metric: str = "f1_macro",
    ) -> int:
        """Clean up experiment, keeping only top K models.

        Args:
            experiment_name: Experiment name
            keep_top_k: Number of models to keep
            metric: Metric for ranking

        Returns:
            Number of models deleted
        """
        models = self.list_models(experiment_name)
        if len(models) <= keep_top_k:
            return 0

        # Sort by metric
        sorted_models = sorted(
            models,
            key=lambda m: m.get("metrics", {}).get(metric, 0),
            reverse=True,
        )

        # Delete models beyond top K
        to_delete = sorted_models[keep_top_k:]
        deleted = 0

        for model in to_delete:
            # Extract trial_id from key
            key = model["key"]
            trial_id = int(key.split("trial_")[1])
            if self.delete_model(experiment_name, trial_id):
                deleted += 1

        logger.info(
            f"Cleaned up {deleted} models from {experiment_name}, "
            f"kept top {keep_top_k}"
        )
        return deleted
