"""Hybrid Ensemble combining Deep Learning and Traditional ML models."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
import numpy as np

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import joblib
except ImportError:
    xgb = None
    RandomForestClassifier = None
    LogisticRegression = None
    joblib = None

from .voting_ensemble import VotingEnsemble

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """Hybrid ensemble combining Transformer models with XGBoost/Random Forest.

    Provides multiple combination strategies:
    - voting: Weighted average of deep model predictions
    - stacking: Meta-learner trained on base model predictions
    - hybrid: Combine deep model embeddings with traditional ML
    """

    def __init__(self, strategy: str = "hybrid"):
        """Initialize hybrid ensemble.

        Args:
            strategy: Combination strategy ('voting', 'stacking', 'hybrid')
        """
        self.strategy = strategy
        self.voting_ensemble = VotingEnsemble()
        self.ml_models: Dict[str, Any] = {}
        self.meta_learner: Optional[Any] = None

        # Weights for hybrid combination
        self.deep_weight = 0.7
        self.ml_weight = 0.3

    def add_model(
        self,
        model_path: str,
        weight: float = 1.0,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Add deep learning model to ensemble.

        Args:
            model_path: Path to model checkpoint
            weight: Weight for this model
            model: Optional pre-loaded model
        """
        self.voting_ensemble.add_model(model_path, weight, model)

    def add_xgboost(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train and add XGBoost model.

        Args:
            features: Training features
            labels: Training labels
            params: XGBoost parameters
        """
        if xgb is None:
            logger.warning("XGBoost not installed, skipping")
            return

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        params = {**default_params, **(params or {})}

        model = xgb.XGBClassifier(**params)
        model.fit(features, labels)
        self.ml_models["xgboost"] = model

        logger.info("Added XGBoost model to ensemble")

    def add_random_forest(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train and add Random Forest model.

        Args:
            features: Training features
            labels: Training labels
            params: Random Forest parameters
        """
        if RandomForestClassifier is None:
            logger.warning("scikit-learn not installed, skipping")
            return

        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "n_jobs": -1,
        }
        params = {**default_params, **(params or {})}

        model = RandomForestClassifier(**params)
        model.fit(features, labels)
        self.ml_models["random_forest"] = model

        logger.info("Added Random Forest model to ensemble")

    def train_meta_learner(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Train meta-learner for stacking.

        Args:
            features: Stacked predictions from base models
            labels: True labels
        """
        if LogisticRegression is None:
            logger.warning("scikit-learn not installed, skipping meta-learner")
            return

        self.meta_learner = LogisticRegression(max_iter=1000, multi_class="multinomial")
        self.meta_learner.fit(features, labels)

        logger.info("Trained meta-learner for stacking")

    def _get_ml_predictions(
        self,
        features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get predictions from ML models.

        Args:
            features: Input features

        Returns:
            Dictionary of predictions by model
        """
        predictions = {}

        for name, model in self.ml_models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)
                predictions[name] = {
                    "probs": probs,
                    "preds": np.argmax(probs, axis=-1),
                }

        return predictions

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        model_factory: Optional[callable] = None,
        extract_features: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """Make ensemble predictions.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            device: Device for inference
            model_factory: Factory to create deep models
            extract_features: Function to extract features for ML models

        Returns:
            Prediction dictionary
        """
        # Get deep model predictions
        deep_preds = self.voting_ensemble.predict(
            input_ids, attention_mask, device, model_factory
        )

        if self.strategy == "voting":
            return deep_preds

        # For hybrid/stacking, we need ML model predictions
        if not self.ml_models:
            return deep_preds

        # Extract features for ML models
        if extract_features is not None:
            ml_features = extract_features(input_ids, attention_mask, device)
        else:
            # Use deep model probabilities as features
            ml_features = np.hstack([
                deep_preds["rna_type_probs"],
                deep_preds["disease_probs"],
                deep_preds["pathogenicity_probs"],
                deep_preds["risk_scores"].reshape(-1, 1),
            ])

        # Get ML predictions
        ml_preds = self._get_ml_predictions(ml_features)

        if self.strategy == "stacking" and self.meta_learner is not None:
            # Stack all predictions for meta-learner
            stacked = np.hstack([
                deep_preds["disease_probs"],
                *[p["probs"] for p in ml_preds.values()],
            ])
            final_probs = self.meta_learner.predict_proba(stacked)
            final_preds = np.argmax(final_probs, axis=-1)

            return {
                **deep_preds,
                "disease_probs": final_probs,
                "disease_pred": final_preds,
            }

        else:  # hybrid strategy
            # Weighted combination
            deep_probs = deep_preds["disease_probs"]

            if ml_preds:
                ml_probs = np.mean(
                    [p["probs"] for p in ml_preds.values()],
                    axis=0,
                )
                combined_probs = (
                    self.deep_weight * deep_probs +
                    self.ml_weight * ml_probs
                )
            else:
                combined_probs = deep_probs

            return {
                **deep_preds,
                "disease_probs": combined_probs,
                "disease_pred": np.argmax(combined_probs, axis=-1),
            }

    def evaluate(
        self,
        dataloader: "torch.utils.data.DataLoader",
        device: torch.device,
        model_factory: Optional[callable] = None,
    ) -> Dict[str, float]:
        """Evaluate ensemble.

        Args:
            dataloader: Data loader
            device: Device for inference
            model_factory: Factory to create models

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, f1_score

        all_preds = []
        all_labels = []

        for batch in dataloader:
            predictions = self.predict(
                batch["input_ids"],
                batch["attention_mask"],
                device,
                model_factory,
            )

            all_preds.extend(predictions["disease_pred"])
            if "disease_label" in batch:
                all_labels.extend(batch["disease_label"].numpy())

        if not all_labels:
            return {}

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        }

    def save(self, path: str) -> None:
        """Save ensemble.

        Args:
            path: Save directory
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "strategy": self.strategy,
            "deep_weight": self.deep_weight,
            "ml_weight": self.ml_weight,
            "ml_models": list(self.ml_models.keys()),
        }
        with open(save_dir / "hybrid_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save voting ensemble
        self.voting_ensemble.save(str(save_dir / "voting"))

        # Save ML models
        if joblib is not None:
            for name, model in self.ml_models.items():
                if name == "xgboost" and xgb is not None:
                    model.save_model(str(save_dir / f"{name}.json"))
                else:
                    joblib.dump(model, save_dir / f"{name}.pkl")

        # Save meta-learner
        if self.meta_learner is not None and joblib is not None:
            joblib.dump(self.meta_learner, save_dir / "meta_learner.pkl")

        logger.info(f"Saved hybrid ensemble to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "HybridEnsemble":
        """Load ensemble.

        Args:
            path: Save directory

        Returns:
            Loaded ensemble
        """
        load_dir = Path(path)

        # Load config
        with open(load_dir / "hybrid_config.json", "r") as f:
            config = json.load(f)

        ensemble = cls(strategy=config["strategy"])
        ensemble.deep_weight = config.get("deep_weight", 0.7)
        ensemble.ml_weight = config.get("ml_weight", 0.3)

        # Load voting ensemble
        ensemble.voting_ensemble = VotingEnsemble.load(str(load_dir / "voting"))

        # Load ML models
        if joblib is not None:
            for name in config.get("ml_models", []):
                if name == "xgboost" and xgb is not None:
                    model_path = load_dir / f"{name}.json"
                    if model_path.exists():
                        model = xgb.XGBClassifier()
                        model.load_model(str(model_path))
                        ensemble.ml_models[name] = model
                else:
                    model_path = load_dir / f"{name}.pkl"
                    if model_path.exists():
                        ensemble.ml_models[name] = joblib.load(model_path)

        # Load meta-learner
        meta_path = load_dir / "meta_learner.pkl"
        if meta_path.exists() and joblib is not None:
            ensemble.meta_learner = joblib.load(meta_path)

        return ensemble
