"""AutoML Trainer for RNA Disease Prediction.

Orchestrates hyperparameter optimization using Optuna with support
for ensemble building and model registry.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    optuna = None

from .config import AutoMLConfig, TrialConfig, SamplerType, PrunerType
from .search_space import RNAModelSearchSpace
from .storage.trial_storage import TrialStorage
from .storage.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result from a single AutoML trial."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    training_time_seconds: float = 0.0
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class AutoMLResult:
    """Result from complete AutoML run."""
    experiment_name: str
    best_trial: TrialResult
    all_trials: List[TrialResult] = field(default_factory=list)
    study_stats: Dict[str, Any] = field(default_factory=dict)
    ensemble_result: Optional[Dict[str, Any]] = None
    total_time_seconds: float = 0.0


class AutoMLTrainer:
    """AutoML trainer using Optuna for hyperparameter optimization.

    Provides automated hyperparameter search with support for:
    - TPE, CMA-ES, and random sampling
    - Hyperband and median pruning
    - Model checkpointing and registry
    - Ensemble building from top trials
    """

    def __init__(
        self,
        config: AutoMLConfig,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize AutoML trainer.

        Args:
            config: AutoML configuration
            train_data: Training data samples
            val_data: Validation data samples
            test_data: Optional test data samples
            device: Device for training
        """
        if optuna is None:
            raise ImportError("optuna is required for AutoML. Install with: pip install optuna")

        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        )

        # Storage
        self.storage = TrialStorage(config.checkpoint_dir)
        self.registry = ModelRegistry(config.checkpoint_dir)

        # Results
        self.trial_results: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None

    def _get_sampler(self) -> "optuna.samplers.BaseSampler":
        """Get Optuna sampler based on config."""
        seed = self.config.seed

        if self.config.sampler == SamplerType.TPE:
            return optuna.samplers.TPESampler(seed=seed)
        elif self.config.sampler == SamplerType.CMA_ES:
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif self.config.sampler == SamplerType.RANDOM:
            return optuna.samplers.RandomSampler(seed=seed)
        else:
            return optuna.samplers.TPESampler(seed=seed)

    def _get_pruner(self) -> "optuna.pruners.BasePruner":
        """Get Optuna pruner based on config."""
        if self.config.pruner == PrunerType.HYPERBAND:
            return optuna.pruners.HyperbandPruner(
                min_resource=5,
                max_resource=self.config.search_space.epochs_max,
                reduction_factor=3,
            )
        elif self.config.pruner == PrunerType.MEDIAN:
            return optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        elif self.config.pruner == PrunerType.SUCCESSIVE_HALVING:
            return optuna.pruners.SuccessiveHalvingPruner()
        else:
            return optuna.pruners.NopPruner()

    def _create_study(self) -> "optuna.Study":
        """Create Optuna study."""
        storage_url = self.config.get_storage_url()

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name=self.config.experiment_name,
            direction=self.config.direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
            storage=storage_url if storage_url else None,
            load_if_exists=True,
        )

        return study

    def _build_model_from_config(self, trial_config: TrialConfig) -> nn.Module:
        """Build model from trial configuration.

        Args:
            trial_config: Trial configuration

        Returns:
            Initialized model
        """
        from ..rna_predictor.config import RNAEncoderConfig, RNAClassifierConfig
        from ..rna_predictor.model import RNAPredictionModel

        encoder_config = RNAEncoderConfig(
            vocab_size=trial_config.vocab_size,
            hidden_size=trial_config.hidden_size,
            num_hidden_layers=trial_config.num_hidden_layers,
            num_attention_heads=trial_config.num_attention_heads,
            intermediate_size=trial_config.intermediate_size,
            hidden_dropout_prob=trial_config.hidden_dropout_prob,
            attention_probs_dropout_prob=trial_config.attention_probs_dropout_prob,
            n_gram_sizes=trial_config.n_gram_sizes,
        )

        classifier_config = RNAClassifierConfig(
            hidden_size=trial_config.hidden_size,
            classifier_dropout=trial_config.classifier_dropout,
            hidden_dims=(trial_config.classifier_hidden_1, trial_config.classifier_hidden_2),
        )

        return RNAPredictionModel(encoder_config, classifier_config)

    def _train_trial(
        self,
        trial: "optuna.Trial",
        trial_config: TrialConfig,
    ) -> Dict[str, float]:
        """Train a single trial.

        Args:
            trial: Optuna trial
            trial_config: Trial configuration

        Returns:
            Evaluation metrics
        """
        from ..rna_predictor.tokenizer import HybridNGramTokenizer
        from ..rna_predictor.trainer import RNATrainer, RNADataset, TrainingConfig

        # Build model and tokenizer
        model = self._build_model_from_config(trial_config)
        tokenizer = HybridNGramTokenizer(
            n_gram_sizes=trial_config.n_gram_sizes,
            max_length=self.config.max_seq_length,
        )

        # Create training config
        training_config = TrainingConfig(
            epochs=trial_config.epochs,
            batch_size=trial_config.batch_size,
            learning_rate=trial_config.learning_rate,
            weight_decay=trial_config.weight_decay,
            warmup_ratio=trial_config.warmup_ratio,
            max_grad_norm=trial_config.gradient_clip_norm,
            early_stopping_patience=trial_config.early_stopping_patience,
            rna_type_weight=trial_config.rna_type_weight,
            disease_weight=trial_config.disease_weight,
            pathogenicity_weight=trial_config.pathogenicity_weight,
            risk_weight=trial_config.risk_weight,
            max_seq_length=self.config.max_seq_length,
        )

        # Create datasets
        train_dataset = RNADataset(self.train_data, tokenizer, self.config.max_seq_length)
        val_dataset = RNADataset(self.val_data, tokenizer, self.config.max_seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=trial_config.batch_size,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=trial_config.batch_size,
            shuffle=False,
            num_workers=2,
        )

        # Create trainer
        trainer = RNATrainer(model, tokenizer, training_config, self.device)

        # Training loop with pruning
        best_metric = 0.0
        patience_counter = 0

        for epoch in range(trial_config.epochs):
            # Train epoch
            trainer.train_epoch(train_loader, trainer.model.parameters())

            # Evaluate
            metrics = trainer.evaluate(val_loader)
            current_metric = metrics.get(self.config.objective_metric, 0.0)

            # Report for pruning
            trial.report(current_metric, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Track best
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

                # Save checkpoint
                model_path = self.registry.save_trial_model(
                    model=model,
                    tokenizer=tokenizer,
                    trial_id=trial.number,
                    metrics=metrics,
                    experiment_name=self.config.experiment_name,
                )
            else:
                patience_counter += 1
                if patience_counter >= trial_config.early_stopping_patience:
                    break

        return metrics

    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial

        Returns:
            Objective metric value
        """
        start_time = datetime.now()

        try:
            # Sample hyperparameters
            trial_config = RNAModelSearchSpace.sample_all(
                trial, self.config.search_space
            )

            # Train
            metrics = self._train_trial(trial, trial_config)
            objective_value = metrics.get(self.config.objective_metric, 0.0)

            # Record result
            training_time = (datetime.now() - start_time).total_seconds()
            result = TrialResult(
                trial_id=trial.number,
                params=trial.params,
                metrics=metrics,
                model_path=self.registry.get_trial_path(
                    self.config.experiment_name, trial.number
                ),
                training_time_seconds=training_time,
                status="completed",
            )
            self.trial_results.append(result)
            self.storage.save_trial_result(result)

            return objective_value

        except optuna.TrialPruned:
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            result = TrialResult(
                trial_id=trial.number,
                params=trial.params if hasattr(trial, "params") else {},
                metrics={},
                status="failed",
                error=str(e),
            )
            self.trial_results.append(result)
            raise

    def run(self) -> AutoMLResult:
        """Run AutoML optimization.

        Returns:
            AutoML result with best trial and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting AutoML experiment: {self.config.experiment_name}")
        logger.info(f"Target: {self.config.n_trials} trials, {self.config.timeout_hours}h timeout")

        # Create study
        study = self._create_study()

        # Run optimization
        timeout = self.config.timeout_hours * 3600 if self.config.timeout_hours else None
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Get results
        total_time = (datetime.now() - start_time).total_seconds()

        # Best trial
        best_optuna_trial = study.best_trial
        self.best_trial = TrialResult(
            trial_id=best_optuna_trial.number,
            params=best_optuna_trial.params,
            metrics={"objective": best_optuna_trial.value},
            status="best",
        )

        # Study statistics
        completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == TrialState.FAIL])

        study_stats = {
            "n_trials": len(study.trials),
            "n_completed": completed,
            "n_pruned": pruned,
            "n_failed": failed,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
        }

        # Build ensemble if enabled
        ensemble_result = None
        if self.config.enable_ensemble and completed >= self.config.ensemble_top_k:
            ensemble_result = self._build_ensemble(study)

        result = AutoMLResult(
            experiment_name=self.config.experiment_name,
            best_trial=self.best_trial,
            all_trials=self.trial_results,
            study_stats=study_stats,
            ensemble_result=ensemble_result,
            total_time_seconds=total_time,
        )

        # Save final result
        self.storage.save_experiment_result(result)

        logger.info(f"AutoML complete. Best {self.config.objective_metric}: {study.best_value:.4f}")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")

        return result

    def _build_ensemble(self, study: "optuna.Study") -> Dict[str, Any]:
        """Build ensemble from top trials.

        Args:
            study: Completed Optuna study

        Returns:
            Ensemble configuration and metrics
        """
        from .ensemble.hybrid_ensemble import HybridEnsemble

        # Get top K completed trials
        completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t.value,
            reverse=(self.config.direction == "maximize"),
        )[: self.config.ensemble_top_k]

        # Build ensemble
        ensemble = HybridEnsemble(strategy=self.config.ensemble_strategy.value)

        for trial in sorted_trials:
            model_path = self.registry.get_trial_path(
                self.config.experiment_name, trial.number
            )
            ensemble.add_model(model_path, weight=trial.value)

        # Save ensemble
        ensemble_path = Path(self.config.checkpoint_dir) / "ensemble"
        ensemble.save(str(ensemble_path))

        return {
            "strategy": self.config.ensemble_strategy.value,
            "n_models": len(sorted_trials),
            "trial_ids": [t.number for t in sorted_trials],
            "weights": [t.value for t in sorted_trials],
            "path": str(ensemble_path),
        }

    def get_param_importance(self, study: "optuna.Study") -> Dict[str, float]:
        """Get hyperparameter importance analysis.

        Args:
            study: Completed Optuna study

        Returns:
            Dictionary of parameter importances
        """
        try:
            importance = optuna.importance.get_param_importances(study)
            return importance
        except Exception as e:
            logger.warning(f"Could not compute param importance: {e}")
            return {}

    def get_optimization_history(
        self, study: "optuna.Study"
    ) -> Dict[str, List[float]]:
        """Get optimization history for visualization.

        Args:
            study: Completed Optuna study

        Returns:
            Dictionary with trial values and best values
        """
        values = []
        best_values = []
        current_best = float("-inf") if self.config.direction == "maximize" else float("inf")

        for trial in study.trials:
            if trial.state == TrialState.COMPLETE:
                values.append(trial.value)
                if self.config.direction == "maximize":
                    current_best = max(current_best, trial.value)
                else:
                    current_best = min(current_best, trial.value)
                best_values.append(current_best)

        return {
            "trial_values": values,
            "best_values": best_values,
        }
