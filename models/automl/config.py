"""AutoML Configuration Schemas."""

from typing import Dict, List, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"


class SamplerType(str, Enum):
    """Optuna sampler types."""
    TPE = "tpe"
    CMA_ES = "cma_es"
    RANDOM = "random"
    GRID = "grid"


class PrunerType(str, Enum):
    """Optuna pruner types."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


class EnsembleStrategy(str, Enum):
    """Ensemble combination strategies."""
    VOTING = "voting"
    STACKING = "stacking"
    HYBRID = "hybrid"
    BEST_SINGLE = "best_single"


@dataclass
class SearchSpaceConfig:
    """Hyperparameter search space configuration.

    Defines the ranges and choices for hyperparameter optimization.
    """
    # Learning rate
    learning_rate_min: float = 1e-6
    learning_rate_max: float = 1e-3
    learning_rate_log: bool = True

    # Batch size
    batch_size_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Dropout rates
    dropout_min: float = 0.0
    dropout_max: float = 0.5

    # Architecture params
    hidden_size_choices: List[int] = field(default_factory=lambda: [256, 384, 512, 768])
    num_layers_min: int = 4
    num_layers_max: int = 12
    num_attention_heads_choices: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    intermediate_size_choices: List[int] = field(default_factory=lambda: [512, 1024, 2048, 3072])

    # Classifier hidden dims
    classifier_hidden_1_choices: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    classifier_hidden_2_choices: List[int] = field(default_factory=lambda: [64, 128, 192, 256])

    # N-gram tokenizer
    ngram_sizes_options: List[Tuple[int, ...]] = field(
        default_factory=lambda: [(1, 3, 5), (3, 5), (3, 4, 5), (3, 5, 7)]
    )
    vocab_size_choices: List[int] = field(default_factory=lambda: [4096, 8192, 16384])

    # Training
    epochs_min: int = 10
    epochs_max: int = 100
    early_stopping_patience_min: int = 3
    early_stopping_patience_max: int = 15
    weight_decay_min: float = 0.0
    weight_decay_max: float = 0.1
    warmup_ratio_min: float = 0.0
    warmup_ratio_max: float = 0.2
    gradient_clip_min: float = 0.5
    gradient_clip_max: float = 2.0

    # Multi-task loss weights
    loss_weight_rna_type_min: float = 0.1
    loss_weight_rna_type_max: float = 0.3
    loss_weight_disease_min: float = 0.3
    loss_weight_disease_max: float = 0.7
    loss_weight_pathogenicity_min: float = 0.1
    loss_weight_pathogenicity_max: float = 0.3
    loss_weight_risk_min: float = 0.05
    loss_weight_risk_max: float = 0.2


@dataclass
class AutoMLConfig:
    """Main AutoML configuration.

    Controls the overall AutoML experiment including optimizer settings,
    search space, ensemble strategy, and resource allocation.
    """
    # Experiment identification
    experiment_name: str = "rna_automl"
    description: str = ""

    # Optimizer settings
    optimizer_type: OptimizerType = OptimizerType.OPTUNA
    n_trials: int = 100
    timeout_hours: Optional[float] = 24.0
    n_jobs: int = 1

    # Optuna-specific settings
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.HYPERBAND
    seed: int = 42

    # Search space
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)

    # Objective
    objective_metric: str = "f1_macro"
    direction: Literal["minimize", "maximize"] = "maximize"

    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.HYBRID
    ensemble_top_k: int = 5

    # Resource management
    use_gpu: bool = True
    gpu_per_trial: float = 1.0
    cpu_per_trial: int = 2
    max_concurrent_trials: int = 1

    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    max_seq_length: int = 512

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 5
    save_every_n_trials: int = 10

    # Storage backend
    storage_backend: Literal["sqlite", "postgresql", "memory"] = "sqlite"
    storage_url: Optional[str] = None

    def get_storage_url(self) -> str:
        """Get database URL for Optuna storage."""
        if self.storage_url:
            return self.storage_url

        if self.storage_backend == "sqlite":
            return f"sqlite:///{self.checkpoint_dir}/optuna.db"
        elif self.storage_backend == "postgresql":
            return "postgresql://localhost/automl"
        else:
            return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "optimizer_type": self.optimizer_type.value,
            "n_trials": self.n_trials,
            "timeout_hours": self.timeout_hours,
            "sampler": self.sampler.value,
            "pruner": self.pruner.value,
            "objective_metric": self.objective_metric,
            "direction": self.direction,
            "enable_ensemble": self.enable_ensemble,
            "ensemble_strategy": self.ensemble_strategy.value,
            "ensemble_top_k": self.ensemble_top_k,
            "use_gpu": self.use_gpu,
        }


@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    # Encoder
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    n_gram_sizes: Tuple[int, ...] = (3, 5)
    vocab_size: int = 8192

    # Classifier
    classifier_dropout: float = 0.2
    classifier_hidden_1: int = 256
    classifier_hidden_2: int = 128

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 50
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5

    # Loss weights
    rna_type_weight: float = 0.2
    disease_weight: float = 0.5
    pathogenicity_weight: float = 0.2
    risk_weight: float = 0.1

    def to_encoder_config(self) -> Dict[str, Any]:
        """Convert to encoder config dict."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "n_gram_sizes": self.n_gram_sizes,
        }

    def to_classifier_config(self) -> Dict[str, Any]:
        """Convert to classifier config dict."""
        return {
            "hidden_size": self.hidden_size,
            "classifier_dropout": self.classifier_dropout,
            "hidden_dims": (self.classifier_hidden_1, self.classifier_hidden_2),
        }

    def to_training_config(self) -> Dict[str, Any]:
        """Convert to training config dict."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.gradient_clip_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "rna_type_weight": self.rna_type_weight,
            "disease_weight": self.disease_weight,
            "pathogenicity_weight": self.pathogenicity_weight,
            "risk_weight": self.risk_weight,
        }
