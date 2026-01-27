"""AutoML API Schemas."""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field
from enum import Enum


class OptimizerType(str, Enum):
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"


class SamplerType(str, Enum):
    TPE = "tpe"
    CMA_ES = "cma_es"
    RANDOM = "random"


class PrunerType(str, Enum):
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    NONE = "none"


class EnsembleStrategy(str, Enum):
    VOTING = "voting"
    STACKING = "stacking"
    HYBRID = "hybrid"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ==================== Request Schemas ====================


class SearchSpaceOverride(BaseModel):
    """Override default search space parameters."""
    learning_rate_min: Optional[float] = Field(None, ge=1e-8, le=1e-2)
    learning_rate_max: Optional[float] = Field(None, ge=1e-6, le=1.0)
    batch_size_choices: Optional[List[int]] = None
    hidden_size_choices: Optional[List[int]] = None
    num_layers_range: Optional[tuple] = None
    dropout_range: Optional[tuple] = None


class AutoMLConfigRequest(BaseModel):
    """Request to create AutoML experiment."""
    experiment_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

    # Optimizer settings
    optimizer_type: OptimizerType = OptimizerType.OPTUNA
    n_trials: int = Field(default=100, ge=1, le=10000)
    timeout_hours: Optional[float] = Field(default=24.0, ge=0.1, le=720.0)

    # Search configuration
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.HYPERBAND
    search_space_override: Optional[SearchSpaceOverride] = None

    # Objective
    objective_metric: str = Field(default="f1_macro")
    direction: Literal["minimize", "maximize"] = "maximize"

    # Resource allocation
    use_gpu: bool = True
    n_parallel_trials: int = Field(default=1, ge=1, le=8)

    # Dataset
    dataset_id: Optional[str] = None
    train_split: float = Field(default=0.8, ge=0.5, le=0.95)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "experiment_name": "rna_disease_v1",
                    "description": "Initial RNA disease prediction optimization",
                    "optimizer_type": "optuna",
                    "n_trials": 100,
                    "objective_metric": "f1_macro",
                    "use_gpu": True,
                }
            ]
        }
    }


class EnsembleConfigRequest(BaseModel):
    """Request to create ensemble from experiment."""
    strategy: EnsembleStrategy = EnsembleStrategy.HYBRID
    top_k: int = Field(default=5, ge=2, le=20)
    include_ml_models: bool = True
    ml_model_types: List[str] = Field(default=["xgboost", "random_forest"])
    voting_weights: Optional[List[float]] = None


# ==================== Response Schemas ====================


class TrialMetrics(BaseModel):
    """Metrics from a trial."""
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None


class TrialParams(BaseModel):
    """Hyperparameters used in a trial."""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    dropout: Optional[float] = None


class TrialSummary(BaseModel):
    """Summary of a single trial."""
    trial_id: int
    status: str
    value: Optional[float] = None
    params: Dict[str, Any]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class TrialDetailResponse(BaseModel):
    """Detailed trial information."""
    trial_id: int
    experiment_id: str
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    training_history: Optional[List[Dict[str, float]]] = None
    model_path: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrialListResponse(BaseModel):
    """List of trials."""
    trials: List[TrialSummary]
    total: int
    best_trial_id: Optional[int] = None


class ExperimentSummary(BaseModel):
    """Summary of an experiment."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    n_trials_completed: int
    n_trials_total: int
    best_value: Optional[float] = None
    created_at: Optional[datetime] = None


class AutoMLExperimentResponse(BaseModel):
    """Response after creating experiment."""
    experiment_id: str
    name: str
    status: str
    created_at: datetime
    config: AutoMLConfigRequest


class ExperimentStatusResponse(BaseModel):
    """Detailed experiment status."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    progress_percent: float
    n_trials_completed: int
    n_trials_total: int
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    best_trial_id: Optional[int] = None
    best_metric_value: Optional[float] = None
    elapsed_time_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    current_trial_id: Optional[int] = None


class ModelInfo(BaseModel):
    """Information about a trained model."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    created_at: Optional[datetime] = None


class ModelComparisonResponse(BaseModel):
    """Comparison of multiple models."""
    experiment_id: str
    models: List[ModelInfo]
    comparison_metrics: Dict[str, List[float]]
    recommendation: Optional[str] = None


class EnsembleResponse(BaseModel):
    """Ensemble creation response."""
    ensemble_id: str
    experiment_id: str
    strategy: EnsembleStrategy
    n_models: int
    model_weights: List[float]
    metrics: Dict[str, float]
    created_at: datetime


class OptimizationHistory(BaseModel):
    """Optimization history for visualization."""
    trial_ids: List[int]
    values: List[float]
    best_values: List[float]


class ParamImportance(BaseModel):
    """Hyperparameter importance."""
    importances: Dict[str, float]
    method: str = "fanova"
