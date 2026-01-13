"""AutoML database models."""

from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from common.database import Base


class AutoMLExperiment(Base):
    """AutoML experiment model."""

    __tablename__ = "automl_experiments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    # Configuration
    optimizer_type: Mapped[str] = mapped_column(String(50), default="optuna")
    sampler: Mapped[str] = mapped_column(String(50), default="tpe")
    pruner: Mapped[str] = mapped_column(String(50), default="hyperband")
    n_trials: Mapped[int] = mapped_column(Integer, default=100)
    timeout_hours: Mapped[float | None] = mapped_column(Float)
    objective_metric: Mapped[str] = mapped_column(String(50), default="f1_macro")
    direction: Mapped[str] = mapped_column(String(20), default="maximize")
    search_space: Mapped[dict | None] = mapped_column(JSONB)

    # Status tracking
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed, stopped
    current_trial: Mapped[int] = mapped_column(Integer, default=0)
    best_trial_id: Mapped[int | None] = mapped_column(Integer)
    best_value: Mapped[float | None] = mapped_column(Float)

    # Resource settings
    use_gpu: Mapped[bool] = mapped_column(Boolean, default=True)
    n_parallel_trials: Mapped[int] = mapped_column(Integer, default=1)

    # Dataset reference
    dataset_id: Mapped[str | None] = mapped_column(String(200))
    train_split: Mapped[float] = mapped_column(Float, default=0.8)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    trials: Mapped[list["AutoMLTrial"]] = relationship(
        "AutoMLTrial", back_populates="experiment", cascade="all, delete-orphan"
    )
    ensembles: Mapped[list["AutoMLEnsemble"]] = relationship(
        "AutoMLEnsemble", back_populates="experiment", cascade="all, delete-orphan"
    )


class AutoMLTrial(Base):
    """AutoML trial model."""

    __tablename__ = "automl_trials"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("automl_experiments.id", ondelete="CASCADE"), nullable=False
    )
    trial_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Hyperparameters
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Results
    value: Mapped[float | None] = mapped_column(Float)  # Objective metric value
    metrics: Mapped[dict | None] = mapped_column(JSONB)  # All evaluation metrics

    # Status
    state: Mapped[str] = mapped_column(String(20), default="running")  # running, complete, pruned, fail

    # Training details
    n_epochs_completed: Mapped[int | None] = mapped_column(Integer)
    training_time_seconds: Mapped[float | None] = mapped_column(Float)

    # Model artifact
    model_path: Mapped[str | None] = mapped_column(String(500))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationship
    experiment: Mapped["AutoMLExperiment"] = relationship("AutoMLExperiment", back_populates="trials")


class AutoMLEnsemble(Base):
    """AutoML ensemble model."""

    __tablename__ = "automl_ensembles"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("automl_experiments.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Ensemble configuration
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)  # voting, stacking, hybrid
    top_k: Mapped[int] = mapped_column(Integer, default=5)

    # Model weights (trial_id -> weight)
    model_weights: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Meta-learner for stacking
    meta_learner_type: Mapped[str | None] = mapped_column(String(50))
    meta_learner_params: Mapped[dict | None] = mapped_column(JSONB)

    # Performance metrics
    metrics: Mapped[dict | None] = mapped_column(JSONB)

    # Model artifact
    model_path: Mapped[str | None] = mapped_column(String(500))

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, building, ready, failed

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationship
    experiment: Mapped["AutoMLExperiment"] = relationship("AutoMLExperiment", back_populates="ensembles")


class RNAAnalysisResult(Base):
    """RNA analysis result model."""

    __tablename__ = "rna_analysis_results"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID | None] = mapped_column(ForeignKey("analyses.id", ondelete="SET NULL"))
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    # Input
    sequence: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    input_rna_type: Mapped[str | None] = mapped_column(String(20))

    # Sequence analysis
    sequence_length: Mapped[int] = mapped_column(Integer, nullable=False)
    gc_content: Mapped[float] = mapped_column(Float, nullable=False)
    detected_rna_type: Mapped[str] = mapped_column(String(20), nullable=False)
    rna_type_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    motifs_found: Mapped[dict | None] = mapped_column(JSONB)

    # Disease predictions
    disease_predictions: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Risk assessment
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    pathogenicity: Mapped[str] = mapped_column(String(50), nullable=False)
    pathogenicity_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Model info
    model_version: Mapped[str | None] = mapped_column(String(50))
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
