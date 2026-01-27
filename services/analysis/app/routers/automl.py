"""AutoML API Router."""

from typing import List, Optional
from datetime import datetime
from uuid import uuid4
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.schemas.automl import (
    AutoMLConfigRequest,
    AutoMLExperimentResponse,
    ExperimentStatusResponse,
    ExperimentStatus,
    TrialListResponse,
    TrialSummary,
    TrialDetailResponse,
    EnsembleConfigRequest,
    EnsembleResponse,
    ModelComparisonResponse,
    ModelInfo,
    OptimizationHistory,
    ParamImportance,
    ExperimentSummary,
)

router = APIRouter(prefix="/api/v1/automl", tags=["AutoML"])

# Add models path to sys.path for imports
def _find_models_path():
    """Find models directory in various environments."""
    current = Path(__file__).resolve()

    # Try fixed Docker path first
    docker_path = Path("/app/models")
    if docker_path.exists():
        return docker_path

    # Try parent directories (up to project root)
    for i in range(len(current.parents)):
        candidate = current.parents[i] / "models"
        if candidate.exists():
            return candidate

    # Default fallback
    return docker_path

MODELS_PATH = _find_models_path()
if str(MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(MODELS_PATH))

# In-memory storage for experiments (would use database in production)
_experiments = {}
_trials = {}


def get_experiment_or_404(experiment_id: str):
    """Get experiment by ID or raise 404."""
    if experiment_id not in _experiments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    return _experiments[experiment_id]


# ==================== Experiment Management ====================


@router.post("/experiments", response_model=AutoMLExperimentResponse)
async def create_experiment(
    config: AutoMLConfigRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create and start a new AutoML experiment.

    This initiates hyperparameter optimization for the RNA disease
    prediction model. The experiment runs asynchronously and can be
    monitored via the status endpoint.
    """
    experiment_id = str(uuid4())

    experiment = {
        "experiment_id": experiment_id,
        "name": config.experiment_name,
        "description": config.description,
        "config": config.model_dump(),
        "status": ExperimentStatus.PENDING,
        "n_trials_completed": 0,
        "n_trials_total": config.n_trials,
        "n_trials_pruned": 0,
        "n_trials_failed": 0,
        "best_trial_id": None,
        "best_metric_value": None,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "elapsed_time_seconds": 0,
    }

    _experiments[experiment_id] = experiment
    _trials[experiment_id] = []

    # Queue background task (placeholder - would use Celery in production)
    background_tasks.add_task(run_experiment_async, experiment_id, config)

    return AutoMLExperimentResponse(
        experiment_id=experiment_id,
        name=config.experiment_name,
        status="pending",
        created_at=experiment["created_at"],
        config=config,
    )


async def run_experiment_async(experiment_id: str, config: AutoMLConfigRequest):
    """Run AutoML experiment asynchronously."""
    import asyncio

    experiment = _experiments.get(experiment_id)
    if not experiment:
        return

    experiment["status"] = ExperimentStatus.RUNNING
    experiment["started_at"] = datetime.utcnow()

    # Simulate trials (placeholder for actual AutoML training)
    import random

    for i in range(min(config.n_trials, 10)):  # Limit to 10 for demo
        await asyncio.sleep(1)  # Simulate training time

        trial = {
            "trial_id": i,
            "status": "completed",
            "value": random.uniform(0.6, 0.95),
            "params": {
                "learning_rate": random.uniform(1e-5, 1e-3),
                "batch_size": random.choice([16, 32, 64]),
                "hidden_size": random.choice([256, 512, 768]),
                "num_layers": random.randint(4, 12),
            },
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
        }

        _trials[experiment_id].append(trial)
        experiment["n_trials_completed"] = len(_trials[experiment_id])

        # Update best trial
        if experiment["best_metric_value"] is None or trial["value"] > experiment["best_metric_value"]:
            experiment["best_metric_value"] = trial["value"]
            experiment["best_trial_id"] = trial["trial_id"]

    experiment["status"] = ExperimentStatus.COMPLETED
    experiment["completed_at"] = datetime.utcnow()
    experiment["elapsed_time_seconds"] = (
        experiment["completed_at"] - experiment["started_at"]
    ).total_seconds()


@router.get("/experiments", response_model=List[ExperimentSummary])
async def list_experiments(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
):
    """List all AutoML experiments."""
    experiments = list(_experiments.values())

    if status_filter:
        experiments = [e for e in experiments if e["status"].value == status_filter]

    # Sort by creation time
    experiments.sort(key=lambda x: x["created_at"], reverse=True)

    # Paginate
    experiments = experiments[skip : skip + limit]

    return [
        ExperimentSummary(
            experiment_id=e["experiment_id"],
            name=e["name"],
            status=e["status"],
            n_trials_completed=e["n_trials_completed"],
            n_trials_total=e["n_trials_total"],
            best_value=e["best_metric_value"],
            created_at=e["created_at"],
        )
        for e in experiments
    ]


@router.get("/experiments/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    """
    Get detailed status of an AutoML experiment.

    Returns current progress, best trial so far, and resource usage.
    """
    experiment = get_experiment_or_404(experiment_id)

    progress = 0.0
    if experiment["n_trials_total"] > 0:
        progress = (experiment["n_trials_completed"] / experiment["n_trials_total"]) * 100

    elapsed = experiment.get("elapsed_time_seconds", 0)
    if experiment["status"] == ExperimentStatus.RUNNING and experiment.get("started_at"):
        elapsed = (datetime.utcnow() - experiment["started_at"]).total_seconds()

    estimated_remaining = None
    if experiment["n_trials_completed"] > 0 and progress > 0:
        avg_time_per_trial = elapsed / experiment["n_trials_completed"]
        remaining_trials = experiment["n_trials_total"] - experiment["n_trials_completed"]
        estimated_remaining = avg_time_per_trial * remaining_trials

    return ExperimentStatusResponse(
        experiment_id=experiment["experiment_id"],
        name=experiment["name"],
        status=experiment["status"],
        progress_percent=round(progress, 1),
        n_trials_completed=experiment["n_trials_completed"],
        n_trials_total=experiment["n_trials_total"],
        n_trials_pruned=experiment.get("n_trials_pruned", 0),
        n_trials_failed=experiment.get("n_trials_failed", 0),
        best_trial_id=experiment.get("best_trial_id"),
        best_metric_value=experiment.get("best_metric_value"),
        elapsed_time_seconds=elapsed,
        estimated_remaining_seconds=estimated_remaining,
    )


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running AutoML experiment."""
    experiment = get_experiment_or_404(experiment_id)

    if experiment["status"] != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment is not running",
        )

    experiment["status"] = ExperimentStatus.STOPPED
    experiment["completed_at"] = datetime.utcnow()

    return {"status": "stopped", "experiment_id": experiment_id}


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an AutoML experiment and its artifacts."""
    get_experiment_or_404(experiment_id)

    del _experiments[experiment_id]
    if experiment_id in _trials:
        del _trials[experiment_id]

    return {"status": "deleted", "experiment_id": experiment_id}


# ==================== Trial Management ====================


@router.get("/experiments/{experiment_id}/trials", response_model=TrialListResponse)
async def list_trials(
    experiment_id: str,
    skip: int = 0,
    limit: int = 50,
    sort_by: str = "value",
    sort_order: str = "desc",
):
    """
    List all trials for an experiment.

    Supports sorting by trial value (metric), duration, or trial number.
    """
    get_experiment_or_404(experiment_id)
    trials = _trials.get(experiment_id, [])

    # Sort
    reverse = sort_order == "desc"
    if sort_by == "value":
        trials = sorted(trials, key=lambda x: x.get("value", 0), reverse=reverse)
    elif sort_by == "trial_id":
        trials = sorted(trials, key=lambda x: x["trial_id"], reverse=reverse)

    # Find best trial
    best_trial_id = None
    if trials:
        best_trial = max(trials, key=lambda x: x.get("value", 0))
        best_trial_id = best_trial["trial_id"]

    # Paginate
    paginated = trials[skip : skip + limit]

    return TrialListResponse(
        trials=[
            TrialSummary(
                trial_id=t["trial_id"],
                status=t["status"],
                value=t.get("value"),
                params=t.get("params", {}),
                start_time=t.get("start_time"),
                end_time=t.get("end_time"),
            )
            for t in paginated
        ],
        total=len(trials),
        best_trial_id=best_trial_id,
    )


@router.get("/experiments/{experiment_id}/trials/{trial_id}", response_model=TrialDetailResponse)
async def get_trial_detail(experiment_id: str, trial_id: int):
    """
    Get detailed information about a specific trial.

    Includes all hyperparameters, metrics, and training curves.
    """
    get_experiment_or_404(experiment_id)
    trials = _trials.get(experiment_id, [])

    trial = next((t for t in trials if t["trial_id"] == trial_id), None)
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found",
        )

    return TrialDetailResponse(
        trial_id=trial["trial_id"],
        experiment_id=experiment_id,
        status=trial["status"],
        params=trial.get("params", {}),
        metrics={"objective": trial.get("value", 0)},
        training_history=None,
        model_path=None,
        created_at=trial.get("start_time"),
        completed_at=trial.get("end_time"),
    )


# ==================== Model & Ensemble ====================


@router.get("/experiments/{experiment_id}/best-model", response_model=ModelInfo)
async def get_best_model(experiment_id: str):
    """Get information about the best model from an experiment."""
    experiment = get_experiment_or_404(experiment_id)

    if experiment.get("best_trial_id") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed trials yet",
        )

    trials = _trials.get(experiment_id, [])
    best_trial = next(
        (t for t in trials if t["trial_id"] == experiment["best_trial_id"]),
        None,
    )

    if not best_trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Best trial not found",
        )

    return ModelInfo(
        trial_id=best_trial["trial_id"],
        params=best_trial.get("params", {}),
        metrics={"objective": best_trial.get("value", 0)},
        model_path=f"/models/automl/{experiment_id}/trial_{best_trial['trial_id']}/model.pt",
        created_at=best_trial.get("end_time"),
    )


@router.post("/experiments/{experiment_id}/ensemble", response_model=EnsembleResponse)
async def create_ensemble(
    experiment_id: str,
    config: EnsembleConfigRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create an ensemble from top performing trials.

    Supports voting, stacking, and hybrid ensemble strategies.
    """
    experiment = get_experiment_or_404(experiment_id)
    trials = _trials.get(experiment_id, [])

    if len(trials) < config.top_k:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough trials. Need {config.top_k}, have {len(trials)}",
        )

    # Get top K trials
    sorted_trials = sorted(trials, key=lambda x: x.get("value", 0), reverse=True)
    top_trials = sorted_trials[: config.top_k]

    ensemble_id = str(uuid4())

    return EnsembleResponse(
        ensemble_id=ensemble_id,
        experiment_id=experiment_id,
        strategy=config.strategy,
        n_models=len(top_trials),
        model_weights=[t.get("value", 1.0) for t in top_trials],
        metrics={"avg_value": sum(t.get("value", 0) for t in top_trials) / len(top_trials)},
        created_at=datetime.utcnow(),
    )


@router.get("/experiments/{experiment_id}/comparison", response_model=ModelComparisonResponse)
async def compare_models(
    experiment_id: str,
    trial_ids: Optional[str] = None,
):
    """
    Compare multiple models from an experiment.

    Returns side-by-side comparison of metrics and parameters.
    """
    get_experiment_or_404(experiment_id)
    trials = _trials.get(experiment_id, [])

    # Parse trial IDs if provided
    if trial_ids:
        ids = [int(x.strip()) for x in trial_ids.split(",")]
        trials = [t for t in trials if t["trial_id"] in ids]

    models = [
        ModelInfo(
            trial_id=t["trial_id"],
            params=t.get("params", {}),
            metrics={"objective": t.get("value", 0)},
        )
        for t in trials
    ]

    # Collect comparison metrics
    comparison = {
        "objective": [t.get("value", 0) for t in trials],
    }

    # Recommendation
    if models:
        best_idx = comparison["objective"].index(max(comparison["objective"]))
        recommendation = f"Trial {models[best_idx].trial_id}이 최고 성능을 보입니다."
    else:
        recommendation = None

    return ModelComparisonResponse(
        experiment_id=experiment_id,
        models=models,
        comparison_metrics=comparison,
        recommendation=recommendation,
    )


# ==================== Analysis & Visualization ====================


@router.get("/experiments/{experiment_id}/optimization-history", response_model=OptimizationHistory)
async def get_optimization_history(experiment_id: str):
    """
    Get optimization history for visualization.

    Returns trial values over time for plotting optimization curves.
    """
    get_experiment_or_404(experiment_id)
    trials = _trials.get(experiment_id, [])

    trial_ids = []
    values = []
    best_values = []
    current_best = float("-inf")

    for t in sorted(trials, key=lambda x: x["trial_id"]):
        trial_ids.append(t["trial_id"])
        value = t.get("value", 0)
        values.append(value)
        current_best = max(current_best, value)
        best_values.append(current_best)

    return OptimizationHistory(
        trial_ids=trial_ids,
        values=values,
        best_values=best_values,
    )


@router.get("/experiments/{experiment_id}/param-importance", response_model=ParamImportance)
async def get_param_importance(experiment_id: str):
    """
    Get hyperparameter importance analysis.

    Shows which hyperparameters have the most impact on model performance.
    """
    get_experiment_or_404(experiment_id)

    # Mock importance values
    importances = {
        "learning_rate": 0.35,
        "hidden_size": 0.25,
        "num_layers": 0.20,
        "batch_size": 0.12,
        "dropout": 0.08,
    }

    return ParamImportance(
        importances=importances,
        method="fanova",
    )
