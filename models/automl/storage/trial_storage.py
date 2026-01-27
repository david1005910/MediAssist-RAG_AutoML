"""Trial Storage for AutoML experiments."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrialStorage:
    """Storage for AutoML trial results and experiment metadata.

    Provides persistence for trial results, experiment configurations,
    and optimization history.
    """

    def __init__(self, base_dir: str = "./checkpoints"):
        """Initialize trial storage.

        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.trials_dir = self.base_dir / "trials"
        self.trials_dir.mkdir(exist_ok=True)

        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

    def save_trial_result(self, result: "TrialResult") -> None:
        """Save trial result to disk.

        Args:
            result: Trial result to save
        """
        trial_file = self.trials_dir / f"trial_{result.trial_id}.json"

        data = {
            "trial_id": result.trial_id,
            "params": result.params,
            "metrics": result.metrics,
            "model_path": result.model_path,
            "training_time_seconds": result.training_time_seconds,
            "status": result.status,
            "error": result.error,
            "saved_at": datetime.now().isoformat(),
        }

        with open(trial_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved trial {result.trial_id} to {trial_file}")

    def load_trial_result(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Load trial result from disk.

        Args:
            trial_id: Trial ID to load

        Returns:
            Trial result dictionary or None if not found
        """
        trial_file = self.trials_dir / f"trial_{trial_id}.json"

        if not trial_file.exists():
            return None

        with open(trial_file, "r") as f:
            return json.load(f)

    def list_trials(self) -> List[Dict[str, Any]]:
        """List all saved trials.

        Returns:
            List of trial result dictionaries
        """
        trials = []
        for trial_file in self.trials_dir.glob("trial_*.json"):
            with open(trial_file, "r") as f:
                trials.append(json.load(f))

        return sorted(trials, key=lambda x: x["trial_id"])

    def get_top_trials(
        self,
        k: int = 5,
        metric: str = "objective",
        direction: str = "maximize",
    ) -> List[Dict[str, Any]]:
        """Get top K trials by metric.

        Args:
            k: Number of trials to return
            metric: Metric to sort by
            direction: 'maximize' or 'minimize'

        Returns:
            List of top K trial results
        """
        trials = self.list_trials()

        # Filter completed trials with the metric
        completed = [
            t for t in trials
            if t["status"] == "completed" and metric in t.get("metrics", {})
        ]

        # Sort by metric
        reverse = direction == "maximize"
        sorted_trials = sorted(
            completed,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=reverse,
        )

        return sorted_trials[:k]

    def save_experiment_result(self, result: "AutoMLResult") -> None:
        """Save complete experiment result.

        Args:
            result: AutoML result to save
        """
        exp_file = self.experiments_dir / f"{result.experiment_name}.json"

        data = {
            "experiment_name": result.experiment_name,
            "best_trial": {
                "trial_id": result.best_trial.trial_id,
                "params": result.best_trial.params,
                "metrics": result.best_trial.metrics,
            },
            "study_stats": result.study_stats,
            "ensemble_result": result.ensemble_result,
            "total_time_seconds": result.total_time_seconds,
            "n_trials": len(result.all_trials),
            "saved_at": datetime.now().isoformat(),
        }

        with open(exp_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved experiment result to {exp_file}")

    def load_experiment_result(
        self, experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load experiment result.

        Args:
            experiment_name: Name of experiment

        Returns:
            Experiment result dictionary or None
        """
        exp_file = self.experiments_dir / f"{experiment_name}.json"

        if not exp_file.exists():
            return None

        with open(exp_file, "r") as f:
            return json.load(f)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiments.

        Returns:
            List of experiment summaries
        """
        experiments = []
        for exp_file in self.experiments_dir.glob("*.json"):
            with open(exp_file, "r") as f:
                data = json.load(f)
                experiments.append({
                    "experiment_name": data["experiment_name"],
                    "n_trials": data.get("n_trials", 0),
                    "best_value": data.get("study_stats", {}).get("best_value"),
                    "saved_at": data.get("saved_at"),
                })

        return experiments

    def update_experiment_status(
        self,
        experiment_name: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update experiment status.

        Args:
            experiment_name: Experiment name
            status: New status
            error: Optional error message
        """
        exp_file = self.experiments_dir / f"{experiment_name}.json"

        if exp_file.exists():
            with open(exp_file, "r") as f:
                data = json.load(f)
        else:
            data = {"experiment_name": experiment_name}

        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        if error:
            data["error"] = error

        with open(exp_file, "w") as f:
            json.dump(data, f, indent=2)

    def clear_experiment(self, experiment_name: str) -> None:
        """Clear all data for an experiment.

        Args:
            experiment_name: Experiment name to clear
        """
        # Remove experiment file
        exp_file = self.experiments_dir / f"{experiment_name}.json"
        if exp_file.exists():
            exp_file.unlink()

        # Remove associated trial files
        for trial_file in self.trials_dir.glob("trial_*.json"):
            trial_file.unlink()

        logger.info(f"Cleared experiment: {experiment_name}")
