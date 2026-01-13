"""Optuna-based Hyperparameter Optimizer."""

from typing import Dict, Any, Optional, Callable
import logging

try:
    import optuna
    from optuna.trial import Trial, TrialState
    from optuna.study import Study
except ImportError:
    optuna = None
    Trial = None
    TrialState = None
    Study = None

from ..config import AutoMLConfig, SamplerType, PrunerType

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Wrapper for Optuna hyperparameter optimization.

    Provides a simplified interface for creating and running Optuna studies
    with configurable samplers, pruners, and storage backends.
    """

    def __init__(self, config: AutoMLConfig):
        """Initialize optimizer.

        Args:
            config: AutoML configuration
        """
        if optuna is None:
            raise ImportError("optuna is required. Install with: pip install optuna")

        self.config = config
        self.study: Optional[Study] = None

    def _create_sampler(self) -> "optuna.samplers.BaseSampler":
        """Create sampler based on configuration."""
        seed = self.config.seed

        samplers = {
            SamplerType.TPE: lambda: optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=10,
                multivariate=True,
            ),
            SamplerType.CMA_ES: lambda: optuna.samplers.CmaEsSampler(seed=seed),
            SamplerType.RANDOM: lambda: optuna.samplers.RandomSampler(seed=seed),
            SamplerType.GRID: lambda: optuna.samplers.GridSampler({}),
        }

        return samplers.get(self.config.sampler, samplers[SamplerType.TPE])()

    def _create_pruner(self) -> "optuna.pruners.BasePruner":
        """Create pruner based on configuration."""
        pruners = {
            PrunerType.HYPERBAND: lambda: optuna.pruners.HyperbandPruner(
                min_resource=5,
                max_resource=self.config.search_space.epochs_max,
                reduction_factor=3,
            ),
            PrunerType.MEDIAN: lambda: optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
            PrunerType.SUCCESSIVE_HALVING: lambda: optuna.pruners.SuccessiveHalvingPruner(
                min_resource=5,
                reduction_factor=2,
            ),
            PrunerType.NONE: lambda: optuna.pruners.NopPruner(),
        }

        return pruners.get(self.config.pruner, pruners[PrunerType.HYPERBAND])()

    def create_study(self) -> Study:
        """Create or load Optuna study.

        Returns:
            Optuna study instance
        """
        storage_url = self.config.get_storage_url()

        self.study = optuna.create_study(
            study_name=self.config.experiment_name,
            direction=self.config.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            storage=storage_url if storage_url else None,
            load_if_exists=True,
        )

        return self.study

    def optimize(
        self,
        objective: Callable[[Trial], float],
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        callbacks: Optional[list] = None,
    ) -> Study:
        """Run optimization.

        Args:
            objective: Objective function to optimize
            n_trials: Number of trials (default from config)
            timeout: Timeout in seconds (default from config)
            n_jobs: Number of parallel jobs
            show_progress_bar: Show progress bar
            callbacks: Optional callbacks

        Returns:
            Completed study
        """
        if self.study is None:
            self.create_study()

        n_trials = n_trials or self.config.n_trials
        timeout = timeout or (self.config.timeout_hours * 3600 if self.config.timeout_hours else None)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
            catch=(Exception,),
        )

        return self.study

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """Get best trial from study.

        Returns:
            Best trial information or None
        """
        if self.study is None:
            return None

        best = self.study.best_trial
        return {
            "trial_id": best.number,
            "value": best.value,
            "params": best.params,
            "datetime_start": best.datetime_start,
            "datetime_complete": best.datetime_complete,
        }

    def get_trials(self, states: Optional[list] = None) -> list:
        """Get trials from study.

        Args:
            states: Filter by trial states

        Returns:
            List of trial dictionaries
        """
        if self.study is None:
            return []

        trials = []
        for trial in self.study.trials:
            if states and trial.state not in states:
                continue
            trials.append({
                "trial_id": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
            })

        return trials

    def get_param_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance.

        Returns:
            Dictionary of parameter importances
        """
        if self.study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Could not compute importance: {e}")
            return {}

    def get_optimization_history(self) -> Dict[str, list]:
        """Get optimization history for visualization.

        Returns:
            Dictionary with trial values and best values
        """
        if self.study is None:
            return {"values": [], "best_values": [], "trial_ids": []}

        values = []
        best_values = []
        trial_ids = []

        current_best = float("-inf") if self.config.direction == "maximize" else float("inf")

        for trial in self.study.trials:
            if trial.state == TrialState.COMPLETE:
                trial_ids.append(trial.number)
                values.append(trial.value)

                if self.config.direction == "maximize":
                    current_best = max(current_best, trial.value)
                else:
                    current_best = min(current_best, trial.value)
                best_values.append(current_best)

        return {
            "trial_ids": trial_ids,
            "values": values,
            "best_values": best_values,
        }

    def get_study_stats(self) -> Dict[str, Any]:
        """Get study statistics.

        Returns:
            Dictionary of statistics
        """
        if self.study is None:
            return {}

        completed = len([t for t in self.study.trials if t.state == TrialState.COMPLETE])
        pruned = len([t for t in self.study.trials if t.state == TrialState.PRUNED])
        failed = len([t for t in self.study.trials if t.state == TrialState.FAIL])
        running = len([t for t in self.study.trials if t.state == TrialState.RUNNING])

        return {
            "n_trials": len(self.study.trials),
            "n_completed": completed,
            "n_pruned": pruned,
            "n_failed": failed,
            "n_running": running,
            "best_value": self.study.best_value if completed > 0 else None,
            "best_trial_number": self.study.best_trial.number if completed > 0 else None,
        }

    @staticmethod
    def visualize_optimization_history(study: Study) -> Any:
        """Create optimization history visualization.

        Args:
            study: Optuna study

        Returns:
            Plotly figure
        """
        return optuna.visualization.plot_optimization_history(study)

    @staticmethod
    def visualize_param_importance(study: Study) -> Any:
        """Create parameter importance visualization.

        Args:
            study: Optuna study

        Returns:
            Plotly figure
        """
        return optuna.visualization.plot_param_importances(study)

    @staticmethod
    def visualize_parallel_coordinate(study: Study) -> Any:
        """Create parallel coordinate visualization.

        Args:
            study: Optuna study

        Returns:
            Plotly figure
        """
        return optuna.visualization.plot_parallel_coordinate(study)
