"""AutoML Module for RNA Disease Prediction.

Provides automated hyperparameter optimization, model selection,
and ensemble creation using Optuna and custom training pipelines.
"""

from .config import AutoMLConfig, SearchSpaceConfig
from .trainer import AutoMLTrainer, TrialResult

__all__ = [
    "AutoMLConfig",
    "SearchSpaceConfig",
    "AutoMLTrainer",
    "TrialResult",
]
