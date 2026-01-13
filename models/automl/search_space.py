"""Hyperparameter Search Space Definitions."""

from typing import Dict, Any
from .config import SearchSpaceConfig, TrialConfig

try:
    import optuna
except ImportError:
    optuna = None


class RNAModelSearchSpace:
    """Search space for RNA prediction model hyperparameters.

    Provides methods to sample hyperparameters from defined search spaces
    using Optuna's trial suggestions.
    """

    @staticmethod
    def sample_encoder_params(
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> Dict[str, Any]:
        """Sample encoder hyperparameters.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Dictionary of encoder parameters
        """
        hidden_size = trial.suggest_categorical(
            "hidden_size", config.hidden_size_choices
        )

        # Ensure num_attention_heads divides hidden_size
        valid_heads = [h for h in config.num_attention_heads_choices if hidden_size % h == 0]
        if not valid_heads:
            valid_heads = [8]  # Default fallback

        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": trial.suggest_int(
                "num_hidden_layers",
                config.num_layers_min,
                config.num_layers_max,
            ),
            "num_attention_heads": trial.suggest_categorical(
                "num_attention_heads", valid_heads
            ),
            "intermediate_size": trial.suggest_categorical(
                "intermediate_size", config.intermediate_size_choices
            ),
            "hidden_dropout_prob": trial.suggest_float(
                "hidden_dropout_prob",
                config.dropout_min,
                config.dropout_max,
            ),
            "attention_probs_dropout_prob": trial.suggest_float(
                "attention_dropout_prob",
                config.dropout_min,
                config.dropout_max * 0.6,  # Lower range for attention
            ),
        }

    @staticmethod
    def sample_classifier_params(
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> Dict[str, Any]:
        """Sample classifier hyperparameters.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Dictionary of classifier parameters
        """
        return {
            "classifier_dropout": trial.suggest_float(
                "classifier_dropout",
                config.dropout_min + 0.1,  # Classifier usually needs more dropout
                config.dropout_max,
            ),
            "classifier_hidden_1": trial.suggest_categorical(
                "classifier_hidden_1", config.classifier_hidden_1_choices
            ),
            "classifier_hidden_2": trial.suggest_categorical(
                "classifier_hidden_2", config.classifier_hidden_2_choices
            ),
        }

    @staticmethod
    def sample_tokenizer_params(
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> Dict[str, Any]:
        """Sample tokenizer hyperparameters.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Dictionary of tokenizer parameters
        """
        # Convert tuples to strings for Optuna categorical
        ngram_options = [str(opt) for opt in config.ngram_sizes_options]
        selected = trial.suggest_categorical("n_gram_sizes", ngram_options)
        # Parse back to tuple
        n_gram_sizes = eval(selected)

        return {
            "n_gram_sizes": n_gram_sizes,
            "vocab_size": trial.suggest_categorical(
                "vocab_size", config.vocab_size_choices
            ),
        }

    @staticmethod
    def sample_training_params(
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> Dict[str, Any]:
        """Sample training hyperparameters.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Dictionary of training parameters
        """
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                config.learning_rate_min,
                config.learning_rate_max,
                log=config.learning_rate_log,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", config.batch_size_choices
            ),
            "epochs": trial.suggest_int(
                "epochs",
                config.epochs_min,
                config.epochs_max,
            ),
            "early_stopping_patience": trial.suggest_int(
                "early_stopping_patience",
                config.early_stopping_patience_min,
                config.early_stopping_patience_max,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                config.weight_decay_min,
                config.weight_decay_max,
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio",
                config.warmup_ratio_min,
                config.warmup_ratio_max,
            ),
            "gradient_clip_norm": trial.suggest_float(
                "gradient_clip_norm",
                config.gradient_clip_min,
                config.gradient_clip_max,
            ),
        }

    @staticmethod
    def sample_loss_weights(
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> Dict[str, Any]:
        """Sample multi-task loss weights.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Dictionary of loss weights (normalized to sum to 1)
        """
        rna_type_weight = trial.suggest_float(
            "rna_type_weight",
            config.loss_weight_rna_type_min,
            config.loss_weight_rna_type_max,
        )
        disease_weight = trial.suggest_float(
            "disease_weight",
            config.loss_weight_disease_min,
            config.loss_weight_disease_max,
        )
        pathogenicity_weight = trial.suggest_float(
            "pathogenicity_weight",
            config.loss_weight_pathogenicity_min,
            config.loss_weight_pathogenicity_max,
        )
        risk_weight = trial.suggest_float(
            "risk_weight",
            config.loss_weight_risk_min,
            config.loss_weight_risk_max,
        )

        # Normalize weights to sum to 1
        total = rna_type_weight + disease_weight + pathogenicity_weight + risk_weight
        return {
            "rna_type_weight": rna_type_weight / total,
            "disease_weight": disease_weight / total,
            "pathogenicity_weight": pathogenicity_weight / total,
            "risk_weight": risk_weight / total,
        }

    @staticmethod
    def sample_ensemble_ml_params(
        trial: "optuna.Trial",
    ) -> Dict[str, Any]:
        """Sample parameters for traditional ML models in ensemble.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of XGBoost and Random Forest parameters
        """
        return {
            # XGBoost parameters
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "xgb_learning_rate": trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.3, log=True
            ),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            # Random Forest parameters
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 5, 20),
            "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
        }

    @classmethod
    def sample_all(
        cls,
        trial: "optuna.Trial",
        config: SearchSpaceConfig,
    ) -> TrialConfig:
        """Sample all hyperparameters for a complete trial.

        Args:
            trial: Optuna trial object
            config: Search space configuration

        Returns:
            Complete TrialConfig with sampled parameters
        """
        encoder_params = cls.sample_encoder_params(trial, config)
        classifier_params = cls.sample_classifier_params(trial, config)
        tokenizer_params = cls.sample_tokenizer_params(trial, config)
        training_params = cls.sample_training_params(trial, config)
        loss_weights = cls.sample_loss_weights(trial, config)

        return TrialConfig(
            # Encoder
            hidden_size=encoder_params["hidden_size"],
            num_hidden_layers=encoder_params["num_hidden_layers"],
            num_attention_heads=encoder_params["num_attention_heads"],
            intermediate_size=encoder_params["intermediate_size"],
            hidden_dropout_prob=encoder_params["hidden_dropout_prob"],
            attention_probs_dropout_prob=encoder_params["attention_probs_dropout_prob"],
            n_gram_sizes=tokenizer_params["n_gram_sizes"],
            vocab_size=tokenizer_params["vocab_size"],
            # Classifier
            classifier_dropout=classifier_params["classifier_dropout"],
            classifier_hidden_1=classifier_params["classifier_hidden_1"],
            classifier_hidden_2=classifier_params["classifier_hidden_2"],
            # Training
            learning_rate=training_params["learning_rate"],
            batch_size=training_params["batch_size"],
            epochs=training_params["epochs"],
            weight_decay=training_params["weight_decay"],
            warmup_ratio=training_params["warmup_ratio"],
            gradient_clip_norm=training_params["gradient_clip_norm"],
            early_stopping_patience=training_params["early_stopping_patience"],
            # Loss weights
            rna_type_weight=loss_weights["rna_type_weight"],
            disease_weight=loss_weights["disease_weight"],
            pathogenicity_weight=loss_weights["pathogenicity_weight"],
            risk_weight=loss_weights["risk_weight"],
        )


def get_default_search_space() -> SearchSpaceConfig:
    """Get default search space configuration."""
    return SearchSpaceConfig()


def get_small_search_space() -> SearchSpaceConfig:
    """Get smaller search space for quick experiments."""
    return SearchSpaceConfig(
        hidden_size_choices=[256, 512],
        num_layers_min=4,
        num_layers_max=8,
        num_attention_heads_choices=[4, 8],
        intermediate_size_choices=[512, 1024],
        batch_size_choices=[16, 32],
        epochs_min=10,
        epochs_max=30,
        ngram_sizes_options=[(3, 5), (1, 3, 5)],
        vocab_size_choices=[4096, 8192],
    )


def get_large_search_space() -> SearchSpaceConfig:
    """Get larger search space for thorough optimization."""
    return SearchSpaceConfig(
        hidden_size_choices=[256, 384, 512, 768, 1024],
        num_layers_min=6,
        num_layers_max=24,
        num_attention_heads_choices=[4, 8, 12, 16, 20],
        intermediate_size_choices=[1024, 2048, 3072, 4096],
        batch_size_choices=[8, 16, 32, 64, 128],
        epochs_min=20,
        epochs_max=200,
        ngram_sizes_options=[(3, 5), (1, 3, 5), (3, 4, 5, 6), (3, 5, 7), (1, 3, 5, 7)],
        vocab_size_choices=[4096, 8192, 16384, 32768],
    )
