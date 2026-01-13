"""Tests for AutoML module."""

import pytest
import tempfile
from pathlib import Path

from models.automl.config import (
    OptimizerType,
    SamplerType,
    PrunerType,
    EnsembleStrategy,
    SearchSpaceConfig,
    AutoMLConfig,
    TrialConfig,
)
from models.automl.search_space import RNAModelSearchSpace
from models.automl.trainer import TrialResult, AutoMLResult
from models.automl.ensemble.voting_ensemble import VotingEnsemble
from models.automl.ensemble.hybrid_ensemble import HybridEnsemble
from models.automl.storage.trial_storage import TrialStorage
from models.automl.storage.model_registry import ModelRegistry
import torch
import torch.nn as nn


class TestOptimizerTypes:
    """Test cases for optimizer type enums."""

    def test_optimizer_types(self):
        """Test optimizer type enum values."""
        assert OptimizerType.OPTUNA.value == "optuna"

    def test_sampler_types(self):
        """Test sampler type enum values."""
        assert SamplerType.TPE.value == "tpe"
        assert SamplerType.CMA_ES.value == "cma_es"
        assert SamplerType.RANDOM.value == "random"

    def test_pruner_types(self):
        """Test pruner type enum values."""
        assert PrunerType.HYPERBAND.value == "hyperband"
        assert PrunerType.MEDIAN.value == "median"
        assert PrunerType.NONE.value == "none"

    def test_ensemble_strategies(self):
        """Test ensemble strategy enum values."""
        assert EnsembleStrategy.VOTING.value == "voting"
        assert EnsembleStrategy.STACKING.value == "stacking"
        assert EnsembleStrategy.HYBRID.value == "hybrid"


class TestSearchSpaceConfig:
    """Test cases for search space configuration."""

    def test_default_config(self):
        """Test default search space config."""
        config = SearchSpaceConfig()

        assert config.hidden_size_choices is not None
        assert config.num_layers_min is not None
        assert config.learning_rate_min is not None

    def test_config_fields_exist(self):
        """Test that config has expected fields."""
        config = SearchSpaceConfig()

        # Check key fields exist
        assert hasattr(config, 'hidden_size_choices')
        assert hasattr(config, 'num_layers_min')
        assert hasattr(config, 'batch_size_choices')
        assert hasattr(config, 'dropout_min')


class TestAutoMLConfig:
    """Test cases for AutoML configuration."""

    def test_default_config(self):
        """Test default AutoML config."""
        config = AutoMLConfig()

        assert config.n_trials >= 1
        assert config.sampler == SamplerType.TPE
        assert config.pruner == PrunerType.HYPERBAND

    def test_custom_config(self):
        """Test custom AutoML config."""
        config = AutoMLConfig(
            experiment_name="test_experiment",
            n_trials=50,
            sampler=SamplerType.RANDOM,
        )

        assert config.experiment_name == "test_experiment"
        assert config.n_trials == 50
        assert config.sampler == SamplerType.RANDOM


class TestTrialConfig:
    """Test cases for trial configuration."""

    def test_trial_config_creation(self):
        """Test trial config creation."""
        config = TrialConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            learning_rate=1e-4,
            batch_size=32,
        )

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6
        assert config.learning_rate == 1e-4

    def test_trial_config_has_all_fields(self):
        """Test trial config has all required fields."""
        config = TrialConfig()

        assert hasattr(config, 'hidden_size')
        assert hasattr(config, 'num_hidden_layers')
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'batch_size')


class TestRNAModelSearchSpace:
    """Test cases for RNA model search space."""

    def test_search_space_has_static_methods(self):
        """Test RNAModelSearchSpace has static sampling methods."""
        assert hasattr(RNAModelSearchSpace, 'sample_all')
        assert hasattr(RNAModelSearchSpace, 'sample_encoder_params')
        assert hasattr(RNAModelSearchSpace, 'sample_training_params')

    def test_sample_all_is_classmethod(self):
        """Test sample_all is a classmethod."""
        # sample_all should be callable as a classmethod
        assert callable(RNAModelSearchSpace.sample_all)


class TestTrialResult:
    """Test cases for trial result."""

    def test_trial_result_creation(self):
        """Test trial result creation."""
        result = TrialResult(
            trial_id=1,
            params={"hidden_size": 512},
            metrics={"accuracy": 0.85, "f1": 0.82},
        )

        assert result.trial_id == 1
        assert result.metrics["accuracy"] == 0.85

    def test_trial_result_default_values(self):
        """Test trial result default values."""
        result = TrialResult(
            trial_id=1,
            params={},
            metrics={},
        )

        assert result.status == "completed"
        assert result.error is None
        assert result.model_path is None


class TestAutoMLResult:
    """Test cases for AutoML result."""

    def test_automl_result_creation(self):
        """Test AutoML result creation."""
        best_trial = TrialResult(
            trial_id=5,
            params={"hidden_size": 512},
            metrics={"f1": 0.85},
        )

        result = AutoMLResult(
            experiment_name="test_exp",
            best_trial=best_trial,
        )

        assert result.experiment_name == "test_exp"
        assert result.best_trial.trial_id == 5

    def test_automl_result_default_values(self):
        """Test AutoML result default values."""
        best_trial = TrialResult(trial_id=1, params={}, metrics={})

        result = AutoMLResult(
            experiment_name="test",
            best_trial=best_trial,
        )

        assert result.all_trials == []
        assert result.study_stats == {}


class TestVotingEnsemble:
    """Test cases for voting ensemble."""

    def test_ensemble_initialization(self):
        """Test ensemble initializes correctly."""
        ensemble = VotingEnsemble()
        assert ensemble is not None

    def test_ensemble_has_models_list(self):
        """Test ensemble has models container."""
        ensemble = VotingEnsemble()
        assert hasattr(ensemble, 'models') or hasattr(ensemble, 'model_paths')


class TestHybridEnsemble:
    """Test cases for hybrid ensemble."""

    def test_ensemble_initialization(self):
        """Test hybrid ensemble initializes correctly."""
        ensemble = HybridEnsemble()
        assert ensemble is not None

    def test_ensemble_has_strategy(self):
        """Test ensemble has strategy attribute."""
        ensemble = HybridEnsemble(strategy="voting")
        assert hasattr(ensemble, 'strategy')


class TestTrialStorage:
    """Test cases for trial storage."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_storage_path):
        """Create a storage instance."""
        return TrialStorage(base_dir=temp_storage_path)

    def test_storage_initialization(self, storage):
        """Test storage initializes correctly."""
        assert storage is not None
        assert storage.base_dir.exists()

    def test_storage_creates_directories(self, storage):
        """Test storage creates required directories."""
        assert storage.trials_dir.exists()
        assert storage.experiments_dir.exists()

    def test_save_and_load_trial(self, storage):
        """Test saving and loading a trial."""
        result = TrialResult(
            trial_id=1,
            params={"hidden_size": 512},
            metrics={"accuracy": 0.85},
        )

        storage.save_trial_result(result)
        loaded = storage.load_trial_result(1)

        assert loaded is not None
        assert loaded["trial_id"] == 1
        assert loaded["metrics"]["accuracy"] == 0.85

    def test_list_trials(self, storage):
        """Test listing trials."""
        for i in range(3):
            result = TrialResult(
                trial_id=i,
                params={"hidden_size": 512},
                metrics={"accuracy": 0.80 + i * 0.05},
            )
            storage.save_trial_result(result)

        trials = storage.list_trials()
        assert len(trials) == 3

    def test_load_nonexistent_trial(self, storage):
        """Test loading nonexistent trial returns None."""
        result = storage.load_trial_result(999)
        assert result is None


class TestModelRegistry:
    """Test cases for model registry."""

    @pytest.fixture
    def temp_registry_path(self):
        """Create a temporary registry path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create a registry instance."""
        return ModelRegistry(base_dir=temp_registry_path)

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry is not None
        assert registry.models_dir.exists()

    def test_get_trial_path(self, registry):
        """Test getting trial path."""
        path = registry.get_trial_path("exp1", 5)
        assert "exp1" in path
        assert "trial_5" in path

    def test_save_trial_model(self, registry):
        """Test saving a trial model."""
        # Create a simple model
        model = nn.Linear(10, 5)

        # Create a mock tokenizer
        class MockTokenizer:
            def save(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        tokenizer = MockTokenizer()
        metrics = {"accuracy": 0.85}

        model_path = registry.save_trial_model(
            model=model,
            tokenizer=tokenizer,
            trial_id=1,
            metrics=metrics,
            experiment_name="exp1",
        )

        assert model_path is not None
        assert Path(model_path).exists()

    def test_list_models(self, registry):
        """Test listing models."""
        # Create and save some models
        model = nn.Linear(10, 5)

        class MockTokenizer:
            def save(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        tokenizer = MockTokenizer()

        for i in range(3):
            registry.save_trial_model(
                model=model,
                tokenizer=tokenizer,
                trial_id=i,
                metrics={"accuracy": 0.80 + i * 0.05},
                experiment_name="exp1",
            )

        models = registry.list_models("exp1")
        assert len(models) == 3


class TestIntegration:
    """Integration tests for AutoML system."""

    def test_config_dataclasses_are_valid(self):
        """Test that config dataclasses work correctly."""
        search_config = SearchSpaceConfig()
        automl_config = AutoMLConfig(
            experiment_name="test",
            search_space=search_config,
        )
        trial_config = TrialConfig()

        assert automl_config.experiment_name == "test"
        assert trial_config.hidden_size > 0

    def test_trial_result_to_storage_flow(self):
        """Test trial result can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrialStorage(base_dir=tmpdir)

            # Create and save trial result
            result = TrialResult(
                trial_id=1,
                params={"hidden_size": 256, "learning_rate": 0.001},
                metrics={"f1_macro": 0.75, "accuracy": 0.80},
                status="completed",
            )

            storage.save_trial_result(result)

            # Load and verify
            loaded = storage.load_trial_result(1)
            assert loaded["params"]["hidden_size"] == 256
            assert loaded["metrics"]["f1_macro"] == 0.75

    def test_model_registry_full_workflow(self):
        """Test model registry save/list/delete workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=tmpdir)

            # Create model
            model = nn.Linear(64, 8)

            class MockTokenizer:
                def save(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)

            tokenizer = MockTokenizer()

            # Save model
            path = registry.save_trial_model(
                model=model,
                tokenizer=tokenizer,
                trial_id=1,
                metrics={"f1": 0.85},
                experiment_name="workflow_test",
            )

            # List models
            models = registry.list_models("workflow_test")
            assert len(models) == 1

            # Delete model
            deleted = registry.delete_model("workflow_test", 1)
            assert deleted is True

            # Verify deletion
            models_after = registry.list_models("workflow_test")
            assert len(models_after) == 0
