"""Tests for symptom classifier."""

import pytest
from models.symptom_classifier import SymptomClassifier


class TestSymptomClassifier:
    """Test cases for SymptomClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return SymptomClassifier()

    def test_predict_returns_list(self, classifier, sample_symptoms):
        """Test that predict returns a list."""
        # Note: This will fail without a trained model
        # Demonstrates test structure
        try:
            predictions = classifier.predict(sample_symptoms)
            assert isinstance(predictions, list)
            assert len(predictions) <= 5
        except ValueError:
            # Model not trained
            pytest.skip("Model not trained")

    def test_prediction_format(self, classifier, sample_symptoms):
        """Test prediction format."""
        try:
            predictions = classifier.predict(sample_symptoms)
            for pred in predictions:
                assert "disease" in pred
                assert "probability" in pred
                assert 0 <= pred["probability"] <= 1
                assert "confidence" in pred
                assert pred["confidence"] in ["high", "medium", "low"]
        except ValueError:
            pytest.skip("Model not trained")

    def test_get_confidence_high(self, classifier):
        """Test high confidence threshold."""
        assert classifier._get_confidence(0.8) == "high"
        assert classifier._get_confidence(0.7) == "high"

    def test_get_confidence_medium(self, classifier):
        """Test medium confidence threshold."""
        assert classifier._get_confidence(0.5) == "medium"
        assert classifier._get_confidence(0.4) == "medium"

    def test_get_confidence_low(self, classifier):
        """Test low confidence threshold."""
        assert classifier._get_confidence(0.3) == "low"
        assert classifier._get_confidence(0.1) == "low"
