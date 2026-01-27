"""Tests for RNA predictor module."""

import pytest
from models.rna_predictor.tokenizer import HybridNGramTokenizer, calculate_gc_content, find_motifs
from models.rna_predictor.config import (
    RNAEncoderConfig,
    RNAClassifierConfig,
    RNAPredictorConfig,
    DISEASE_CLASSES,
    RNA_TYPES,
    PATHOGENICITY_LEVELS,
)
from models.rna_predictor.model import (
    CNNNgramEncoder,
    RNAEncoder,
    RNADiseaseClassifier,
    RNAPredictionModel,
    RNAPredictor,
)
import torch


class TestHybridNGramTokenizer:
    """Test cases for HybridNGramTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return HybridNGramTokenizer()

    def test_tokenizer_initialization(self, tokenizer):
        """Test that tokenizer initializes with correct vocab size."""
        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size > 0
        # Should have special tokens + nucleotides + n-grams
        assert vocab_size >= 5 + 4  # 5 special tokens + A, U, G, C

    def test_encode_returns_dict(self, tokenizer):
        """Test encoding returns a dictionary."""
        sequence = "AUGCAUGC"
        result = tokenizer.encode(sequence)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_encode_input_ids(self, tokenizer):
        """Test encoding produces correct input_ids format."""
        sequence = "AUGCAUGC"
        result = tokenizer.encode(sequence)

        assert isinstance(result["input_ids"], list)
        # First token should be CLS (id=2)
        assert result["input_ids"][0] == 2  # CLS token

    def test_encode_with_max_length(self, tokenizer):
        """Test encoding respects max_length."""
        sequence = "AUGC" * 200  # Long sequence
        result = tokenizer.encode(sequence, max_length=100)

        assert len(result["input_ids"]) == 100
        assert len(result["attention_mask"]) == 100

    def test_encode_with_padding(self, tokenizer):
        """Test encoding adds padding when needed."""
        sequence = "AUGC"
        result = tokenizer.encode(sequence, max_length=50, padding=True)

        assert len(result["input_ids"]) == 50
        # Check for padding tokens (id=0)
        assert 0 in result["input_ids"]  # PAD token present

    def test_decode_sequence(self, tokenizer):
        """Test decoding returns sequence."""
        sequence = "AUGCAUGC"
        result = tokenizer.encode(sequence)
        decoded = tokenizer.decode(result["input_ids"])

        # Decoded should contain original nucleotides (maybe as n-grams)
        assert len(decoded) > 0

    def test_invalid_nucleotides_handled(self, tokenizer):
        """Test that invalid nucleotides are handled as UNK."""
        sequence = "AUGXYZ"  # X, Y, Z are invalid
        result = tokenizer.encode(sequence)

        assert isinstance(result, dict)
        assert len(result["input_ids"]) > 0

    def test_tokenize_method(self, tokenizer):
        """Test tokenize method returns list of tokens."""
        sequence = "AUGCAUGC"
        tokens = tokenizer.tokenize(sequence)

        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestGCContentCalculation:
    """Test cases for GC content calculation."""

    def test_gc_content_all_gc(self):
        """Test GC content for all GC sequence."""
        content = calculate_gc_content("GGCC")
        assert content == 100.0

    def test_gc_content_no_gc(self):
        """Test GC content for no GC sequence."""
        content = calculate_gc_content("AUAU")
        assert content == 0.0

    def test_gc_content_half(self):
        """Test GC content for half GC sequence."""
        content = calculate_gc_content("AUGC")
        assert content == 50.0

    def test_gc_content_empty(self):
        """Test GC content for empty sequence."""
        content = calculate_gc_content("")
        assert content == 0.0

    def test_gc_content_with_t(self):
        """Test GC content handles T as U."""
        content = calculate_gc_content("ATGC")  # T should be converted to U
        assert content == 50.0


class TestMotifFinding:
    """Test cases for motif finding."""

    def test_find_polya_signal(self):
        """Test finding poly-A signal."""
        motifs = find_motifs("AAUAAA")
        assert "Poly(A)" in motifs

    def test_find_kozak(self):
        """Test finding Kozak sequence."""
        motifs = find_motifs("GCCAUGG")
        assert "Kozak" in motifs

    def test_find_au_rich(self):
        """Test finding AU-rich element."""
        motifs = find_motifs("AUUUA")
        assert "AU-rich" in motifs

    def test_no_motifs_found(self):
        """Test when no motifs are found."""
        motifs = find_motifs("GGGGGGGG")
        assert isinstance(motifs, list)


class TestRNAEncoderConfig:
    """Test cases for RNA encoder configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RNAEncoderConfig()

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.max_position_embeddings == 512

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RNAEncoderConfig(
            hidden_size=256,
            num_hidden_layers=6,
        )

        assert config.hidden_size == 256
        assert config.num_hidden_layers == 6


class TestCNNNgramEncoder:
    """Test cases for CNN N-gram encoder."""

    @pytest.fixture
    def cnn_encoder(self):
        """Create a CNN encoder instance."""
        return CNNNgramEncoder(
            vocab_size=1100,
            hidden_size=128,
            kernel_sizes=(3, 5),
            out_channels=64,
        )

    def test_forward_shape(self, cnn_encoder):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = cnn_encoder(input_ids)

        assert output.shape == (batch_size, seq_len, 128)  # hidden_size


class TestRNAEncoder:
    """Test cases for RNA encoder."""

    @pytest.fixture
    def encoder_config(self):
        """Create a small encoder config for testing."""
        return RNAEncoderConfig(
            vocab_size=1100,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=64,
            cnn_out_channels=64,
        )

    @pytest.fixture
    def encoder(self, encoder_config):
        """Create an encoder instance."""
        return RNAEncoder(encoder_config)

    def test_forward_returns_tuple(self, encoder, encoder_config):
        """Test forward pass returns tuple of outputs."""
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = encoder(input_ids)

        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_forward_shapes(self, encoder, encoder_config):
        """Test forward pass output shapes."""
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        sequence_output, pooled_output = encoder(input_ids)

        assert sequence_output.shape == (batch_size, seq_len, encoder_config.hidden_size)
        assert pooled_output.shape == (batch_size, encoder_config.hidden_size)


class TestRNADiseaseClassifier:
    """Test cases for RNA disease classifier."""

    @pytest.fixture
    def classifier_config(self):
        """Create classifier config for testing."""
        return RNAClassifierConfig(
            hidden_size=128,
            num_rna_types=5,
            num_diseases=8,
        )

    @pytest.fixture
    def classifier(self, classifier_config):
        """Create a classifier instance."""
        return RNADiseaseClassifier(classifier_config)

    def test_forward_output_keys(self, classifier):
        """Test forward pass output contains expected keys."""
        batch_size = 2
        hidden_size = 128
        pooled_output = torch.randn(batch_size, hidden_size)

        outputs = classifier(pooled_output)

        assert "rna_type_logits" in outputs
        assert "disease_logits" in outputs
        assert "pathogenicity_logits" in outputs
        assert "risk_score" in outputs

    def test_forward_output_shapes(self, classifier, classifier_config):
        """Test forward pass output shapes."""
        batch_size = 2
        hidden_size = 128
        pooled_output = torch.randn(batch_size, hidden_size)

        outputs = classifier(pooled_output)

        assert outputs["rna_type_logits"].shape == (batch_size, classifier_config.num_rna_types)
        assert outputs["disease_logits"].shape == (batch_size, classifier_config.num_diseases)
        assert outputs["pathogenicity_logits"].shape == (batch_size, len(PATHOGENICITY_LEVELS))
        assert outputs["risk_score"].shape == (batch_size,)


class TestRNAPredictionModel:
    """Test cases for complete RNA prediction model."""

    @pytest.fixture
    def small_config(self):
        """Create small configs for fast testing."""
        encoder = RNAEncoderConfig(
            vocab_size=1100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            cnn_out_channels=32,
        )
        classifier = RNAClassifierConfig(
            hidden_size=64,
            num_rna_types=5,
            num_diseases=8,
        )
        return encoder, classifier

    @pytest.fixture
    def model(self, small_config):
        """Create a model instance."""
        encoder_config, classifier_config = small_config
        return RNAPredictionModel(encoder_config, classifier_config)

    def test_forward_pass(self, model):
        """Test complete forward pass."""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = model(input_ids)

        assert "rna_type_logits" in outputs
        assert "disease_logits" in outputs
        assert "risk_score" in outputs

    def test_model_parameter_count(self, model):
        """Test model has trainable parameters."""
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0


class TestRNAPredictor:
    """Test cases for RNA predictor service."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance with small config."""
        config = RNAPredictorConfig(
            encoder=RNAEncoderConfig(
                vocab_size=1100,
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=128,
                max_position_embeddings=64,
                cnn_out_channels=32,
            ),
            classifier=RNAClassifierConfig(
                hidden_size=64,
                num_rna_types=5,
                num_diseases=8,
            ),
        )
        return RNAPredictor(config)

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor is not None
        assert predictor.tokenizer is not None

    def test_predict_returns_dict(self, predictor):
        """Test predict returns expected structure."""
        sequence = "AUGCAUGCAUGCAUGCAUGC"

        result = predictor.predict(sequence)

        assert isinstance(result, dict)
        assert "sequence_analysis" in result
        assert "disease_predictions" in result
        assert "risk_assessment" in result
        assert "disclaimer" in result

    def test_sequence_analysis_fields(self, predictor):
        """Test sequence analysis contains expected fields."""
        sequence = "AUGCAUGCAUGCAUGCAUGC"

        result = predictor.predict(sequence)
        analysis = result["sequence_analysis"]

        assert "length" in analysis
        assert "gc_content" in analysis
        assert "detected_rna_type" in analysis
        assert "rna_type_confidence" in analysis

    def test_disease_predictions_format(self, predictor):
        """Test disease predictions have correct format."""
        sequence = "AUGCAUGCAUGCAUGCAUGC"

        result = predictor.predict(sequence)
        predictions = result["disease_predictions"]

        assert isinstance(predictions, list)
        if len(predictions) > 0:
            pred = predictions[0]
            assert "disease" in pred
            assert "probability" in pred
            assert 0 <= pred["probability"] <= 1

    def test_risk_assessment_format(self, predictor):
        """Test risk assessment has correct format."""
        sequence = "AUGCAUGCAUGCAUGCAUGC"

        result = predictor.predict(sequence)
        risk = result["risk_assessment"]

        assert "risk_score" in risk
        assert "risk_level" in risk
        assert 0 <= risk["risk_score"] <= 100

    def test_batch_predict(self, predictor):
        """Test batch prediction."""
        sequences = [
            "AUGCAUGCAUGCAUGCAUGC",
            "GCGCGCGCGCGCGCGCGCGC",
        ]

        results = predictor.predict_batch(sequences)

        assert isinstance(results, list)
        assert len(results) == 2


class TestDiseaseAndRNATypeConstants:
    """Test cases for disease and RNA type constants."""

    def test_disease_classes_not_empty(self):
        """Test disease classes are defined."""
        assert len(DISEASE_CLASSES) > 0

    def test_rna_types_not_empty(self):
        """Test RNA types are defined."""
        assert len(RNA_TYPES) > 0

    def test_disease_classes_format(self):
        """Test disease classes are tuples with 3 elements."""
        for disease in DISEASE_CLASSES:
            assert isinstance(disease, tuple)
            assert len(disease) == 3  # (Korean name, English name, ICD code)

    def test_rna_types_are_strings(self):
        """Test RNA types are strings."""
        for rna_type in RNA_TYPES:
            assert isinstance(rna_type, str)

    def test_pathogenicity_levels_defined(self):
        """Test pathogenicity levels are defined."""
        assert len(PATHOGENICITY_LEVELS) > 0
        assert "benign" in PATHOGENICITY_LEVELS
        assert "pathogenic" in PATHOGENICITY_LEVELS
