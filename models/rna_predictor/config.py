"""Configuration classes for RNA Predictor models."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RNAEncoderConfig:
    """Configuration for RNA BERT-style encoder.

    Attributes:
        vocab_size: Size of the vocabulary (default ~1,097 for hybrid N-gram)
        hidden_size: Dimension of hidden layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Dimension of feedforward layer
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention
        max_position_embeddings: Maximum sequence length
        n_gram_sizes: Tuple of N-gram kernel sizes for CNN encoder
        cnn_out_channels: Output channels for CNN N-gram encoder
    """
    vocab_size: int = 1097
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    n_gram_sizes: Tuple[int, ...] = (3, 5)
    cnn_out_channels: int = 128
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


@dataclass
class RNAClassifierConfig:
    """Configuration for RNA disease classifier.

    Attributes:
        hidden_size: Input dimension from encoder
        num_rna_types: Number of RNA type classes
        num_diseases: Number of disease classes
        classifier_dropout: Dropout probability for classifier
        hidden_dims: Dimensions for classifier hidden layers
    """
    hidden_size: int = 512
    num_rna_types: int = 5  # mRNA, siRNA, circRNA, lncRNA, other
    num_diseases: int = 8
    classifier_dropout: float = 0.2
    hidden_dims: Tuple[int, int] = (256, 128)


@dataclass
class RNAPredictorConfig:
    """Complete configuration for RNA Predictor.

    Attributes:
        encoder: Encoder configuration
        classifier: Classifier configuration
        model_path: Path to model weights
        device: Device for inference ('cuda' or 'cpu')
    """
    encoder: RNAEncoderConfig = field(default_factory=RNAEncoderConfig)
    classifier: RNAClassifierConfig = field(default_factory=RNAClassifierConfig)
    model_path: Optional[str] = None
    device: str = "auto"


# RNA Types
RNA_TYPES = [
    "mRNA",
    "siRNA",
    "circRNA",
    "lncRNA",
    "other",
]

# Disease classes for prediction
DISEASE_CLASSES = [
    ("정상/저위험", "Normal/Low Risk", "N/A"),
    ("RNA 변이 관련 질환", "RNA Mutation Related Disease", "Q89.9"),
    ("siRNA 치료 반응 예측", "siRNA Therapy Response Prediction", "Z51.1"),
    ("ASO 효능 예측", "ASO Efficacy Prediction", "Z51.1"),
    ("UTR 변이 병원성", "UTR Variant Pathogenicity", "Q99.8"),
    ("유전성 근육 질환", "Hereditary Muscle Disease", "G71.9"),
    ("신경퇴행성 질환", "Neurodegenerative Disease", "G31.9"),
    ("암 관련 RNA 이상", "Cancer-related RNA Abnormality", "C80.1"),
]

# Pathogenicity levels
PATHOGENICITY_LEVELS = [
    "benign",
    "likely_benign",
    "uncertain",
    "likely_pathogenic",
    "pathogenic",
]

# Risk levels
RISK_LEVELS = [
    ("low", 0, 25),
    ("moderate", 25, 50),
    ("high", 50, 75),
    ("critical", 75, 100),
]

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0,
}
