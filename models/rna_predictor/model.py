"""RNA Encoder and Disease Classifier models.

Implements BERT-style RNA encoder with hybrid N-gram features and
multi-task disease classifier for RNA sequence analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    RNAEncoderConfig,
    RNAClassifierConfig,
    RNAPredictorConfig,
    RNA_TYPES,
    DISEASE_CLASSES,
    PATHOGENICITY_LEVELS,
    RISK_LEVELS,
    CONFIDENCE_THRESHOLDS,
)
from .tokenizer import HybridNGramTokenizer, calculate_gc_content, find_motifs


class CNNNgramEncoder(nn.Module):
    """CNN-based N-gram feature extractor.

    Extracts N-gram features using parallel convolutions with different
    kernel sizes, inspired by the RNAGenesis hybrid tokenization approach.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        kernel_sizes: Tuple[int, ...] = (3, 5),
        out_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Parallel convolutions for different N-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Combine all conv outputs
        self.fc = nn.Linear(out_channels * len(kernel_sizes), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract N-gram features.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            N-gram features [batch, seq_len, hidden_size]
        """
        # Embed tokens
        x = self.embedding(input_ids)  # [batch, seq_len, hidden]

        # Transpose for convolution [batch, hidden, seq_len]
        x = x.transpose(1, 2)

        # Apply parallel convolutions
        conv_outputs = [conv(x) for conv in self.convs]

        # Concatenate and transpose back
        x = torch.cat(conv_outputs, dim=1)  # [batch, out_channels * n_kernels, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, out_channels * n_kernels]

        # Project to hidden size
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-LayerNorm.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        # Self-attention with pre-norm
        normed = self.norm1(hidden_states)
        attn_output, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # FFN with pre-norm
        normed = self.norm2(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = hidden_states + self.dropout(ffn_output)

        return hidden_states


class RNAEncoder(nn.Module):
    """BERT-style RNA sequence encoder.

    Combines token embeddings with CNN-based N-gram features and
    processes through transformer layers to produce contextualized
    representations.
    """

    def __init__(self, config: RNAEncoderConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # CNN N-gram encoder
        self.ngram_encoder = CNNNgramEncoder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            kernel_sizes=config.n_gram_sizes,
            out_channels=config.cnn_out_channels,
            dropout=config.hidden_dropout_prob,
        )

        # Combine embeddings
        self.embed_combine = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.embed_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode RNA sequence.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1 for valid, 0 for padding)

        Returns:
            Tuple of:
                - sequence_output: Contextualized representations [batch, seq_len, hidden]
                - pooled_output: Pooled representation [batch, hidden]
        """
        batch_size, seq_len = input_ids.shape

        # Get position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        # N-gram features
        ngram_features = self.ngram_encoder(input_ids)

        # Combine embeddings
        combined = torch.cat([token_embeds + position_embeds, ngram_features], dim=-1)
        hidden_states = self.embed_combine(combined)
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_dropout(hidden_states)

        # Convert attention mask for MultiheadAttention (0 for valid, 1 for padding)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask).bool()

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        sequence_output = self.final_norm(hidden_states)

        # Pool [CLS] token
        pooled_output = self.pooler(sequence_output[:, 0, :])

        return sequence_output, pooled_output


class RNADiseaseClassifier(nn.Module):
    """Multi-task classifier for RNA disease prediction.

    Predicts:
    - RNA type (mRNA, siRNA, circRNA, lncRNA, other)
    - Disease class (8 categories)
    - Risk score (0-100)
    - Pathogenicity level (5 categories)
    """

    def __init__(self, config: RNAClassifierConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
        )

        # Task-specific heads
        self.rna_type_head = nn.Linear(config.hidden_dims[1], config.num_rna_types)
        self.disease_head = nn.Linear(config.hidden_dims[1], config.num_diseases)
        self.pathogenicity_head = nn.Linear(config.hidden_dims[1], len(PATHOGENICITY_LEVELS))
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict from pooled encoder output.

        Args:
            pooled_output: Pooled representation [batch, hidden]

        Returns:
            Dictionary with predictions:
                - rna_type_logits: [batch, num_rna_types]
                - disease_logits: [batch, num_diseases]
                - pathogenicity_logits: [batch, 5]
                - risk_score: [batch] (0-1, will be scaled to 0-100)
        """
        shared_features = self.shared(pooled_output)

        return {
            "rna_type_logits": self.rna_type_head(shared_features),
            "disease_logits": self.disease_head(shared_features),
            "pathogenicity_logits": self.pathogenicity_head(shared_features),
            "risk_score": self.risk_head(shared_features).squeeze(-1),
        }


class RNAPredictionModel(nn.Module):
    """Combined RNA encoder and classifier model."""

    def __init__(
        self,
        encoder_config: RNAEncoderConfig,
        classifier_config: RNAClassifierConfig,
    ):
        super().__init__()
        self.encoder = RNAEncoder(encoder_config)
        self.classifier = RNADiseaseClassifier(classifier_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and classifier.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with all predictions
        """
        _, pooled_output = self.encoder(input_ids, attention_mask)
        return self.classifier(pooled_output)


class RNAPredictor:
    """RNA sequence disease prediction service.

    Provides high-level interface for RNA sequence analysis with
    lazy loading, batch processing, and formatted output.
    """

    def __init__(
        self,
        config: Optional[RNAPredictorConfig] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize predictor.

        Args:
            config: Predictor configuration
            model_path: Path to model weights (overrides config.model_path)
        """
        self.config = config or RNAPredictorConfig()
        self._model: Optional[RNAPredictionModel] = None
        self._tokenizer: Optional[HybridNGramTokenizer] = None
        self._device: Optional[torch.device] = None

        if model_path:
            self.config.model_path = model_path

    @property
    def device(self) -> torch.device:
        """Get device for inference."""
        if self._device is None:
            if self.config.device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.config.device)
        return self._device

    @property
    def model(self) -> RNAPredictionModel:
        """Lazy load model."""
        if self._model is None:
            self._model = RNAPredictionModel(
                self.config.encoder,
                self.config.classifier,
            )
            if self.config.model_path:
                self.load(self.config.model_path)
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self) -> HybridNGramTokenizer:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = HybridNGramTokenizer(
                n_gram_sizes=self.config.encoder.n_gram_sizes,
                max_length=self.config.encoder.max_position_embeddings,
            )
        return self._tokenizer

    def _get_confidence(self, probability: float) -> str:
        """Get confidence level from probability."""
        if probability >= CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif probability >= CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    def _get_risk_level(self, score: float) -> str:
        """Get risk level from score (0-100)."""
        for level, low, high in RISK_LEVELS:
            if low <= score < high:
                return level
        return "critical"

    def predict(
        self,
        sequence: str,
        rna_type: Optional[str] = None,
    ) -> Dict:
        """Predict diseases from RNA sequence.

        Args:
            sequence: RNA sequence (A, U, G, C nucleotides)
            rna_type: Optional RNA type hint

        Returns:
            Dictionary with predictions:
                - sequence_analysis: Sequence statistics and detected type
                - disease_predictions: List of disease predictions
                - risk_assessment: Risk score and level
                - disclaimer: Medical disclaimer
        """
        # Encode sequence
        encoded = self.tokenizer.encode(
            sequence,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Process predictions
        rna_type_probs = F.softmax(outputs["rna_type_logits"], dim=-1)[0]
        disease_probs = F.softmax(outputs["disease_logits"], dim=-1)[0]
        pathogenicity_probs = F.softmax(outputs["pathogenicity_logits"], dim=-1)[0]
        risk_score = outputs["risk_score"][0].item() * 100

        # Get predicted RNA type
        rna_type_idx = rna_type_probs.argmax().item()
        detected_rna_type = RNA_TYPES[rna_type_idx]
        rna_type_confidence = rna_type_probs[rna_type_idx].item()

        # Get disease predictions (sorted by probability)
        disease_predictions = []
        sorted_indices = disease_probs.argsort(descending=True)
        for idx in sorted_indices:
            prob = disease_probs[idx].item()
            if prob < 0.05:  # Skip very low probability predictions
                continue
            disease_info = DISEASE_CLASSES[idx]
            disease_predictions.append({
                "disease": disease_info[0],  # Korean name
                "disease_en": disease_info[1],  # English name
                "icd_code": disease_info[2],
                "probability": round(prob, 4),
                "confidence": self._get_confidence(prob),
            })

        # Get pathogenicity
        pathogenicity_idx = pathogenicity_probs.argmax().item()
        pathogenicity = PATHOGENICITY_LEVELS[pathogenicity_idx]
        pathogenicity_confidence = pathogenicity_probs[pathogenicity_idx].item()

        # Sequence analysis
        gc_content = calculate_gc_content(sequence)
        motifs = find_motifs(sequence)

        # Risk factors
        risk_factors = []
        if risk_score >= 50:
            risk_factors.append("높은 전체 위험 점수")
        if pathogenicity in ["likely_pathogenic", "pathogenic"]:
            risk_factors.append("병원성 예측")
        if gc_content < 30 or gc_content > 70:
            risk_factors.append("비정상적 GC 함량")

        # Recommendations
        recommendations = []
        risk_level = self._get_risk_level(risk_score)
        if risk_level in ["high", "critical"]:
            recommendations.append("전문 의료진 상담을 권장합니다")
            recommendations.append("추가 진단 검사를 고려하세요")
        if pathogenicity in ["likely_pathogenic", "pathogenic"]:
            recommendations.append("유전 상담을 권장합니다")

        return {
            "sequence_analysis": {
                "length": len(sequence),
                "gc_content": round(gc_content, 2),
                "detected_rna_type": rna_type or detected_rna_type,
                "rna_type_confidence": round(rna_type_confidence, 4),
                "motifs_found": motifs,
            },
            "disease_predictions": disease_predictions[:5],  # Top 5
            "risk_assessment": {
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "pathogenicity": pathogenicity,
                "pathogenicity_confidence": round(pathogenicity_confidence, 4),
                "factors": risk_factors,
                "recommendations": recommendations,
            },
            "disclaimer": "이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요.",
        }

    def predict_batch(
        self,
        sequences: List[str],
        rna_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Batch prediction for multiple sequences.

        Args:
            sequences: List of RNA sequences
            rna_types: Optional list of RNA type hints

        Returns:
            List of prediction dictionaries
        """
        if rna_types is None:
            rna_types = [None] * len(sequences)

        results = []
        for seq, rna_type in zip(sequences, rna_types):
            results.append(self.predict(seq, rna_type))

        return results

    def save(self, path: str) -> None:
        """Save model weights and tokenizer.

        Args:
            path: Directory path for saving
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), save_dir / "model.pt")

        # Save tokenizer
        self.tokenizer.save(str(save_dir / "tokenizer"))

    def load(self, path: str) -> None:
        """Load model weights and tokenizer.

        Args:
            path: Directory path for loading
        """
        load_dir = Path(path)

        # Load model weights
        model_path = load_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        # Load tokenizer if exists
        tokenizer_path = load_dir / "tokenizer"
        if tokenizer_path.exists():
            self._tokenizer = HybridNGramTokenizer.load(str(tokenizer_path))
