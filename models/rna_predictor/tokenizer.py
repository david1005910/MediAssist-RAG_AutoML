"""Hybrid N-gram Tokenizer for RNA sequences.

Implements tokenization combining single nucleotides with N-gram features,
inspired by the RNAGenesis paper's hybrid tokenization approach.
"""

from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
from itertools import product
import re
import json
from pathlib import Path


class HybridNGramTokenizer:
    """Tokenizer combining single nucleotides with N-gram features.

    This tokenizer creates a vocabulary that includes:
    - Single nucleotides: A, U, G, C
    - 3-mers: All possible 3-nucleotide combinations (64 tokens)
    - 5-mers: All possible 5-nucleotide combinations (1024 tokens)
    - Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]

    Attributes:
        n_gram_sizes: Tuple of N-gram sizes to include
        max_length: Maximum sequence length for encoding
        vocab: Dictionary mapping tokens to IDs
        reverse_vocab: Dictionary mapping IDs to tokens
    """

    NUCLEOTIDES = ["A", "U", "G", "C"]
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }

    def __init__(
        self,
        n_gram_sizes: Tuple[int, ...] = (1, 3, 5),
        max_length: int = 512,
        min_freq: int = 1,
        vocab_size: Optional[int] = None,
    ):
        """Initialize the tokenizer.

        Args:
            n_gram_sizes: Tuple of N-gram sizes (e.g., (1, 3, 5) for 1-mer, 3-mer, 5-mer)
            max_length: Maximum sequence length for encoding
            min_freq: Minimum frequency for vocabulary (used when building from corpus)
            vocab_size: Maximum vocabulary size (None for unlimited)
        """
        self.n_gram_sizes = n_gram_sizes
        self.max_length = max_length
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        # Build default vocabulary
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self._build_default_vocab()

    def _build_default_vocab(self) -> None:
        """Build default vocabulary with all possible N-grams."""
        # Add special tokens
        self.vocab = dict(self.SPECIAL_TOKENS)

        # Add all possible N-grams for each size
        for n in sorted(self.n_gram_sizes):
            for ngram in product(self.NUCLEOTIDES, repeat=n):
                token = "".join(ngram)
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # Build reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def build_vocab_from_corpus(self, sequences: List[str]) -> None:
        """Build vocabulary from a corpus of RNA sequences.

        This method can be used to create a frequency-filtered vocabulary
        when training on a specific dataset.

        Args:
            sequences: List of RNA sequences
        """
        ngram_counts: Counter = Counter()

        for seq in sequences:
            seq = self._normalize_sequence(seq)
            for n in self.n_gram_sizes:
                for i in range(len(seq) - n + 1):
                    ngram = seq[i : i + n]
                    if self._is_valid_ngram(ngram):
                        ngram_counts[ngram] += 1

        # Start with special tokens
        self.vocab = dict(self.SPECIAL_TOKENS)

        # Filter by frequency and optionally limit vocab size
        common_ngrams = [
            ngram
            for ngram, count in ngram_counts.most_common()
            if count >= self.min_freq
        ]

        if self.vocab_size:
            max_ngrams = self.vocab_size - len(self.SPECIAL_TOKENS)
            common_ngrams = common_ngrams[:max_ngrams]

        for ngram in common_ngrams:
            self.vocab[ngram] = len(self.vocab)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _normalize_sequence(self, sequence: str) -> str:
        """Normalize RNA sequence (uppercase, T->U conversion)."""
        return sequence.upper().replace("T", "U")

    def _is_valid_ngram(self, ngram: str) -> bool:
        """Check if N-gram contains only valid nucleotides."""
        return bool(re.match(r"^[AUGC]+$", ngram))

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize RNA sequence into tokens.

        Uses greedy longest-match tokenization, preferring longer N-grams.

        Args:
            sequence: RNA sequence string

        Returns:
            List of tokens
        """
        sequence = self._normalize_sequence(sequence)
        tokens = []
        i = 0

        while i < len(sequence):
            matched = False
            # Try to match longest N-gram first
            for n in sorted(self.n_gram_sizes, reverse=True):
                ngram = sequence[i : i + n]
                if ngram in self.vocab:
                    tokens.append(ngram)
                    i += n
                    matched = True
                    break

            if not matched:
                # Unknown nucleotide
                tokens.append("[UNK]")
                i += 1

        return tokens

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        """Encode RNA sequence to token IDs.

        Args:
            sequence: RNA sequence string
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            max_length: Maximum length (default: self.max_length)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: Return type ('pt' for PyTorch, None for lists)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        max_len = max_length or self.max_length
        tokens = self.tokenize(sequence)

        # Add special tokens
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # Truncation
        if truncation and len(tokens) > max_len:
            if add_special_tokens:
                tokens = tokens[: max_len - 1] + ["[SEP]"]
            else:
                tokens = tokens[:max_len]

        # Convert to IDs
        input_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

        # Attention mask
        attention_mask = [1] * len(input_ids)

        # Padding
        if padding:
            padding_length = max_len - len(input_ids)
            input_ids.extend([self.vocab["[PAD]"]] * padding_length)
            attention_mask.extend([0] * padding_length)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch

            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result

    def encode_batch(
        self,
        sequences: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[List[int]], "torch.Tensor"]]:
        """Encode multiple RNA sequences.

        Args:
            sequences: List of RNA sequences
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            max_length: Maximum length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return type ('pt' for PyTorch, None for lists)

        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        all_input_ids = []
        all_attention_masks = []

        for seq in sequences:
            encoded = self.encode(
                seq,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None,
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

        if return_tensors == "pt":
            import torch

            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to RNA sequence.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded RNA sequence
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, "[UNK]")
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)

        return "".join(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def save(self, path: str) -> None:
        """Save tokenizer configuration and vocabulary.

        Args:
            path: Path to save directory
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "n_gram_sizes": list(self.n_gram_sizes),
            "max_length": self.max_length,
            "min_freq": self.min_freq,
            "vocab_size": self.vocab_size,
        }

        with open(save_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        with open(save_dir / "vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HybridNGramTokenizer":
        """Load tokenizer from saved files.

        Args:
            path: Path to save directory

        Returns:
            Loaded tokenizer instance
        """
        load_dir = Path(path)

        with open(load_dir / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(
            n_gram_sizes=tuple(config["n_gram_sizes"]),
            max_length=config["max_length"],
            min_freq=config["min_freq"],
            vocab_size=config.get("vocab_size"),
        )

        with open(load_dir / "vocab.json", "r") as f:
            tokenizer.vocab = json.load(f)

        tokenizer.reverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}

        return tokenizer


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of an RNA sequence.

    Args:
        sequence: RNA sequence

    Returns:
        GC content as a percentage (0-100)
    """
    sequence = sequence.upper().replace("T", "U")
    if len(sequence) == 0:
        return 0.0

    gc_count = sequence.count("G") + sequence.count("C")
    return (gc_count / len(sequence)) * 100


def find_motifs(sequence: str, motif_patterns: Optional[Dict[str, str]] = None) -> List[str]:
    """Find known RNA motifs in sequence.

    Args:
        sequence: RNA sequence
        motif_patterns: Dictionary of motif name to regex pattern

    Returns:
        List of found motif names
    """
    default_patterns = {
        "Kozak": r"[AG]CCAUGG",
        "Poly(A)": r"AAUAAA",
        "AU-rich": r"AUUUA",
        "Iron-responsive": r"CAGUGN",
        "IRES": r"CCCC.{10,50}AUG",
    }

    patterns = motif_patterns or default_patterns
    sequence = sequence.upper().replace("T", "U")
    found = []

    for motif_name, pattern in patterns.items():
        if re.search(pattern, sequence):
            found.append(motif_name)

    return found
