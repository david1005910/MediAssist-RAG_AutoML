"""Medical Named Entity Recognition using BioBERT."""

from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class MedicalNER:
    """Medical entity recognition using BioBERT embeddings."""

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, text: str) -> np.ndarray:
        """Extract embeddings from text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        if self.model is None:
            self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().flatten()

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract medical entities from text.

        Args:
            text: Input text to analyze.

        Returns:
            List of extracted entities with their types.
        """
        # TODO: Implement NER extraction
        return []
