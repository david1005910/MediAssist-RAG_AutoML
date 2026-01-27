"""Cross-encoder reranking."""

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank results using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query.
            documents: List of document texts.
            top_k: Number of top results to return.

        Returns:
            Reranked list of (document, score) tuples.
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score pairs
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(doc, float(score)) for doc, score in ranked[:top_k]]
