"""Hybrid retrieval combining dense and sparse methods."""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    content: str
    metadata: Dict
    score: float


class HybridRetriever:
    """Combine dense and sparse retrieval with RRF."""

    def __init__(
        self,
        vectorstore,
        embedder,
        dense_weight: float = 0.5,
        k: int = 60,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.k = k

        # BM25 index
        self.bm25 = None
        self.documents = []

    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents for BM25.

        Args:
            documents: List of documents to index.
        """
        self.documents = documents
        tokenized = [self._tokenize(d["content"]) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """Retrieve documents using hybrid approach.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of retrieval results.
        """
        # Dense retrieval
        dense_results = self._dense_retrieve(query, self.k)

        # Sparse retrieval
        sparse_results = self._sparse_retrieve(query, self.k)

        # Combine with RRF
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            top_k,
        )

        return combined

    def _dense_retrieve(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[str, float]]:
        """Dense retrieval using vector similarity."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, 1.0 / (i + 1)) for i, (doc, _) in enumerate(results)]

    def _sparse_retrieve(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[str, float]]:
        """Sparse retrieval using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]

        return [
            (self.documents[i]["content"], 1.0 / (rank + 1))
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Number of final results.
            k: RRF constant.

        Returns:
            Combined and reranked results.
        """
        scores = {}

        # Add dense scores
        for rank, (content, _) in enumerate(dense_results):
            scores[content] = scores.get(content, 0) + self.dense_weight / (k + rank + 1)

        # Add sparse scores
        for rank, (content, _) in enumerate(sparse_results):
            scores[content] = scores.get(content, 0) + self.sparse_weight / (k + rank + 1)

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(content=content, metadata={}, score=score)
            for content, score in sorted_results[:top_k]
        ]
