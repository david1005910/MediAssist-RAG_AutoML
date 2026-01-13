"""Schemas for RAG medical literature search API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """Source document reference."""
    title: str = Field(description="Document title")
    authors: Optional[str] = Field(default=None, description="Authors")
    year: Optional[str] = Field(default=None, description="Publication year")
    journal: Optional[str] = Field(default=None, description="Journal name")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    relevance: float = Field(ge=0, le=1, description="Relevance score")
    sparse_score: Optional[float] = Field(default=None, ge=0, le=1, description="SPLADE sparse score")
    dense_score: Optional[float] = Field(default=None, ge=0, le=1, description="Dense embedding score")


class HybridScore(BaseModel):
    """Hybrid search score breakdown."""
    sparse_score: float = Field(ge=0, le=1, description="SPLADE sparse search score")
    dense_score: float = Field(ge=0, le=1, description="Dense embedding search score")
    combined_score: float = Field(ge=0, le=1, description="Combined hybrid score (sparse*0.3 + dense*0.7)")


class SearchResult(BaseModel):
    """Single search result."""
    content: str = Field(description="Relevant text excerpt")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    score: float = Field(description="Relevance score")
    hybrid_score: Optional[HybridScore] = Field(default=None, description="Hybrid score breakdown")


class LiteratureSearchRequest(BaseModel):
    """Request for literature search."""
    query: str = Field(min_length=2, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "폐렴 치료 항생제",
                    "top_k": 5
                }
            ]
        }
    }


class LiteratureSearchResponse(BaseModel):
    """Response from literature search."""
    results: List[SearchResult] = Field(description="Search results")
    total_found: int = Field(description="Total results found")
    query: str = Field(description="Original query")


class RAGQueryRequest(BaseModel):
    """Request for RAG-based question answering."""
    question: str = Field(min_length=5, description="Question to answer")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (symptoms, patient info)"
    )
    include_sources: bool = Field(default=True, description="Include source citations")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "폐렴의 일반적인 치료법은 무엇인가요?",
                    "context": {
                        "symptoms": ["발열", "기침", "호흡곤란"],
                        "patient_age": 65
                    },
                    "include_sources": True
                }
            ]
        }
    }


class KnowledgeGraphNode(BaseModel):
    """Knowledge graph node."""
    id: str = Field(description="Node ID")
    label: str = Field(description="Node label/name")
    type: str = Field(description="Node type (Disease, Symptom, Treatment, etc.)")
    properties: Dict[str, Any] = Field(default={}, description="Node properties")


class KnowledgeGraphEdge(BaseModel):
    """Knowledge graph edge/relationship."""
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: str = Field(description="Relationship type")
    properties: Dict[str, Any] = Field(default={}, description="Edge properties")


class KnowledgeGraphData(BaseModel):
    """Knowledge graph data for visualization."""
    nodes: List[KnowledgeGraphNode] = Field(default=[], description="Graph nodes")
    edges: List[KnowledgeGraphEdge] = Field(default=[], description="Graph edges")
    context: str = Field(default="", description="Formatted graph context")


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str = Field(description="Generated answer")
    sources: List[DocumentSource] = Field(description="Source documents")
    context_used: Optional[str] = Field(default=None, description="Context text used")
    confidence: str = Field(description="Answer confidence: high/medium/low")
    knowledge_graph: Optional[KnowledgeGraphData] = Field(
        default=None,
        description="Related knowledge graph data"
    )
    disclaimer: str = Field(
        default="이 정보는 의료 전문가의 진단을 대체할 수 없습니다. 반드시 전문의와 상담하세요.",
        description="Medical disclaimer"
    )


class DocumentIngestionRequest(BaseModel):
    """Request to add documents to the knowledge base."""
    documents: List[Dict[str, Any]] = Field(description="Documents to add")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": [
                        {
                            "content": "폐렴은 폐 조직의 염증성 질환입니다...",
                            "title": "폐렴 개요",
                            "source": "의학 교과서",
                            "year": "2024"
                        }
                    ]
                }
            ]
        }
    }


class DocumentIngestionResponse(BaseModel):
    """Response from document ingestion."""
    chunks_added: int = Field(description="Number of chunks added")
    status: str = Field(description="Ingestion status")


class PubMedSearchRequest(BaseModel):
    """Request for PubMed search."""
    query: str = Field(min_length=2, description="Search query (supports PubMed syntax)")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    ingest: bool = Field(default=True, description="Whether to ingest results into VectorDB")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "pneumonia treatment antibiotics",
                    "max_results": 10,
                    "ingest": True
                },
                {
                    "query": "COVID-19[Title] AND treatment[Title/Abstract]",
                    "max_results": 20,
                    "ingest": True
                }
            ]
        }
    }


class PubMedArticle(BaseModel):
    """PubMed article details."""
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract")
    authors: str = Field(description="Authors")
    year: str = Field(description="Publication year")
    journal: str = Field(description="Journal name")
    doi: Optional[str] = Field(default=None, description="DOI")
    keywords: List[str] = Field(default=[], description="MeSH keywords")
    source: str = Field(default="PubMed", description="Source database")


class PubMedSearchResponse(BaseModel):
    """Response from PubMed search."""
    articles: List[PubMedArticle] = Field(description="Found articles")
    total_found: int = Field(description="Number of articles found")
    ingested: int = Field(default=0, description="Number of articles ingested to VectorDB")
    query: str = Field(description="Original query")


class AcademicSource(BaseModel):
    """Available academic search source."""
    id: str = Field(description="Source identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Source description")


class UnifiedSearchRequest(BaseModel):
    """Request for unified academic search across multiple sources."""
    query: str = Field(min_length=2, description="Search query")
    sources: Optional[List[str]] = Field(
        default=None,
        description="Sources to search (default: all). Options: pubmed, semantic_scholar, crossref, openalex, koreamed"
    )
    max_results_per_source: int = Field(default=10, ge=1, le=50, description="Max results per source")
    ingest: bool = Field(default=False, description="Whether to ingest results into VectorDB")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "COVID-19 treatment",
                    "sources": ["pubmed", "semantic_scholar"],
                    "max_results_per_source": 10,
                    "ingest": True
                }
            ]
        }
    }


class UnifiedSearchResponse(BaseModel):
    """Response from unified academic search."""
    query: str = Field(description="Original query")
    sources_searched: List[str] = Field(description="Sources that were searched")
    results: Dict[str, List[PubMedArticle]] = Field(description="Results by source")
    total_found: int = Field(description="Total articles found across all sources")
    ingested: int = Field(default=0, description="Number of articles ingested to VectorDB")
