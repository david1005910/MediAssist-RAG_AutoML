"""RAG medical literature search API router with Qdrant + SPLADE hybrid search + Neo4j Knowledge Graph + PubMed."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import uuid
import numpy as np
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

from ..config import settings
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Prefetch,
    FusionQuery,
    Query as QdrantQuery,
)

from app.schemas.rag import (
    LiteratureSearchRequest,
    LiteratureSearchResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentIngestionRequest,
    DocumentIngestionResponse,
    SearchResult,
    DocumentSource,
    HybridScore,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeGraphData,
    PubMedSearchRequest,
    PubMedSearchResponse,
    PubMedArticle,
    AcademicSource,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)

# Add rag module to path
RAG_PATH = Path("/app/rag")
if not RAG_PATH.exists():
    RAG_PATH = Path(__file__).parents[4] / "rag"
if str(RAG_PATH.parent) not in sys.path:
    sys.path.insert(0, str(RAG_PATH.parent))

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Literature Search"])

# Global instances
_qdrant_client: Optional[QdrantClient] = None
_dense_embedder = None
_sparse_embedder = None
_neo4j_driver = None

# Hybrid search weights: sparse:dense = 3:7
SPARSE_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# PubMed API Configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")  # Optional: increases rate limit


def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Search PubMed and return list of PMIDs.

    Args:
        query: Search query (supports PubMed search syntax)
        max_results: Maximum number of results to return

    Returns:
        List of PubMed IDs (PMIDs)
    """
    search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        print(f"[PubMed] Search '{query}' returned {len(pmids)} results")
        return pmids
    except Exception as e:
        print(f"[PubMed] Search error: {e}")
        return []


def fetch_pubmed_details(pmids: List[str]) -> List[Dict]:
    """Fetch detailed article information from PubMed.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of article dictionaries with title, abstract, authors, etc.
    """
    if not pmids:
        return []

    fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml"
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        response = requests.get(fetch_url, params=params, timeout=60)
        response.raise_for_status()

        articles = []
        root = ET.fromstring(response.content)

        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract PMID
                pmid = article.findtext(".//PMID", "")

                # Extract title
                title = article.findtext(".//ArticleTitle", "")

                # Extract abstract (may have multiple parts)
                abstract_parts = []
                for abstract_text in article.findall(".//AbstractText"):
                    label = abstract_text.get("Label", "")
                    text = abstract_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                # Extract authors
                authors_list = []
                for author in article.findall(".//Author"):
                    last_name = author.findtext("LastName", "")
                    fore_name = author.findtext("ForeName", "")
                    if last_name:
                        authors_list.append(f"{last_name} {fore_name}".strip())
                authors = ", ".join(authors_list[:5])  # Limit to first 5 authors
                if len(authors_list) > 5:
                    authors += " et al."

                # Extract publication year
                year = article.findtext(".//PubDate/Year", "")
                if not year:
                    medline_date = article.findtext(".//PubDate/MedlineDate", "")
                    if medline_date:
                        year = medline_date[:4]

                # Extract journal
                journal = article.findtext(".//Journal/Title", "")
                if not journal:
                    journal = article.findtext(".//Journal/ISOAbbreviation", "")

                # Extract DOI
                doi = ""
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text or ""
                        break

                # Extract keywords/MeSH terms
                keywords = []
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    keywords.append(mesh.text or "")

                # Only include if we have title and abstract
                if title and abstract:
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "doi": doi,
                        "keywords": keywords[:10],  # Limit keywords
                        "content": f"{title}\n\n{abstract}",
                        "source": "PubMed"
                    })
            except Exception as e:
                print(f"[PubMed] Error parsing article: {e}")
                continue

        print(f"[PubMed] Fetched {len(articles)} articles with abstracts")
        return articles

    except Exception as e:
        print(f"[PubMed] Fetch error: {e}")
        return []


def search_and_fetch_pubmed(query: str, max_results: int = 20) -> List[Dict]:
    """Search PubMed and fetch article details in one call.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of article dictionaries ready for ingestion
    """
    pmids = search_pubmed(query, max_results)
    if not pmids:
        return []
    return fetch_pubmed_details(pmids)


# ============================================================================
# Semantic Scholar API
# ============================================================================
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, max_results: int = 20) -> List[Dict]:
    """Search Semantic Scholar for academic papers.

    Args:
        query: Search query
        max_results: Maximum number of results (max 100)

    Returns:
        List of article dictionaries
    """
    import time

    search_url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "paperId,title,abstract,authors,year,venue,citationCount,openAccessPdf,externalIds"
    }

    # Retry logic for rate limiting
    max_retries = 3
    data = None

    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, params=params, timeout=30)

            # Handle rate limiting with retry
            if response.status_code == 429:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                print(f"[SemanticScholar] Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"[SemanticScholar] Search error after {max_retries} attempts: {e}")
                return []
            time.sleep((attempt + 1) * 2)
            continue

    if data is None:
        print(f"[SemanticScholar] Max retries exceeded for query: {query}")
        return []

    articles = []
    for paper in data.get("data", []):
        abstract = paper.get("abstract", "")
        title = paper.get("title", "")

        if not title or not abstract:
            continue

        # Extract authors
        authors_list = [a.get("name", "") for a in paper.get("authors", [])[:5]]
        authors = ", ".join(authors_list)
        if len(paper.get("authors", [])) > 5:
            authors += " et al."

        # Get DOI from external IDs
        external_ids = paper.get("externalIds", {})
        doi = external_ids.get("DOI", "")
        pmid = external_ids.get("PubMed", "")

        articles.append({
            "paper_id": paper.get("paperId", ""),
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": str(paper.get("year", "")),
            "journal": paper.get("venue", ""),
            "doi": doi,
            "citations": paper.get("citationCount", 0),
            "content": f"{title}\n\n{abstract}",
            "source": "SemanticScholar"
        })

    print(f"[SemanticScholar] Search '{query}' returned {len(articles)} results")
    return articles


# ============================================================================
# CrossRef API
# ============================================================================
CROSSREF_BASE_URL = "https://api.crossref.org"


def search_crossref(query: str, max_results: int = 20) -> List[Dict]:
    """Search CrossRef for academic papers by DOI and metadata.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of article dictionaries
    """
    search_url = f"{CROSSREF_BASE_URL}/works"
    params = {
        "query": query,
        "rows": max_results,
        "filter": "type:journal-article",
        "select": "DOI,title,author,published-print,container-title,abstract"
    }

    try:
        response = requests.get(
            search_url,
            params=params,
            timeout=30,
            headers={"User-Agent": "MediAssist/1.0 (mailto:contact@example.com)"}
        )
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("message", {}).get("items", []):
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            abstract = item.get("abstract", "")

            if not title:
                continue

            # Clean abstract (remove HTML/XML tags)
            if abstract:
                import re
                abstract = re.sub(r'<[^>]+>', '', abstract)

            # Extract authors
            authors_list = []
            for author in item.get("author", [])[:5]:
                name = f"{author.get('family', '')} {author.get('given', '')}".strip()
                if name:
                    authors_list.append(name)
            authors = ", ".join(authors_list)
            if len(item.get("author", [])) > 5:
                authors += " et al."

            # Extract year
            pub_date = item.get("published-print", {}).get("date-parts", [[]])
            year = str(pub_date[0][0]) if pub_date and pub_date[0] else ""

            # Journal
            container = item.get("container-title", [])
            journal = container[0] if container else ""

            articles.append({
                "doi": item.get("DOI", ""),
                "pmid": "",
                "title": title,
                "abstract": abstract if abstract else f"[Abstract not available] {title}",
                "authors": authors,
                "year": year,
                "journal": journal,
                "content": f"{title}\n\n{abstract}" if abstract else title,
                "source": "CrossRef"
            })

        print(f"[CrossRef] Search '{query}' returned {len(articles)} results")
        return articles

    except Exception as e:
        print(f"[CrossRef] Search error: {e}")
        return []


# ============================================================================
# OpenAlex API
# ============================================================================
OPENALEX_BASE_URL = "https://api.openalex.org"


def search_openalex(query: str, max_results: int = 20) -> List[Dict]:
    """Search OpenAlex for academic papers.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of article dictionaries
    """
    search_url = f"{OPENALEX_BASE_URL}/works"
    params = {
        "search": query,
        "per_page": max_results,
        "filter": "type:article",  # Fixed: use 'article' not 'journal-article'
        "select": "id,doi,title,authorships,publication_year,primary_location,abstract_inverted_index,cited_by_count"
    }

    try:
        response = requests.get(
            search_url,
            params=params,
            timeout=30,
            headers={"User-Agent": "MediAssist/1.0 (mailto:contact@example.com)"}
        )
        response.raise_for_status()
        data = response.json()

        articles = []
        for work in data.get("results", []):
            title = work.get("title", "")

            # Reconstruct abstract from inverted index
            abstract = ""
            abstract_inv = work.get("abstract_inverted_index", {})
            if abstract_inv:
                # Reconstruct text from inverted index
                word_positions = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join([w for _, w in word_positions])

            if not title:
                continue

            # Extract authors
            authors_list = []
            for authorship in work.get("authorships", [])[:5]:
                author_name = authorship.get("author", {}).get("display_name", "")
                if author_name:
                    authors_list.append(author_name)
            authors = ", ".join(authors_list)
            if len(work.get("authorships", [])) > 5:
                authors += " et al."

            # Journal from primary location
            primary_loc = work.get("primary_location", {}) or {}
            source = primary_loc.get("source", {}) or {}
            journal = source.get("display_name", "")

            # Extract DOI
            doi = work.get("doi", "")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")

            articles.append({
                "openalex_id": work.get("id", ""),
                "doi": doi,
                "pmid": "",
                "title": title,
                "abstract": abstract if abstract else f"[Abstract not available] {title}",
                "authors": authors,
                "year": str(work.get("publication_year", "")),
                "journal": journal,
                "citations": work.get("cited_by_count", 0),
                "content": f"{title}\n\n{abstract}" if abstract else title,
                "source": "OpenAlex"
            })

        print(f"[OpenAlex] Search '{query}' returned {len(articles)} results")
        return articles

    except Exception as e:
        print(f"[OpenAlex] Search error: {e}")
        return []


# ============================================================================
# KoreaMed API (Korean Medical Literature)
# ============================================================================
KOREAMED_BASE_URL = "https://koreamed.org"


def search_koreamed(query: str, max_results: int = 20) -> List[Dict]:
    """Search for Korean medical literature via PubMed.

    Note: KoreaMed doesn't have a public API, so we use PubMed's E-utilities
    to search for Korean medical journals indexed in PubMed.

    Args:
        query: Search query (Korean or English)
        max_results: Maximum number of results

    Returns:
        List of article dictionaries
    """
    # Search PubMed for Korean journals using affiliation filter
    # Korean journals in PubMed often have Korean author affiliations
    search_query = f"({query}) AND (Korea[Affiliation] OR Korean[Title])"

    try:
        # Use PubMed E-utilities to search
        search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": search_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        search_data = search_response.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            print(f"[KoreaMed] No results found for query: {query}")
            return []

        # Fetch article details
        fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml"
        }

        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()

        articles = []
        root = ET.fromstring(fetch_response.content)

        for article_elem in root.findall(".//PubmedArticle"):
            medline = article_elem.find(".//MedlineCitation")
            if medline is None:
                continue

            pmid = medline.findtext(".//PMID", "")
            article = medline.find(".//Article")
            if article is None:
                continue

            title = article.findtext(".//ArticleTitle", "")
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""

            # Authors
            authors_list = []
            for author in article.findall(".//Author")[:5]:
                lastname = author.findtext("LastName", "")
                forename = author.findtext("ForeName", "")
                if lastname:
                    authors_list.append(f"{lastname} {forename}".strip())
            authors = ", ".join(authors_list)
            if len(article.findall(".//Author")) > 5:
                authors += " et al."

            # Journal and year
            journal_elem = article.find(".//Journal")
            journal = journal_elem.findtext(".//Title", "") if journal_elem else ""
            year = ""
            pub_date = journal_elem.find(".//PubDate") if journal_elem else None
            if pub_date is not None:
                year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4] if pub_date.findtext("MedlineDate") else ""

            # DOI
            doi = ""
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            if title:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract if abstract else f"[초록 없음] {title}",
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "doi": doi,
                    "keywords": [],
                    "content": f"{title}\n\n{abstract}" if abstract else title,
                    "source": "KoreaMed"
                })

        print(f"[KoreaMed] Search '{query}' returned {len(articles)} results")
        return articles

    except Exception as e:
        print(f"[KoreaMed] Search error: {e}")
        return []


# ============================================================================
# Unified Academic Search
# ============================================================================
AVAILABLE_SOURCES = {
    "pubmed": {
        "name": "PubMed",
        "description": "NCBI 의학 논문 데이터베이스",
        "search_fn": search_and_fetch_pubmed
    },
    "semantic_scholar": {
        "name": "Semantic Scholar",
        "description": "AI 기반 학술 검색 (Rate limit 주의)",
        "search_fn": search_semantic_scholar
    },
    "crossref": {
        "name": "CrossRef",
        "description": "DOI 기반 학술 메타데이터",
        "search_fn": search_crossref
    },
    "openalex": {
        "name": "OpenAlex",
        "description": "2억+ 오픈 학술 자료",
        "search_fn": search_openalex
    },
    "koreamed": {
        "name": "KoreaMed",
        "description": "한국 의학 논문 데이터베이스",
        "search_fn": search_koreamed
    }
}


def unified_academic_search(
    query: str,
    sources: List[str] = None,
    max_results_per_source: int = 10
) -> Dict[str, List[Dict]]:
    """Search multiple academic sources and combine results.

    Args:
        query: Search query
        sources: List of source names to search (default: all)
        max_results_per_source: Max results per source

    Returns:
        Dictionary with results from each source
    """
    if sources is None:
        sources = list(AVAILABLE_SOURCES.keys())

    results = {}
    for source in sources:
        if source.lower() in AVAILABLE_SOURCES:
            source_info = AVAILABLE_SOURCES[source.lower()]
            try:
                source_results = source_info["search_fn"](query, max_results_per_source)
                results[source] = source_results
            except Exception as e:
                print(f"[UnifiedSearch] Error searching {source}: {e}")
                results[source] = []

    return results


def get_neo4j_driver():
    """Get or create Neo4j driver for knowledge graph."""
    global _neo4j_driver
    try:
        if _neo4j_driver is None:
            print(f"[Neo4j RAG] Connecting to {settings.NEO4J_URI}...")
            _neo4j_driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
        _neo4j_driver.verify_connectivity()
        print(f"[Neo4j RAG] Connected successfully")
        return _neo4j_driver
    except Exception as e:
        print(f"[Neo4j RAG] Connection failed: {e}")
        _neo4j_driver = None
        return None


def extract_search_terms(query: str) -> List[str]:
    """Extract meaningful search terms from a query.

    Filters out common Korean particles and strips suffixes.
    """
    import re

    # Korean particles (standalone)
    stop_words = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로',
                  '에서', '한', '된', '하는', '있는', '없는', '같은', '대한', '위한', '위해',
                  '어떤', '무엇', '무엇인가요', '무엇인가', '어떻게', '왜', '뭐', '어디'}

    # Korean suffixes to strip from words (attached particles)
    suffixes = ['의', '은', '는', '이', '가', '을', '를', '에', '와', '과', '도', '로', '으로',
                '에서', '에게', '한테', '께서', '라고', '이라고', '처럼', '같이', '만큼',
                '인가요', '인가', '입니까', '습니까', '나요', '인지', '면', '하면']

    # Split by spaces and common punctuation
    words = re.split(r'[\s,?!。，？！]+', query)

    terms = []
    for word in words:
        if not word or len(word) < 2:
            continue
        if word in stop_words:
            continue

        # Try to strip Korean suffixes
        cleaned = word
        for suffix in sorted(suffixes, key=len, reverse=True):  # Try longer suffixes first
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
                cleaned = cleaned[:-len(suffix)]
                break

        if len(cleaned) >= 2 and cleaned not in stop_words:
            terms.append(cleaned)

    return terms


def search_knowledge_graph(query: str, limit: int = 10) -> Dict:
    """Search the Neo4j knowledge graph for relevant medical entities.

    Args:
        query: Search term (disease, symptom, treatment, etc.)
        limit: Maximum number of results

    Returns:
        Dictionary with nodes, edges, and formatted context
    """
    # Extract individual search terms
    search_terms = extract_search_terms(query)
    print(f"[Neo4j RAG] Search terms: {search_terms}")

    if not search_terms:
        return {"nodes": [], "edges": [], "context": ""}

    driver = get_neo4j_driver()
    if not driver:
        print("[Neo4j RAG] No driver available")
        return {"nodes": [], "edges": [], "context": ""}

    try:
        with driver.session() as session:
            # Search for nodes matching any of the search terms
            # Build dynamic WHERE clause for multiple terms
            where_conditions = []
            params = {"max_results": limit}

            for i, term in enumerate(search_terms[:5]):  # Limit to first 5 terms
                param_name = f"term{i}"
                params[param_name] = term
                where_conditions.append(f"""
                    (toLower(coalesce(n.name, '')) CONTAINS toLower(${param_name})
                     OR toLower(coalesce(n.name_en, '')) CONTAINS toLower(${param_name})
                     OR toLower(coalesce(n.description, '')) CONTAINS toLower(${param_name}))
                """)

            where_clause = " OR ".join(where_conditions)

            cypher = f"""
            MATCH (n)
            WHERE {where_clause}
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
            LIMIT $max_results
            """

            result = session.run(cypher, params)
            records = list(result)
            print(f"[Neo4j RAG] Found {len(records)} records")

            nodes_dict = {}
            edges = []

            for record in records:
                n = record.get("n")
                if n:
                    node_id = str(n.element_id)
                    labels = record.get("n_labels", [])
                    if node_id not in nodes_dict:
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "label": n.get("name", n.get("title", labels[0] if labels else "Unknown")),
                            "type": labels[0] if labels else "Unknown",
                            "properties": dict(n)
                        }

                m = record.get("m")
                if m:
                    node_id = str(m.element_id)
                    labels = record.get("m_labels", [])
                    if node_id not in nodes_dict:
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "label": m.get("name", m.get("title", labels[0] if labels else "Unknown")),
                            "type": labels[0] if labels else "Unknown",
                            "properties": dict(m)
                        }

                r = record.get("r")
                rel_type = record.get("rel_type")
                if r and n and m and rel_type:
                    edges.append({
                        "source": str(n.element_id),
                        "target": str(m.element_id),
                        "type": rel_type,
                        "properties": dict(r)
                    })

            # Generate context text from graph data
            context = _format_graph_context(list(nodes_dict.values()), edges)

            result_data = {
                "nodes": list(nodes_dict.values()),
                "edges": edges,
                "context": context
            }
            print(f"[Neo4j RAG] Returning {len(result_data['nodes'])} nodes, {len(result_data['edges'])} edges")
            return result_data

    except Exception as e:
        print(f"[Neo4j RAG] Search error: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": [], "context": ""}


def _format_graph_context(nodes: List[Dict], edges: List[Dict]) -> str:
    """Format graph data into readable context for RAG."""
    if not nodes:
        return ""

    context_parts = []

    # Group nodes by type
    nodes_by_type = {}
    for node in nodes:
        node_type = node.get("type", "Unknown")
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)

    # Format each node type
    type_labels = {
        "Disease": "질병",
        "Symptom": "증상",
        "Treatment": "치료법",
        "Drug": "약물",
        "DiagnosticTest": "진단검사",
        "ImageFinding": "영상소견",
        "RiskLevel": "위험도"
    }

    for node_type, type_nodes in nodes_by_type.items():
        type_label = type_labels.get(node_type, node_type)
        names = [n.get("label", "") for n in type_nodes if n.get("label")]
        if names:
            context_parts.append(f"【{type_label}】: {', '.join(names)}")

    # Format relationships
    rel_labels = {
        "HAS_SYMPTOM": "증상",
        "TREATED_BY": "치료",
        "USES_DRUG": "약물 사용",
        "DIAGNOSED_BY": "진단",
        "HAS_RISK_LEVEL": "위험도",
        "INDICATES": "징후",
        "DETECTS": "발견"
    }

    relationships = []
    for edge in edges:
        rel_type = edge.get("type", "")
        rel_label = rel_labels.get(rel_type, rel_type)

        # Find source and target node names
        source_node = next((n for n in nodes if n.get("id") == edge.get("source")), None)
        target_node = next((n for n in nodes if n.get("id") == edge.get("target")), None)

        if source_node and target_node:
            source_name = source_node.get("label", "")
            target_name = target_node.get("label", "")

            # Add probability if available
            prob = edge.get("properties", {}).get("probability")
            if prob:
                relationships.append(f"{source_name} → {target_name} ({rel_label}, {prob}%)")
            else:
                relationships.append(f"{source_name} → {target_name} ({rel_label})")

    if relationships:
        context_parts.append(f"【관계】: {'; '.join(relationships[:10])}")  # Limit to 10 relationships

    return "\n".join(context_parts)

# Collection name
COLLECTION_NAME = "medical_literature_hybrid"

# Sample medical knowledge base for demo
SAMPLE_MEDICAL_DOCUMENTS = [
    {
        "content": """폐렴(Pneumonia)은 폐 실질 조직의 염증성 질환입니다.
        원인에 따라 세균성, 바이러스성, 진균성, 비정형 폐렴으로 분류됩니다.
        주요 증상으로는 발열, 기침, 가래, 호흡곤란, 흉통 등이 있습니다.
        치료는 원인균에 따라 적절한 항생제를 선택하며,
        중증의 경우 입원 치료와 산소 요법이 필요할 수 있습니다.""",
        "title": "폐렴의 진단과 치료",
        "authors": "대한호흡기학회",
        "year": "2024",
        "journal": "대한내과학회지",
        "pmid": "DEMO001"
    },
    {
        "content": """지역사회획득 폐렴(Community-Acquired Pneumonia, CAP)의 치료 지침:
        1. 경증 외래 환자: 아목시실린 또는 마크로라이드 단독 요법
        2. 중등도 입원 환자: 베타락탐 + 마크로라이드 병용 또는 호흡기 퀴놀론 단독
        3. 중증/ICU 환자: 베타락탐 + 마크로라이드 또는 베타락탐 + 호흡기 퀴놀론
        치료 기간은 일반적으로 5-7일이며, 임상 반응에 따라 조절합니다.""",
        "title": "지역사회획득 폐렴 항생제 치료 가이드라인",
        "authors": "대한감염학회",
        "year": "2023",
        "journal": "감염과 화학요법",
        "pmid": "DEMO002"
    },
    {
        "content": """인플루엔자(독감)는 인플루엔자 바이러스에 의한 급성 호흡기 감염질환입니다.
        A형과 B형이 주로 유행하며, 갑작스러운 고열(38°C 이상),
        두통, 근육통, 피로감이 특징적입니다.
        치료에는 오셀타미비르(타미플루)가 사용되며,
        증상 발현 48시간 이내에 투여 시 가장 효과적입니다.
        고위험군(65세 이상, 만성질환자, 임산부)은
        매년 예방접종이 권장됩니다.""",
        "title": "인플루엔자 감염의 진단과 치료",
        "authors": "질병관리청",
        "year": "2024",
        "journal": "주간 건강과 질병",
        "pmid": "DEMO003"
    },
    {
        "content": """급성 상기도 감염(감기)은 바이러스에 의한 자가 제한적 질환입니다.
        리노바이러스가 가장 흔한 원인이며, 콧물, 코막힘, 인후통,
        기침, 미열 등의 증상이 나타납니다.
        항생제는 세균성 합병증이 없는 한 권장되지 않습니다.
        증상 완화를 위해 충분한 휴식, 수분 섭취,
        해열진통제 등의 대증 치료가 권장됩니다.""",
        "title": "급성 상기도 감염의 관리",
        "authors": "대한가정의학회",
        "year": "2023",
        "journal": "가정의학회지",
        "pmid": "DEMO004"
    },
    {
        "content": """흉부 X-ray는 폐렴 진단의 기본 검사입니다.
        폐렴 시 폐포 침윤, 기관지 공기음영, 늑막삼출 등이 관찰됩니다.
        CT는 X-ray에서 불명확한 경우나 합병증 평가에 유용합니다.
        영상 소견만으로 원인균을 특정하기는 어려우며,
        객담 배양, 혈액 배양, 소변 항원 검사 등을
        통해 원인균을 확인해야 합니다.""",
        "title": "폐렴의 영상 진단",
        "authors": "대한영상의학회",
        "year": "2024",
        "journal": "대한영상의학회지",
        "pmid": "DEMO005"
    },
    {
        "content": """고혈압(Hypertension)은 수축기 혈압 140mmHg 이상 또는
        이완기 혈압 90mmHg 이상인 상태입니다.
        1차 약물 치료로 ACE억제제, ARB, 칼슘채널차단제,
        티아지드계 이뇨제가 권장됩니다.
        생활습관 개선이 중요하며, 염분 섭취 제한(하루 6g 이하),
        규칙적인 운동, 체중 관리, 금연이 필요합니다.""",
        "title": "고혈압 치료 가이드라인",
        "authors": "대한고혈압학회",
        "year": "2023",
        "journal": "Korean Circulation Journal",
        "pmid": "DEMO006"
    },
    {
        "content": """당뇨병 환자의 폐렴은 일반인에 비해
        발생 위험이 1.5-2배 높고 예후도 불량합니다.
        혈당 조절이 불량한 경우 면역 기능 저하로
        감염에 취약해지며, 폐렴 발생 시 입원율과 사망률이 증가합니다.
        당뇨 환자는 폐렴구균 백신과 인플루엔자 백신
        접종이 강력히 권장됩니다.""",
        "title": "당뇨병 환자의 호흡기 감염 관리",
        "authors": "대한당뇨병학회",
        "year": "2024",
        "journal": "Diabetes & Metabolism Journal",
        "pmid": "DEMO007"
    },
    {
        "content": """CURB-65 점수는 폐렴의 중증도 평가 도구입니다.
        Confusion(의식혼란), Urea(혈중요소질소 >7mmol/L),
        Respiratory rate(호흡수 ≥30/분), Blood pressure(혈압 <90/60mmHg),
        65세 이상 연령 각각 1점으로 총 5점입니다.
        0-1점: 외래 치료, 2점: 단기 입원 고려,
        3점 이상: 입원 치료, 4-5점: 중환자실 고려.""",
        "title": "폐렴 중증도 평가 CURB-65",
        "authors": "대한중환자의학회",
        "year": "2023",
        "journal": "대한중환자의학회지",
        "pmid": "DEMO008"
    },
]


def get_qdrant_client() -> QdrantClient:
    """Get or initialize Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

        try:
            _qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            # Test connection
            _qdrant_client.get_collections()
            print(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            # Fallback to in-memory for demo
            print(f"Qdrant connection failed ({e}), using in-memory storage")
            _qdrant_client = QdrantClient(":memory:")

    return _qdrant_client


def get_dense_embedder():
    """Get or initialize the dense embedder (BioBERT)."""
    global _dense_embedder
    if _dense_embedder is None:
        try:
            from rag.embedding.embedder import BioBERTEmbedder
            _dense_embedder = BioBERTEmbedder()
        except ImportError:
            # Fallback: use sentence-transformers
            from sentence_transformers import SentenceTransformer
            _dense_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _dense_embedder


def get_sparse_embedder():
    """Get or initialize the BM25 sparse embedder for multilingual support."""
    global _sparse_embedder
    if _sparse_embedder is None:
        try:
            from fastembed import SparseTextEmbedding
            # Use Qdrant/bm25 for better multilingual (Korean) support
            _sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
            print("[Sparse] Using Qdrant/bm25 model for multilingual support")
        except Exception as e:
            print(f"Sparse embedder initialization failed: {e}")
            _sparse_embedder = None
    return _sparse_embedder


def embed_dense(text: str) -> List[float]:
    """Generate dense embedding for text."""
    embedder = get_dense_embedder()

    if hasattr(embedder, 'embed'):
        # BioBERTEmbedder
        return embedder.embed(text).tolist()
    else:
        # SentenceTransformer
        return embedder.encode(text).tolist()


def tokenize_korean(text: str) -> str:
    """Tokenize Korean text by adding spaces between characters for better BM25 matching.

    This helps BM25 match partial Korean words.
    """
    import re

    # Check if text contains Korean
    if not re.search(r'[가-힣]', text):
        return text

    # Split Korean text into individual morpheme-like units
    # Add spaces around Korean character sequences
    result = []
    current_word = ""

    for char in text:
        if '가' <= char <= '힣':
            current_word += char
        else:
            if current_word:
                # Add the Korean word with character-level tokens
                # Also keep the full word for exact matches
                if len(current_word) > 1:
                    result.append(current_word)  # Full word
                    # Add 2-gram subwords for better matching
                    for i in range(len(current_word) - 1):
                        result.append(current_word[i:i+2])
                else:
                    result.append(current_word)
                current_word = ""
            if char.strip():
                result.append(char)

    if current_word:
        if len(current_word) > 1:
            result.append(current_word)
            for i in range(len(current_word) - 1):
                result.append(current_word[i:i+2])
        else:
            result.append(current_word)

    tokenized = " ".join(result)
    print(f"[Korean Tokenizer] '{text}' -> '{tokenized}'")
    return tokenized


def embed_sparse(text: str) -> Tuple[List[int], List[float]]:
    """Generate sparse embedding for text with Korean support.

    Uses custom tokenization for Korean text (2-gram based) since BM25
    doesn't properly support Korean character tokenization.

    Returns:
        Tuple of (indices, values) for sparse vector
    """
    import re

    # Check if text contains Korean characters
    has_korean = bool(re.search(r'[가-힣]', text))

    if has_korean:
        # Use custom Korean tokenizer for better matching
        print(f"[Sparse] Using Korean tokenizer for text with Korean characters")
        return _simple_sparse_embed(text)

    # For English-only text, try BM25 first
    embedder = get_sparse_embedder()

    if embedder is not None:
        try:
            embeddings = list(embedder.embed([text]))
            if embeddings:
                sparse = embeddings[0]
                indices = sparse.indices.tolist()
                values = sparse.values.tolist()
                print(f"[BM25] Generated {len(indices)} sparse indices from text")
                return indices, values
        except Exception as e:
            print(f"BM25 embedding failed: {e}, falling back to simple tokenizer")

    # Fallback to simple tokenizer
    return _simple_sparse_embed(text)


def _deterministic_hash(text: str) -> int:
    """Generate a deterministic hash that's consistent across Python processes."""
    # Use MD5 for consistent hashing (not for security, just for consistency)
    hash_bytes = hashlib.md5(text.encode('utf-8')).digest()
    # Convert first 4 bytes to integer (max int32 range)
    return int.from_bytes(hash_bytes[:4], byteorder='big') % (2**31 - 1)


def _simple_sparse_embed(text: str) -> Tuple[List[int], List[float]]:
    """Simple sparse embedding fallback using term frequency with Korean support."""
    from collections import Counter
    import re

    # Tokenize text - split Korean into characters/morphemes and English into words
    korean_pattern = re.compile(r'[가-힣]+')
    english_pattern = re.compile(r'[a-zA-Z]+')

    tokens = []

    # Extract Korean tokens - use 2-gram for better matching
    korean_matches = korean_pattern.findall(text)
    for match in korean_matches:
        # Add full word
        tokens.append(match)
        # Add 2-grams for partial matching (important for Korean)
        if len(match) >= 2:
            for i in range(len(match) - 1):
                tokens.append(match[i:i+2])

    # Extract English tokens
    english_matches = english_pattern.findall(text.lower())
    tokens.extend(english_matches)

    token_counts = Counter(tokens)

    # Create sparse vector with hash-based indices
    indices = []
    values = []

    # Use larger vocabulary space for better distribution
    VOCAB_SIZE = 2**31 - 1  # Max int32

    for token, count in token_counts.items():
        # Use deterministic hash for consistent indexing across processes
        idx = _deterministic_hash(token)
        indices.append(idx)
        # TF-IDF like weighting
        values.append(float(1 + np.log(count)))

    print(f"[Korean Sparse] Generated {len(indices)} indices from {len(tokens)} tokens")
    return indices, values


def ensure_collection_exists():
    """Ensure the Qdrant collection exists with proper configuration."""
    client = get_qdrant_client()

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        # Get dense embedding dimension
        sample_dense = embed_dense("test")
        dense_dim = len(sample_dense)

        # Create collection with hybrid vector configuration
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        print(f"Created collection '{COLLECTION_NAME}' with dense dim={dense_dim}")


def generate_doc_id(content: str) -> str:
    """Generate unique document ID from content as UUID."""
    # Create a UUID from MD5 hash of content
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return str(uuid.UUID(content_hash))


def hybrid_search(
    query: str,
    top_k: int = 5,
    sparse_weight: float = SPARSE_WEIGHT,
    dense_weight: float = DENSE_WEIGHT
) -> List[Dict]:
    """Perform hybrid search with SPLADE (sparse) and dense embeddings.

    Args:
        query: Search query text
        top_k: Number of results to return
        sparse_weight: Weight for sparse (SPLADE) scores (default 0.3)
        dense_weight: Weight for dense embedding scores (default 0.7)

    Returns:
        List of search results with hybrid scores
    """
    client = get_qdrant_client()
    ensure_collection_exists()

    # Generate embeddings
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)

    # Separate searches for sparse and dense to get individual scores
    try:
        # Dense search using query_points
        dense_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            limit=top_k * 2,  # Get more for fusion
            with_payload=True,
        ).points

        # Sparse search
        sparse_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using="sparse",
            limit=top_k * 2,
            with_payload=True,
        ).points

        # Debug logging
        print(f"[Hybrid Search] Query: {query[:30]}...")
        print(f"[Hybrid Search] Sparse indices count: {len(sparse_indices)}, values count: {len(sparse_values)}")
        print(f"[Hybrid Search] Dense results: {len(dense_results)}, Sparse results: {len(sparse_results)}")
        if sparse_results:
            print(f"[Hybrid Search] Sparse result IDs: {[str(r.id)[:8] for r in sparse_results[:3]]}")
            print(f"[Hybrid Search] Sparse scores: {[r.score for r in sparse_results[:3]]}")
        else:
            print("[Hybrid Search] WARNING: No sparse results found!")

    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        # Return empty if search fails
        return []

    # Build score maps
    dense_scores = {str(r.id): r.score for r in dense_results}
    sparse_scores = {str(r.id): r.score for r in sparse_results}

    # Normalize scores to 0-1 range
    max_dense = max(dense_scores.values()) if dense_scores else 1.0
    max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0

    if max_dense > 0:
        dense_scores = {k: v / max_dense for k, v in dense_scores.items()}
    if max_sparse > 0:
        sparse_scores = {k: v / max_sparse for k, v in sparse_scores.items()}

    # Combine all document IDs
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

    # Calculate hybrid scores
    results = []
    for doc_id in all_ids:
        d_score = dense_scores.get(doc_id, 0.0)
        s_score = sparse_scores.get(doc_id, 0.0)
        combined = (sparse_weight * s_score) + (dense_weight * d_score)

        # Get payload from either result set
        payload = None
        for r in dense_results + sparse_results:
            if str(r.id) == doc_id:
                payload = r.payload
                break

        if payload:
            # Clamp scores to be >= 0 (negative scores can occur with dissimilar vectors)
            results.append({
                "id": doc_id,
                "content": payload.get("content", ""),
                "metadata": {
                    "title": payload.get("title", ""),
                    "authors": payload.get("authors", ""),
                    "year": payload.get("year", ""),
                    "journal": payload.get("journal", ""),
                    "pmid": payload.get("pmid", ""),
                },
                "sparse_score": max(0.0, s_score),
                "dense_score": max(0.0, d_score),
                "combined_score": max(0.0, combined),
            })

    # Sort by combined score and return top_k
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:top_k]


@router.post("/search", response_model=LiteratureSearchResponse)
async def search_literature(request: LiteratureSearchRequest):
    """
    Search medical literature database using hybrid search.

    Uses SPLADE sparse embeddings (30%) + Dense embeddings (70%)
    for improved semantic and keyword search.
    """
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        # Check if collection has documents
        collection_info = client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            await load_sample_documents()

        # Perform hybrid search
        results = hybrid_search(request.query, request.top_k)

        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                content=result["content"],
                metadata=result["metadata"],
                score=result["combined_score"],
                hybrid_score=HybridScore(
                    sparse_score=result["sparse_score"],
                    dense_score=result["dense_score"],
                    combined_score=result["combined_score"],
                )
            ))

        return LiteratureSearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Answer medical questions using RAG with hybrid search + Knowledge Graph.

    Uses SPLADE (30%) + Dense (70%) hybrid retrieval
    for finding relevant medical literature, plus Neo4j knowledge graph
    for structured medical entity relationships.
    """
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        # Check if collection has documents
        collection_info = client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            await load_sample_documents()

        # Expand query with context
        expanded_query = request.question
        if request.context:
            if request.context.get("symptoms"):
                symptoms = request.context["symptoms"]
                if isinstance(symptoms, list):
                    expanded_query += f" 증상: {', '.join(symptoms)}"

        # Perform hybrid search on documents
        results = hybrid_search(expanded_query, top_k=5)

        # Search knowledge graph for related entities
        graph_data = search_knowledge_graph(request.question, limit=15)

        # Build context and sources
        sources = []
        context_parts = []

        for i, result in enumerate(results):
            context_parts.append(f"[{i+1}] {result['content']}")

            sources.append(DocumentSource(
                title=result["metadata"].get("title", "Unknown"),
                authors=result["metadata"].get("authors"),
                year=result["metadata"].get("year"),
                journal=result["metadata"].get("journal"),
                pmid=result["metadata"].get("pmid"),
                relevance=result["combined_score"],
                sparse_score=result["sparse_score"],
                dense_score=result["dense_score"],
            ))

        # Add knowledge graph context
        if graph_data.get("context"):
            context_parts.append(f"\n[지식 그래프]\n{graph_data['context']}")

        context_text = "\n\n".join(context_parts)

        # Generate answer with graph context
        if not context_parts:
            answer = "관련 문헌을 찾을 수 없습니다."
            confidence = "low"
        else:
            answer = _generate_extractive_answer(
                request.question,
                context_parts,
                sources,
                graph_context=graph_data.get("context", "")
            )
            confidence = "high" if len(sources) >= 3 and sources[0].relevance > 0.7 else "medium"

        # Build knowledge graph response
        kg_response = None
        if graph_data.get("nodes"):
            kg_response = KnowledgeGraphData(
                nodes=[
                    KnowledgeGraphNode(
                        id=n["id"],
                        label=n["label"],
                        type=n["type"],
                        properties=n.get("properties", {})
                    ) for n in graph_data["nodes"]
                ],
                edges=[
                    KnowledgeGraphEdge(
                        source=e["source"],
                        target=e["target"],
                        type=e["type"],
                        properties=e.get("properties", {})
                    ) for e in graph_data["edges"]
                ],
                context=graph_data.get("context", "")
            )

        return RAGQueryResponse(
            answer=answer,
            sources=sources if request.include_sources else [],
            context_used=context_text if request.include_sources else None,
            confidence=confidence,
            knowledge_graph=kg_response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


def _generate_extractive_answer(
    question: str,
    contexts: List[str],
    sources: List[DocumentSource],
    graph_context: str = ""
) -> str:
    """Generate a comprehensive extractive answer from context and knowledge graph."""
    question_lower = question.lower()
    answer_parts = []

    # Header
    answer_parts.append(f"📋 질문: {question}\n")
    answer_parts.append("=" * 50 + "\n")

    # Categorize and extract relevant information
    treatment_info = []
    symptom_info = []
    diagnosis_info = []
    general_info = []

    for i, ctx in enumerate(contexts):
        clean_ctx = ctx
        if ctx.startswith("["):
            clean_ctx = ctx.split("]", 1)[-1].strip()

        # Skip knowledge graph context in categorization
        if clean_ctx.startswith("[지식 그래프]"):
            continue

        if any(keyword in clean_ctx for keyword in ["치료", "약물", "항생제", "투여", "요법"]):
            treatment_info.append((clean_ctx, sources[i] if i < len(sources) else None))
        if any(keyword in clean_ctx for keyword in ["증상", "발열", "기침", "통증", "두통"]):
            symptom_info.append((clean_ctx, sources[i] if i < len(sources) else None))
        if any(keyword in clean_ctx for keyword in ["진단", "검사", "X-ray", "CT", "평가"]):
            diagnosis_info.append((clean_ctx, sources[i] if i < len(sources) else None))
        general_info.append((clean_ctx, sources[i] if i < len(sources) else None))

    # Add Knowledge Graph section if available
    if graph_context:
        answer_parts.append("\n🕸️ 【지식 그래프 정보】\n")
        answer_parts.append(graph_context)
        answer_parts.append("\n")

    # Build comprehensive answer with hybrid scores
    if "치료" in question_lower or "약" in question_lower or treatment_info:
        answer_parts.append("\n🏥 【치료 정보】\n")
        for info, src in treatment_info[:2]:
            answer_parts.append(f"\n{info}\n")
            if src:
                score_info = f"[Sparse: {src.sparse_score:.1%}, Dense: {src.dense_score:.1%}]"
                answer_parts.append(f"   📚 출처: {src.title} ({src.year}) {score_info}\n")

    if "증상" in question_lower or symptom_info:
        answer_parts.append("\n🩺 【증상 정보】\n")
        for info, src in symptom_info[:2]:
            answer_parts.append(f"\n{info}\n")
            if src:
                score_info = f"[Sparse: {src.sparse_score:.1%}, Dense: {src.dense_score:.1%}]"
                answer_parts.append(f"   📚 출처: {src.title} ({src.year}) {score_info}\n")

    if "진단" in question_lower or diagnosis_info:
        answer_parts.append("\n🔬 【진단 정보】\n")
        for info, src in diagnosis_info[:2]:
            answer_parts.append(f"\n{info}\n")
            if src:
                score_info = f"[Sparse: {src.sparse_score:.1%}, Dense: {src.dense_score:.1%}]"
                answer_parts.append(f"   📚 출처: {src.title} ({src.year}) {score_info}\n")

    if len(answer_parts) <= 3:  # Adjusted for graph section
        answer_parts.append("\n📖 【관련 의학 정보】\n")
        for info, src in general_info[:3]:
            answer_parts.append(f"\n{info}\n")
            if src:
                score_info = f"[Sparse: {src.sparse_score:.1%}, Dense: {src.dense_score:.1%}]"
                answer_parts.append(f"   📚 출처: {src.title} ({src.year}) {score_info}\n")

    # Summary section with hybrid scores
    answer_parts.append("\n" + "=" * 50)
    answer_parts.append("\n📌 【참고 문헌 및 유사도 점수】\n")
    answer_parts.append(f"   ※ 검색 방식: Sparse(SPLADE) {SPARSE_WEIGHT:.0%} + Dense(BioBERT) {DENSE_WEIGHT:.0%} + 지식그래프(Neo4j)\n\n")

    for i, src in enumerate(sources[:5], 1):
        answer_parts.append(f"  [{i}] {src.title}")
        if src.authors:
            answer_parts.append(f"      저자: {src.authors}")
        if src.year:
            answer_parts.append(f"      발행연도: {src.year}")
        if src.journal:
            answer_parts.append(f"      저널: {src.journal}")
        if src.sparse_score is not None and src.dense_score is not None:
            answer_parts.append(f"      🔍 Sparse 점수: {src.sparse_score:.1%}")
            answer_parts.append(f"      📊 Dense 점수: {src.dense_score:.1%}")
            answer_parts.append(f"      ⚡ 종합 점수: {src.relevance:.1%}")
        answer_parts.append("")

    return "\n".join(answer_parts)


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_documents(request: DocumentIngestionRequest):
    """
    Add documents to the knowledge base with hybrid embeddings.

    Documents are embedded with both SPLADE (sparse) and dense embeddings.
    """
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        points = []

        for doc in request.documents:
            content = doc.get("content", "")
            if not content:
                continue

            doc_id = generate_doc_id(content)

            # Generate embeddings
            dense_vector = embed_dense(content)
            sparse_indices, sparse_values = embed_sparse(content)

            point = PointStruct(
                id=doc_id,
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    )
                },
                payload={
                    "content": content,
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                    "year": doc.get("year", ""),
                    "source": doc.get("source", ""),
                    "journal": doc.get("journal", ""),
                    "pmid": doc.get("pmid", ""),
                }
            )
            points.append(point)

        if points:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

        return DocumentIngestionResponse(
            chunks_added=len(points),
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/load-sample-data")
async def load_sample_documents():
    """Load sample medical documents into the knowledge base."""
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        points = []

        for doc in SAMPLE_MEDICAL_DOCUMENTS:
            content = doc["content"]
            doc_id = generate_doc_id(content)

            # Check if already exists
            try:
                existing = client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[doc_id],
                )
                if existing:
                    continue
            except Exception:
                pass

            # Generate embeddings
            dense_vector = embed_dense(content)
            sparse_indices, sparse_values = embed_sparse(content)

            point = PointStruct(
                id=doc_id,
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    )
                },
                payload={
                    "content": content,
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                    "year": doc.get("year", ""),
                    "journal": doc.get("journal", ""),
                    "pmid": doc.get("pmid", ""),
                }
            )
            points.append(point)

        if points:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

        collection_info = client.get_collection(COLLECTION_NAME)

        return {
            "status": "success",
            "documents_added": len(points),
            "total_documents": collection_info.points_count,
            "search_method": f"Hybrid (Sparse: {SPARSE_WEIGHT:.0%}, Dense: {DENSE_WEIGHT:.0%})"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample data: {str(e)}")


@router.get("/stats")
async def get_collection_stats():
    """Get statistics about the knowledge base."""
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        collection_info = client.get_collection(COLLECTION_NAME)

        return {
            "collection_name": COLLECTION_NAME,
            "document_count": collection_info.points_count,
            "sample_documents_available": len(SAMPLE_MEDICAL_DOCUMENTS),
            "search_method": "Hybrid Search",
            "sparse_weight": f"{SPARSE_WEIGHT:.0%} (BM25)",
            "dense_weight": f"{DENSE_WEIGHT:.0%} (BioBERT)",
            "vector_config": {
                "dense": {
                    "model": "BioBERT / all-MiniLM-L6-v2",
                    "type": "dense embedding"
                },
                "sparse": {
                    "model": "Qdrant/BM25 (Multilingual)",
                    "type": "sparse embedding"
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/graph")
async def get_vectordb_graph(
    query: Optional[str] = Query(None, description="Optional search query to filter documents"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of documents")
):
    """Get VectorDB documents as a graph for visualization.

    Returns documents as nodes and query-relevance connections as edges.
    When a query is provided, shows each document's relevance to the search query.
    """
    try:
        ensure_collection_exists()
        client = get_qdrant_client()

        # Check if collection has documents
        collection_info = client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            await load_sample_documents()

        nodes = []
        edges = []

        if query:
            # Search for documents matching query
            results = hybrid_search(query, top_k=limit)

            # Add central Query node
            query_node_id = "query_node"
            nodes.append({
                "id": query_node_id,
                "label": f"🔍 {query[:20]}..." if len(query) > 20 else f"🔍 {query}",
                "type": "Query",
                "properties": {
                    "query_text": query,
                    "result_count": len(results)
                }
            })

            for i, result in enumerate(results):
                doc_id = result["id"]
                nodes.append({
                    "id": doc_id,
                    "label": result["metadata"].get("title", f"문서 {i+1}")[:30],
                    "type": "Document",
                    "properties": {
                        "title": result["metadata"].get("title", ""),
                        "authors": result["metadata"].get("authors", ""),
                        "year": result["metadata"].get("year", ""),
                        "journal": result["metadata"].get("journal", ""),
                        "content_preview": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"],
                        "combined_score": result["combined_score"],
                        "sparse_score": result["sparse_score"],
                        "dense_score": result["dense_score"],
                    }
                })

                # Create edge from document to query node showing relevance
                edges.append({
                    "source": doc_id,
                    "target": query_node_id,
                    "type": "RELEVANCE",
                    "properties": {
                        "combined_score": round(result["combined_score"], 3),
                        "sparse_score": round(result["sparse_score"], 3),
                        "dense_score": round(result["dense_score"], 3),
                        "rank": i + 1
                    }
                })
        else:
            # Get all documents without query
            all_points = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]

            for i, point in enumerate(all_points):
                payload = point.payload or {}
                nodes.append({
                    "id": str(point.id),
                    "label": payload.get("title", f"문서 {i+1}")[:30],
                    "type": "Document",
                    "properties": {
                        "title": payload.get("title", ""),
                        "authors": payload.get("authors", ""),
                        "year": payload.get("year", ""),
                        "journal": payload.get("journal", ""),
                        "content_preview": payload.get("content", "")[:100] + "..." if len(payload.get("content", "")) > 100 else payload.get("content", ""),
                    }
                })

            # Create edges based on shared keywords (simple approach)
            for i, node1 in enumerate(nodes):
                content1 = node1["properties"].get("content_preview", "").lower()
                for j, node2 in enumerate(nodes):
                    if i < j:
                        content2 = node2["properties"].get("content_preview", "").lower()

                        # Check for shared medical terms
                        shared_terms = []
                        medical_terms = ["폐렴", "치료", "증상", "항생제", "진단", "감염", "호흡", "발열", "기침"]
                        for term in medical_terms:
                            if term in content1 and term in content2:
                                shared_terms.append(term)

                        if len(shared_terms) >= 2:
                            edges.append({
                                "source": node1["id"],
                                "target": node2["id"],
                                "type": "RELATED",
                                "properties": {
                                    "shared_terms": shared_terms
                                }
                            })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_documents": collection_info.points_count,
            "query": query
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {str(e)}")


# ============================================================================
# Academic Search API Endpoints
# ============================================================================

@router.get("/academic/sources")
async def get_available_sources():
    """Get list of available academic search sources."""
    sources = []
    for source_id, info in AVAILABLE_SOURCES.items():
        sources.append(AcademicSource(
            id=source_id,
            name=info["name"],
            description=info["description"]
        ))
    return {"sources": sources}


@router.post("/academic/search")
async def unified_academic_search_endpoint(request: UnifiedSearchRequest):
    """Search academic literature across multiple sources.

    Searches PubMed, Semantic Scholar, CrossRef, OpenAlex, and/or KoreaMed
    based on the sources parameter. Results can optionally be ingested into VectorDB.
    """
    try:
        # Determine which sources to search
        sources_to_search = request.sources
        if not sources_to_search:
            sources_to_search = list(AVAILABLE_SOURCES.keys())

        # Validate sources
        invalid_sources = [s for s in sources_to_search if s.lower() not in AVAILABLE_SOURCES]
        if invalid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sources: {invalid_sources}. Available: {list(AVAILABLE_SOURCES.keys())}"
            )

        # Perform unified search
        print(f"[AcademicSearch] Searching '{request.query}' in sources: {sources_to_search}")
        raw_results = unified_academic_search(
            query=request.query,
            sources=sources_to_search,
            max_results_per_source=request.max_results_per_source
        )

        # Convert to response format
        formatted_results = {}
        total_found = 0

        for source, articles in raw_results.items():
            formatted_articles = []
            for article in articles:
                formatted_articles.append(PubMedArticle(
                    pmid=article.get("pmid", ""),
                    title=article.get("title", ""),
                    abstract=article.get("abstract", ""),
                    authors=article.get("authors", ""),
                    year=article.get("year", ""),
                    journal=article.get("journal", ""),
                    doi=article.get("doi"),
                    keywords=article.get("keywords", []),
                    source=article.get("source", source)
                ))
            formatted_results[source] = formatted_articles
            total_found += len(formatted_articles)

        # Ingest if requested
        ingested_count = 0
        if request.ingest and total_found > 0:
            try:
                ensure_collection_exists()
                client = get_qdrant_client()
                points = []

                for source, articles in raw_results.items():
                    for article in articles:
                        content = article.get("content", "")
                        if not content or len(content) < 50:
                            continue

                        doc_id = generate_doc_id(content)

                        # Generate embeddings
                        dense_vector = embed_dense(content)
                        sparse_indices, sparse_values = embed_sparse(content)

                        point = PointStruct(
                            id=doc_id,
                            vector={
                                "dense": dense_vector,
                                "sparse": SparseVector(
                                    indices=sparse_indices,
                                    values=sparse_values,
                                )
                            },
                            payload={
                                "content": content,
                                "title": article.get("title", ""),
                                "authors": article.get("authors", ""),
                                "year": article.get("year", ""),
                                "journal": article.get("journal", ""),
                                "pmid": article.get("pmid", ""),
                                "doi": article.get("doi", ""),
                                "source": article.get("source", source),
                            }
                        )
                        points.append(point)

                if points:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )
                    ingested_count = len(points)
                    print(f"[AcademicSearch] Ingested {ingested_count} articles to VectorDB")

            except Exception as e:
                print(f"[AcademicSearch] Ingestion error: {e}")

        return UnifiedSearchResponse(
            query=request.query,
            sources_searched=sources_to_search,
            results=formatted_results,
            total_found=total_found,
            ingested=ingested_count
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Academic search failed: {str(e)}")


@router.post("/academic/pubmed")
async def search_pubmed_endpoint(request: PubMedSearchRequest):
    """Search PubMed for medical literature.

    Uses NCBI E-utilities API to search and fetch article details.
    """
    try:
        articles = search_and_fetch_pubmed(request.query, request.max_results)

        # Convert to response format
        formatted_articles = []
        for article in articles:
            formatted_articles.append(PubMedArticle(
                pmid=article.get("pmid", ""),
                title=article.get("title", ""),
                abstract=article.get("abstract", ""),
                authors=article.get("authors", ""),
                year=article.get("year", ""),
                journal=article.get("journal", ""),
                doi=article.get("doi"),
                keywords=article.get("keywords", []),
                source="PubMed"
            ))

        # Ingest if requested
        ingested_count = 0
        if request.ingest and formatted_articles:
            try:
                ensure_collection_exists()
                client = get_qdrant_client()
                points = []

                for article in articles:
                    content = article.get("content", "")
                    if not content:
                        continue

                    doc_id = generate_doc_id(content)
                    dense_vector = embed_dense(content)
                    sparse_indices, sparse_values = embed_sparse(content)

                    point = PointStruct(
                        id=doc_id,
                        vector={
                            "dense": dense_vector,
                            "sparse": SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            )
                        },
                        payload={
                            "content": content,
                            "title": article.get("title", ""),
                            "authors": article.get("authors", ""),
                            "year": article.get("year", ""),
                            "journal": article.get("journal", ""),
                            "pmid": article.get("pmid", ""),
                            "doi": article.get("doi", ""),
                            "source": "PubMed",
                        }
                    )
                    points.append(point)

                if points:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    ingested_count = len(points)

            except Exception as e:
                print(f"[PubMed] Ingestion error: {e}")

        return PubMedSearchResponse(
            articles=formatted_articles,
            total_found=len(formatted_articles),
            ingested=ingested_count,
            query=request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubMed search failed: {str(e)}")
