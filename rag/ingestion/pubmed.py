"""PubMed document fetcher."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import requests


@dataclass
class PubMedArticle:
    """PubMed article data."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str] = None


class PubMedFetcher:
    """Fetch articles from PubMed API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[str]:
        """Search PubMed and return PMIDs.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            List of PubMed IDs.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch article details by PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of article objects.
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        response.raise_for_status()

        # TODO: Parse XML response
        return []

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[Dict]:
        """Search and fetch articles in one call.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            List of article dictionaries.
        """
        pmids = self.search(query, max_results)
        articles = self.fetch(pmids)

        return [
            {
                "content": f"{a.title}\n\n{a.abstract}",
                "title": a.title,
                "authors": ", ".join(a.authors),
                "year": a.year,
                "source": f"PubMed:{a.pmid}",
            }
            for a in articles
        ]
