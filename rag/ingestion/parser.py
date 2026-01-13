"""Document parser for various formats."""

from typing import Dict, Optional
from pathlib import Path
import re


class DocumentParser:
    """Parse documents from various formats."""

    def parse(self, file_path: str) -> Dict:
        """Parse a document file.

        Args:
            file_path: Path to the document.

        Returns:
            Parsed document with content and metadata.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix == ".html":
            return self._parse_html(file_path)
        elif suffix == ".txt":
            return self._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_pdf(self, file_path: str) -> Dict:
        """Parse PDF document."""
        try:
            import pypdf

            reader = pypdf.PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages)

            return {
                "content": self._clean_text(text),
                "title": Path(file_path).stem,
                "source": file_path,
            }
        except ImportError:
            raise ImportError("pypdf is required for PDF parsing")

    def _parse_html(self, file_path: str) -> Dict:
        """Parse HTML document."""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            text = soup.get_text(separator="\n")
            title = soup.title.string if soup.title else Path(file_path).stem

            return {
                "content": self._clean_text(text),
                "title": title,
                "source": file_path,
            }
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML parsing")

    def _parse_text(self, file_path: str) -> Dict:
        """Parse plain text document."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "content": self._clean_text(text),
            "title": Path(file_path).stem,
            "source": file_path,
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
