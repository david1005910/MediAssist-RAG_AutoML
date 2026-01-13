"""Answer generation using LLM."""

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class AnswerGenerator:
    """Generate answers using retrieved context and LLM."""

    MEDICAL_PROMPT = """당신은 의료 전문가를 위한 진단 보조 AI입니다.
아래 참고 문헌을 바탕으로 질문에 답변하세요.

## 참고 문헌
{context}

## 질문
{question}

## 지침
1. 문헌에 근거하여 답변하세요
2. 불확실한 경우 명시적으로 표현하세요
3. 참고문헌을 [1], [2] 형식으로 인용하세요
4. 이 정보는 참고용이며 최종 진단은 의사가 결정해야 함을 명시하세요

## 답변"""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
        ) if api_key else None

        self.prompt = PromptTemplate(
            template=self.MEDICAL_PROMPT,
            input_variables=["context", "question"],
        )

    def generate(
        self,
        question: str,
        documents: List[Dict],
    ) -> Dict:
        """Generate an answer based on retrieved documents.

        Args:
            question: The user's question.
            documents: List of retrieved documents with content and metadata.

        Returns:
            Generated answer with citations.
        """
        if not self.llm:
            return {
                "answer": "LLM이 구성되지 않았습니다.",
                "sources": [],
            }

        # Format context
        context = self._format_context(documents)

        # Generate
        chain = self.prompt | self.llm
        result = chain.invoke({
            "context": context,
            "question": question,
        })

        # Extract citations
        citations = self._extract_citations(result.content, documents)

        return {
            "answer": result.content,
            "sources": citations,
        }

    def _format_context(self, documents: List[Dict]) -> str:
        """Format documents as numbered context."""
        parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", "Unknown")
            content = doc.get("content", "")
            parts.append(f"[{i}] {title}\n{content}")
        return "\n\n".join(parts)

    def _extract_citations(
        self,
        answer: str,
        documents: List[Dict],
    ) -> List[Dict]:
        """Extract cited sources from the answer."""
        citations = []
        for i, doc in enumerate(documents, 1):
            if f"[{i}]" in answer:
                metadata = doc.get("metadata", {})
                citations.append({
                    "index": i,
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors"),
                    "year": metadata.get("year"),
                })
        return citations
