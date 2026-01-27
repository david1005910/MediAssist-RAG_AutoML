"""Medical RAG system for literature-based question answering."""

from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


class MedicalRAG:
    """RAG system for medical literature search and question answering."""

    PROMPT_TEMPLATE = """당신은 의료 전문가를 위한 진단 보조 AI입니다.

## 참고 문헌
{context}

## 질문
{question}

## 지침
1. 문헌에 근거하여 답변하세요
2. 불확실한 경우 명시하세요
3. 참고문헌을 [1], [2] 형식으로 인용하세요
4. 이 정보는 참고용임을 명시하세요

## 답변"""

    def __init__(
        self,
        embedding_model: str = "dmis-lab/biobert-v1.1",
        chroma_persist_dir: str = "./chroma_db",
        openai_api_key: Optional[str] = None,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda"},
        )

        self.vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=self.embeddings,
        )

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=openai_api_key,
        ) if openai_api_key else None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents to the vector store.

        Args:
            documents: List of documents with content and metadata.

        Returns:
            Number of chunks added.
        """
        texts = []
        metadatas = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                    "year": doc.get("year", ""),
                })

        self.vectorstore.add_texts(texts, metadatas=metadatas)
        return len(texts)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant documents with reranking.

        Args:
            query: Search query.
            top_k: Number of initial results to retrieve.

        Returns:
            Reranked list of relevant documents.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        if not results:
            return []

        # Rerank with cross-encoder
        pairs = [[query, doc.page_content] for doc, _ in results]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for (doc, _), score in reranked
        ]

    def query(self, question: str, context: Optional[Dict] = None) -> Dict:
        """Answer a medical question using RAG.

        Args:
            question: The question to answer.
            context: Optional context with symptoms or other info.

        Returns:
            Answer with sources.
        """
        # Expand query with context
        expanded = question
        if context and context.get("symptoms"):
            expanded += f" 증상: {', '.join(context['symptoms'])}"

        # Search for relevant documents
        docs = self.search(expanded)

        if not docs:
            return {
                "answer": "관련 문헌을 찾을 수 없습니다.",
                "sources": [],
            }

        # Build context
        context_text = "\n\n".join([
            f"[{i+1}] {d['content']}"
            for i, d in enumerate(docs)
        ])

        if self.llm is None:
            return {
                "answer": "LLM이 구성되지 않았습니다.",
                "context": context_text,
                "sources": [
                    {
                        "title": d["metadata"].get("title"),
                        "authors": d["metadata"].get("authors"),
                        "year": d["metadata"].get("year"),
                        "relevance": d["score"],
                    }
                    for d in docs
                ],
            }

        # Generate answer
        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        chain = prompt | self.llm
        result = chain.invoke({"context": context_text, "question": question})

        return {
            "answer": result.content,
            "sources": [
                {
                    "title": d["metadata"].get("title"),
                    "authors": d["metadata"].get("authors"),
                    "year": d["metadata"].get("year"),
                    "relevance": d["score"],
                }
                for d in docs
            ],
        }
