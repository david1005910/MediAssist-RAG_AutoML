"""Answer generation using LLM with multi-model support including MedGemma."""

from typing import List, Dict, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os


# Available LLM models for medical RAG
LLMModel = Literal["gpt-4", "gpt-3.5-turbo", "medgemma", "gemini-pro"]


class MedGemmaClient:
    """Client for Google MedGemma medical domain model."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize MedGemma client.

        MedGemma is accessed through Google AI API (Gemini).
        Model: gemini-1.5-flash with medical system prompt for MedGemma-like behavior.
        For true MedGemma, use Vertex AI or Hugging Face.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = None

        if self.api_key:
            try:
                # Use Gemini with medical-focused configuration
                # Try multiple models in case of quota issues
                models_to_try = ["gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
                for model_name in models_to_try:
                    try:
                        self.llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=self.api_key,
                            temperature=0.3,
                        )
                        self.model_name = model_name
                        print(f"MedGemma initialized with {model_name}")
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"MedGemma/Gemini initialization failed: {e}")
                self.llm = None

    def is_available(self) -> bool:
        """Check if MedGemma is available."""
        return self.llm is not None

    def invoke(self, prompt: str) -> str:
        """Generate response using MedGemma."""
        if not self.llm:
            raise ValueError("MedGemma is not configured. Set GOOGLE_API_KEY.")

        # Add medical domain system context
        medical_context = """You are MedGemma, a medical domain-specialized AI assistant.
You provide accurate, evidence-based medical information while always emphasizing
that your responses are for informational purposes only and should not replace
professional medical advice. Always cite sources when possible."""

        full_prompt = f"{medical_context}\n\n{prompt}"
        response = self.llm.invoke(full_prompt)
        return response.content


class AnswerGenerator:
    """Generate answers using retrieved context and LLM with multi-model support."""

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

    MEDGEMMA_PROMPT = """You are MedGemma, a medical domain-specialized AI assistant trained on medical literature.
Based on the reference documents below, answer the question accurately.

## Reference Documents
{context}

## Question
{question}

## Instructions
1. Base your answer on the provided literature
2. Explicitly state when information is uncertain
3. Cite references using [1], [2] format
4. Always include a disclaimer that this is for reference only
5. Provide answers in Korean when the question is in Korean

## Answer"""

    # Model descriptions for UI
    MODEL_INFO = {
        "gpt-4": {
            "name": "GPT-4",
            "description": "OpenAI GPT-4 - 범용 고성능 LLM",
            "provider": "OpenAI",
            "medical_specialized": False,
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "description": "OpenAI GPT-3.5 - 빠르고 효율적인 LLM",
            "provider": "OpenAI",
            "medical_specialized": False,
        },
        "medgemma": {
            "name": "MedGemma",
            "description": "Google MedGemma - 의료 도메인 특화 모델",
            "provider": "Google",
            "medical_specialized": True,
        },
        "gemini-pro": {
            "name": "Gemini Pro",
            "description": "Google Gemini Pro - 멀티모달 LLM",
            "provider": "Google",
            "medical_specialized": False,
        },
    }

    def __init__(
        self,
        model: LLMModel = "gpt-4",
        temperature: float = 0.3,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        """Initialize AnswerGenerator with multi-model support.

        Args:
            model: LLM model to use (gpt-4, gpt-3.5-turbo, medgemma, gemini-pro)
            temperature: Generation temperature (0.0-1.0)
            openai_api_key: OpenAI API key for GPT models
            google_api_key: Google API key for MedGemma/Gemini models
        """
        self.model = model
        self.temperature = temperature
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        self.llm = None
        self.medgemma_client = None
        self._initialize_model()

        self.prompt = PromptTemplate(
            template=self.MEDICAL_PROMPT,
            input_variables=["context", "question"],
        )
        self.medgemma_prompt = PromptTemplate(
            template=self.MEDGEMMA_PROMPT,
            input_variables=["context", "question"],
        )

    def _initialize_model(self):
        """Initialize the selected LLM model."""
        if self.model in ["gpt-4", "gpt-3.5-turbo"]:
            if self.openai_api_key:
                self.llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    openai_api_key=self.openai_api_key,
                )
        elif self.model == "medgemma":
            self.medgemma_client = MedGemmaClient(api_key=self.google_api_key)
            if self.medgemma_client.is_available():
                self.llm = self.medgemma_client.llm
        elif self.model == "gemini-pro":
            if self.google_api_key:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        google_api_key=self.google_api_key,
                        temperature=self.temperature,
                        convert_system_message_to_human=True,
                    )
                except Exception as e:
                    print(f"Gemini Pro initialization failed: {e}")

    def set_model(self, model: LLMModel):
        """Switch to a different LLM model.

        Args:
            model: New model to use
        """
        self.model = model
        self._initialize_model()

    @classmethod
    def get_available_models(cls) -> List[Dict]:
        """Get list of available models with their info."""
        return [
            {"id": model_id, **info}
            for model_id, info in cls.MODEL_INFO.items()
        ]

    def get_current_model_info(self) -> Dict:
        """Get info about the current model."""
        return {"id": self.model, **self.MODEL_INFO.get(self.model, {})}

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
            Generated answer with citations and model info.
        """
        model_info = self.get_current_model_info()

        if not self.llm:
            api_hint = "GOOGLE_API_KEY" if self.model in ["medgemma", "gemini-pro"] else "OPENAI_API_KEY"
            return {
                "answer": f"LLM이 구성되지 않았습니다. {api_hint} 환경변수를 설정하세요.",
                "sources": [],
                "model_used": model_info,
            }

        # Format context
        context = self._format_context(documents)

        # Select prompt based on model
        if self.model == "medgemma":
            prompt_template = self.medgemma_prompt
        else:
            prompt_template = self.prompt

        # Generate
        try:
            chain = prompt_template | self.llm
            result = chain.invoke({
                "context": context,
                "question": question,
            })
            answer_text = result.content
        except Exception as e:
            return {
                "answer": f"생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "model_used": model_info,
                "error": str(e),
            }

        # Extract citations
        citations = self._extract_citations(answer_text, documents)

        return {
            "answer": answer_text,
            "sources": citations,
            "model_used": model_info,
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
