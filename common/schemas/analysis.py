"""Shared analysis schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SymptomInput(BaseModel):
    """Symptom input schema."""
    name: str = Field(..., min_length=1, max_length=200)
    severity: Optional[int] = Field(None, ge=1, le=10)
    duration_days: Optional[int] = Field(None, ge=0)
    description: Optional[str] = Field(None, max_length=1000)


class PatientInfo(BaseModel):
    """Patient info schema."""
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    medical_history: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    """Prediction result schema."""
    disease: str
    icd_code: str
    probability: float = Field(..., ge=0, le=1)
    confidence: str = Field(..., pattern="^(high|medium|low)$")


class RiskAssessment(BaseModel):
    """Risk assessment schema."""
    score: float = Field(..., ge=0, le=100)
    level: str = Field(..., pattern="^(low|moderate|high|critical)$")
    factors: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    """Source reference schema."""
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    relevance: float = Field(..., ge=0, le=1)
    url: Optional[str] = None


class RAGResultSchema(BaseModel):
    """RAG result schema."""
    query: str
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Analysis response schema."""
    id: UUID
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None

    predictions: Optional[list[PredictionResult]] = None
    risk_assessment: Optional[RiskAssessment] = None
    rag_result: Optional[RAGResultSchema] = None

    disclaimer: str = "이 분석 결과는 참고용이며, 최종 진단은 의사가 결정해야 합니다."

    class Config:
        from_attributes = True
