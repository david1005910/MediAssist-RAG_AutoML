"""Schemas for symptom analysis API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Symptom(BaseModel):
    """Single symptom input."""
    name: str = Field(description="Symptom name (e.g., '발열', 'headache')")
    severity: int = Field(default=5, ge=1, le=10, description="Severity 1-10")
    duration_days: int = Field(default=1, ge=1, description="Duration in days")


class PatientInfo(BaseModel):
    """Optional patient information."""
    age: Optional[int] = Field(default=None, ge=0, le=150, description="Patient age")
    gender: Optional[str] = Field(default=None, description="'male' or 'female'")
    medical_history: Optional[List[str]] = Field(default=None, description="Past conditions")


class SymptomAnalysisRequest(BaseModel):
    """Request for symptom analysis."""
    symptoms: List[Symptom] = Field(min_length=1, description="List of symptoms")
    patient_info: Optional[PatientInfo] = Field(default=None, description="Patient metadata")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of predictions")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symptoms": [
                        {"name": "발열", "severity": 7, "duration_days": 3},
                        {"name": "기침", "severity": 5, "duration_days": 3},
                        {"name": "두통", "severity": 4, "duration_days": 2}
                    ],
                    "patient_info": {
                        "age": 35,
                        "gender": "male",
                        "medical_history": ["고혈압"]
                    },
                    "top_k": 5
                }
            ]
        }
    }


class DiseasePrediction(BaseModel):
    """Single disease prediction."""
    disease: str = Field(description="Predicted disease name")
    icd_code: str = Field(description="ICD-10 code")
    probability: float = Field(ge=0, le=1, description="Prediction probability")
    confidence: str = Field(description="Confidence level: high/medium/low")
    description: Optional[str] = Field(default=None, description="Disease description")


class RiskAssessment(BaseModel):
    """Risk assessment for the patient."""
    risk_score: float = Field(ge=0, le=100, description="Overall risk score 0-100")
    risk_level: str = Field(description="Risk level: low/medium/high/critical")
    factors: List[str] = Field(description="Contributing risk factors")
    recommendations: List[str] = Field(description="Recommended actions")


class SymptomAnalysisResponse(BaseModel):
    """Response from symptom analysis."""
    predictions: List[DiseasePrediction] = Field(description="Disease predictions")
    extracted_symptoms: List[str] = Field(description="Normalized symptom names")
    risk_assessment: RiskAssessment = Field(description="Risk evaluation")
    disclaimer: str = Field(
        default="이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요.",
        description="Medical disclaimer"
    )


class SymptomNERRequest(BaseModel):
    """Request for symptom NER extraction."""
    text: str = Field(min_length=1, description="Free text describing symptoms")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "3일 전부터 열이 나고 기침이 심해졌습니다. 머리도 아프고 몸살 기운이 있어요."
                }
            ]
        }
    }


class SymptomNERResponse(BaseModel):
    """Response from symptom NER extraction."""
    extracted_symptoms: List[Symptom] = Field(description="Extracted symptoms")
    original_text: str = Field(description="Original input text")
    entities: List[Dict[str, Any]] = Field(description="Raw NER entities")
