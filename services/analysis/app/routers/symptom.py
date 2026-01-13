"""Symptom analysis API router."""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from fastapi import APIRouter, HTTPException

from app.schemas.symptom import (
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    SymptomNERRequest,
    SymptomNERResponse,
    DiseasePrediction,
    RiskAssessment,
    Symptom,
)

# Add models to path
MODELS_PATH = Path("/app/models")
if not MODELS_PATH.exists():
    MODELS_PATH = Path(__file__).parents[4] / "models"
if str(MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(MODELS_PATH))

router = APIRouter(prefix="/api/v1/symptom", tags=["Symptom Analysis"])

# Global model instance
_ner_model = None

# Demo disease database with symptom patterns
DISEASE_DATABASE = {
    "급성 상기도 감염": {
        "icd_code": "J06.9",
        "symptoms": ["발열", "기침", "인후통", "콧물", "두통"],
        "description": "감기 증상을 유발하는 바이러스 감염",
        "risk_weight": 0.2,
    },
    "인플루엔자": {
        "icd_code": "J11",
        "symptoms": ["고열", "발열", "근육통", "몸살", "기침", "두통", "피로"],
        "description": "인플루엔자 바이러스에 의한 급성 호흡기 감염",
        "risk_weight": 0.4,
    },
    "폐렴": {
        "icd_code": "J18.9",
        "symptoms": ["발열", "기침", "호흡곤란", "가래", "흉통", "오한"],
        "description": "폐 조직의 염증성 질환",
        "risk_weight": 0.7,
    },
    "급성 위장염": {
        "icd_code": "K52.9",
        "symptoms": ["구토", "설사", "복통", "발열", "오심"],
        "description": "위장관의 급성 염증",
        "risk_weight": 0.3,
    },
    "편두통": {
        "icd_code": "G43.9",
        "symptoms": ["두통", "구역", "빛 민감", "소리 민감", "시각 장애"],
        "description": "반복적인 중등도 이상의 두통",
        "risk_weight": 0.2,
    },
    "급성 기관지염": {
        "icd_code": "J20.9",
        "symptoms": ["기침", "가래", "흉부 불편감", "발열", "피로"],
        "description": "기관지의 급성 염증",
        "risk_weight": 0.4,
    },
    "알레르기성 비염": {
        "icd_code": "J30.4",
        "symptoms": ["콧물", "재채기", "코막힘", "눈 가려움", "두통"],
        "description": "알레르겐에 의한 비강 점막 염증",
        "risk_weight": 0.1,
    },
    "고혈압": {
        "icd_code": "I10",
        "symptoms": ["두통", "어지러움", "가슴 두근거림", "피로", "시력 저하"],
        "description": "지속적으로 높은 혈압 상태",
        "risk_weight": 0.5,
    },
}

# Symptom aliases for Korean-English mapping
SYMPTOM_ALIASES = {
    "fever": "발열",
    "cough": "기침",
    "headache": "두통",
    "sore throat": "인후통",
    "runny nose": "콧물",
    "muscle pain": "근육통",
    "fatigue": "피로",
    "nausea": "구역",
    "vomiting": "구토",
    "diarrhea": "설사",
    "chest pain": "흉통",
    "shortness of breath": "호흡곤란",
    "열": "발열",
    "고열": "발열",
    "몸살": "근육통",
}


def get_ner_model():
    """Get or initialize NER model."""
    global _ner_model
    if _ner_model is None:
        from symptom_classifier.ner import MedicalNER
        _ner_model = MedicalNER()
    return _ner_model


def normalize_symptom(symptom_name: str) -> str:
    """Normalize symptom name using aliases."""
    name_lower = symptom_name.lower().strip()
    return SYMPTOM_ALIASES.get(name_lower, symptom_name)


def calculate_disease_scores(
    symptoms: List[Symptom],
    patient_info: Optional[Dict] = None,
) -> List[Dict]:
    """Calculate disease match scores based on symptoms."""
    symptom_names = [normalize_symptom(s.name) for s in symptoms]
    severity_avg = np.mean([s.severity for s in symptoms])
    duration_avg = np.mean([s.duration_days for s in symptoms])

    scores = []
    for disease, info in DISEASE_DATABASE.items():
        # Calculate symptom overlap
        disease_symptoms = set(info["symptoms"])
        patient_symptoms = set(symptom_names)

        overlap = len(disease_symptoms & patient_symptoms)
        coverage = overlap / len(disease_symptoms) if disease_symptoms else 0

        # Boost score based on severity and duration
        severity_factor = 1 + (severity_avg - 5) * 0.1
        duration_factor = 1 + min(duration_avg / 7, 1) * 0.2

        # Age adjustment
        age_factor = 1.0
        if patient_info and patient_info.get("age"):
            age = patient_info["age"]
            if age > 65 or age < 5:
                age_factor = 1.2  # Higher risk for elderly and young children

        # Calculate final score
        base_score = coverage * severity_factor * duration_factor * age_factor

        if overlap > 0:
            scores.append({
                "disease": disease,
                "icd_code": info["icd_code"],
                "score": base_score,
                "overlap": overlap,
                "description": info["description"],
                "risk_weight": info["risk_weight"],
            })

    # Sort by score descending
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def calculate_risk_assessment(
    predictions: List[Dict],
    symptoms: List[Symptom],
    patient_info: Optional[Dict] = None,
) -> RiskAssessment:
    """Calculate overall risk assessment."""
    factors = []
    recommendations = []

    # Base risk from top prediction
    base_risk = 0
    if predictions:
        top_pred = predictions[0]
        base_risk = top_pred.get("risk_weight", 0.3) * top_pred.get("score", 0.5) * 100

    # Severity factor - use max severity (if any symptom is severe, it's a risk)
    severity_max = max([s.severity for s in symptoms])
    if severity_max >= 7:
        base_risk += 15
        factors.append(f"높은 증상 심각도 (최대 {severity_max}/10)")
        recommendations.append("증상이 심하므로 가능한 빨리 의료기관을 방문하세요")

    # Duration factor
    duration_max = max([s.duration_days for s in symptoms])
    if duration_max >= 7:
        base_risk += 10
        factors.append(f"장기간 증상 지속 ({duration_max}일)")
        recommendations.append("증상이 오래 지속되고 있으니 전문의 상담을 권장합니다")

    # Age factor
    if patient_info:
        age = patient_info.get("age", 0)
        if age >= 65:
            base_risk += 15
            factors.append("65세 이상 고령")
            recommendations.append("고령층은 합병증 위험이 높으니 주의가 필요합니다")
        elif age <= 5:
            base_risk += 10
            factors.append("5세 이하 영유아")
            recommendations.append("영유아는 소아과 전문의 진료를 권장합니다")

        # Medical history
        history = patient_info.get("medical_history", [])
        if history:
            base_risk += len(history) * 5
            factors.append(f"기저질환 {len(history)}개")

    # Number of symptoms
    if len(symptoms) >= 5:
        base_risk += 5
        factors.append(f"다수의 증상 ({len(symptoms)}개)")

    # Determine risk level
    risk_score = min(base_risk, 100)
    if risk_score >= 70:
        risk_level = "critical"
        recommendations.insert(0, "즉시 응급실 방문을 권장합니다")
    elif risk_score >= 50:
        risk_level = "high"
        recommendations.insert(0, "가능한 빨리 의료기관을 방문하세요")
    elif risk_score >= 30:
        risk_level = "medium"
        recommendations.append("증상이 악화되면 의료기관을 방문하세요")
    else:
        risk_level = "low"
        recommendations.append("충분한 휴식을 취하고 수분을 섭취하세요")

    if not factors:
        factors.append("특별한 위험 요인 없음")

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        factors=factors,
        recommendations=recommendations,
    )


@router.post("/analyze", response_model=SymptomAnalysisResponse)
async def analyze_symptoms(request: SymptomAnalysisRequest):
    """
    Analyze symptoms and predict possible diseases.

    Uses BioBERT embeddings and rule-based matching to suggest
    potential diagnoses based on reported symptoms.

    **Note**: This is a demonstration system. Always consult a medical
    professional for actual diagnosis.
    """
    try:
        # Normalize symptoms
        normalized_symptoms = [
            normalize_symptom(s.name) for s in request.symptoms
        ]

        # Calculate disease scores
        patient_info = request.patient_info.model_dump() if request.patient_info else None
        scores = calculate_disease_scores(request.symptoms, patient_info)

        # Convert to predictions
        predictions = []
        total_score = sum(s["score"] for s in scores[:request.top_k]) or 1

        for score_info in scores[:request.top_k]:
            # Normalize probability
            prob = score_info["score"] / total_score if total_score > 0 else 0
            prob = min(prob, 0.95)  # Cap at 95%

            confidence = "high" if prob >= 0.5 else "medium" if prob >= 0.3 else "low"

            predictions.append(DiseasePrediction(
                disease=score_info["disease"],
                icd_code=score_info["icd_code"],
                probability=prob,
                confidence=confidence,
                description=score_info["description"],
            ))

        # If no matches, return general response
        if not predictions:
            predictions.append(DiseasePrediction(
                disease="분류 불가",
                icd_code="R69",
                probability=1.0,
                confidence="low",
                description="입력된 증상으로 특정 질환을 예측하기 어렵습니다",
            ))

        # Calculate risk assessment
        risk = calculate_risk_assessment(
            [s for s in scores[:request.top_k]],
            request.symptoms,
            patient_info,
        )

        return SymptomAnalysisResponse(
            predictions=predictions,
            extracted_symptoms=normalized_symptoms,
            risk_assessment=risk,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/extract", response_model=SymptomNERResponse)
async def extract_symptoms(request: SymptomNERRequest):
    """
    Extract symptoms from free-text description using NER.

    Uses BioBERT-based Named Entity Recognition to identify
    medical symptoms from natural language text.
    """
    try:
        # Simple rule-based extraction for demo
        # In production, this would use the trained NER model
        text = request.text.lower()

        extracted = []
        found_symptoms = []

        # Check for known symptoms in text
        symptom_patterns = {
            "열": ("발열", 6),
            "고열": ("발열", 8),
            "기침": ("기침", 5),
            "두통": ("두통", 5),
            "머리": ("두통", 4),
            "목 아프": ("인후통", 5),
            "목이 아프": ("인후통", 5),
            "목도 아프": ("인후통", 5),
            "목 아파": ("인후통", 5),
            "목이 아파": ("인후통", 5),
            "목도 아파": ("인후통", 5),
            "목도 아픕": ("인후통", 5),
            "목이 아픕": ("인후통", 5),
            "목 아픕": ("인후통", 5),
            "목이 따": ("인후통", 5),
            "목 따": ("인후통", 5),
            "인후통": ("인후통", 5),
            "목아프": ("인후통", 5),
            "콧물": ("콧물", 4),
            "코막힘": ("코막힘", 4),
            "몸살": ("근육통", 6),
            "근육": ("근육통", 5),
            "피곤": ("피로", 4),
            "피로": ("피로", 5),
            "구토": ("구토", 6),
            "토하": ("구토", 6),
            "설사": ("설사", 6),
            "복통": ("복통", 5),
            "배 아프": ("복통", 5),
            "배가 아프": ("복통", 5),
            "숨": ("호흡곤란", 7),
            "호흡": ("호흡곤란", 6),
            "가래": ("가래", 4),
            "오한": ("오한", 5),
        }

        # Severity modifiers
        high_severity_words = ["심한", "심하게", "매우", "너무", "극심한", "많이", "엄청"]
        mild_severity_words = ["약간", "조금", "살짝", "가벼운", "경미한"]

        import re

        for pattern, (symptom_name, default_severity) in symptom_patterns.items():
            if pattern in text and symptom_name not in found_symptoms:
                found_symptoms.append(symptom_name)

                # Calculate adjusted severity based on modifiers
                severity = default_severity

                # Check for high severity modifiers near the symptom
                pattern_pos = text.find(pattern)
                context_start = max(0, pattern_pos - 10)
                context_end = min(len(text), pattern_pos + len(pattern) + 10)
                context = text[context_start:context_end]

                for word in high_severity_words:
                    if word in context:
                        severity = min(10, severity + 2)
                        break

                for word in mild_severity_words:
                    if word in context:
                        severity = max(1, severity - 2)
                        break

                # Special handling for fever with temperature
                if symptom_name == "발열":
                    temp_match = re.search(r'(\d{2}(?:\.\d)?)\s*도', text)
                    if temp_match:
                        temp = float(temp_match.group(1))
                        if temp >= 39:
                            severity = 8
                        elif temp >= 38.5:
                            severity = 7
                        elif temp >= 38:
                            severity = 6
                        elif temp >= 37.5:
                            severity = 5

                extracted.append(Symptom(
                    name=symptom_name,
                    severity=severity,
                    duration_days=1,
                ))

        # Extract duration if mentioned
        duration_match = re.search(r'(\d+)\s*일', text)
        if duration_match and extracted:
            duration = int(duration_match.group(1))
            for symptom in extracted:
                symptom.duration_days = duration

        return SymptomNERResponse(
            extracted_symptoms=extracted,
            original_text=request.text,
            entities=[{"text": s.name, "type": "SYMPTOM"} for s in extracted],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/diseases")
async def list_diseases():
    """List all known diseases in the system."""
    return {
        "diseases": [
            {
                "name": name,
                "icd_code": info["icd_code"],
                "symptoms": info["symptoms"],
                "description": info["description"],
            }
            for name, info in DISEASE_DATABASE.items()
        ]
    }


@router.get("/symptoms")
async def list_symptoms():
    """List all recognized symptoms."""
    all_symptoms = set()
    for info in DISEASE_DATABASE.values():
        all_symptoms.update(info["symptoms"])

    return {
        "symptoms": sorted(list(all_symptoms)),
        "aliases": SYMPTOM_ALIASES,
    }
