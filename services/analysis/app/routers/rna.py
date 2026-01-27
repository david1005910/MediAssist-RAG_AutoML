"""RNA Analysis API Router."""

from typing import List, Optional
import sys
from pathlib import Path
import re
import json

from fastapi import APIRouter, HTTPException, status, UploadFile, File

from app.schemas.rna import (
    RNAAnalysisRequest,
    RNAAnalysisResponse,
    RNABatchRequest,
    RNABatchResponse,
    RNAValidationRequest,
    RNAValidationResponse,
    RNATypeInfo,
    DiseaseInfo,
    SequenceAnalysis,
    DiseasePrediction,
    RNARiskAssessment,
    RNAFileUploadResponse,
    RNASampleDataResponse,
)

router = APIRouter(prefix="/api/v1/rna", tags=["RNA Analysis"])

# Add models path to sys.path for imports
def _find_models_path():
    """Find models directory in various environments."""
    current = Path(__file__).resolve()

    # Try fixed Docker path first
    docker_path = Path("/app/models")
    if docker_path.exists():
        return docker_path

    # Try parent directories (up to project root)
    for i in range(len(current.parents)):
        candidate = current.parents[i] / "models"
        if candidate.exists():
            return candidate

    # Default fallback
    return docker_path

MODELS_PATH = _find_models_path()
if str(MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(MODELS_PATH))

# Global model instance (lazy loading)
_rna_predictor = None


def get_rna_predictor():
    """Get or initialize RNA predictor model."""
    global _rna_predictor
    if _rna_predictor is None:
        try:
            from rna_predictor import RNAPredictor
            _rna_predictor = RNAPredictor()
        except Exception as e:
            print(f"[RNA Predictor] Could not load model: {e}")
            _rna_predictor = None
    return _rna_predictor


# RNA Types
RNA_TYPES = [
    RNATypeInfo(
        code="mRNA",
        name="Messenger RNA",
        description="전령 RNA - 단백질 합성을 위한 유전 정보 전달",
    ),
    RNATypeInfo(
        code="siRNA",
        name="Small Interfering RNA",
        description="소간섭 RNA - 유전자 발현 조절에 관여",
    ),
    RNATypeInfo(
        code="circRNA",
        name="Circular RNA",
        description="원형 RNA - 유전자 조절 및 miRNA 스펀지 역할",
    ),
    RNATypeInfo(
        code="lncRNA",
        name="Long Non-coding RNA",
        description="긴 비코딩 RNA - 다양한 세포 과정 조절",
    ),
    RNATypeInfo(
        code="other",
        name="Other RNA Types",
        description="기타 RNA 유형",
    ),
]

# Detectable diseases with detailed descriptions
DISEASES = [
    DiseaseInfo(
        name="정상/저위험",
        name_en="Normal/Low Risk",
        icd_code="N/A",
        description="분석된 RNA 서열이 정상 범위 내에 있으며, 알려진 병원성 변이가 감지되지 않았습니다.",
        clinical_significance="임상적으로 유의미한 이상 소견 없음",
        recommendation="정기적인 건강 검진을 권장합니다.",
        related_genes=[],
    ),
    DiseaseInfo(
        name="RNA 변이 관련 질환",
        name_en="RNA Mutation Related Disease",
        icd_code="Q89.9",
        description="RNA 서열에서 단백질 기능에 영향을 줄 수 있는 변이가 감지되었습니다. 스플라이싱 이상, 코돈 변이, 또는 조절 영역 변화가 포함될 수 있습니다.",
        clinical_significance="유전자 발현 및 단백질 합성에 영향을 미칠 수 있는 변이",
        recommendation="유전 상담 및 추가 분자 진단 검사를 권장합니다.",
        related_genes=["BRCA1", "BRCA2", "TP53", "MLH1", "MSH2"],
    ),
    DiseaseInfo(
        name="siRNA 치료 반응 예측",
        name_en="siRNA Therapy Response Prediction",
        icd_code="Z51.1",
        description="이 RNA 서열은 siRNA(소간섭 RNA) 치료제의 표적이 될 수 있습니다. siRNA는 특정 mRNA를 분해하여 질병 유발 단백질의 생성을 억제합니다.",
        clinical_significance="RNA 간섭(RNAi) 기반 치료의 잠재적 표적",
        recommendation="siRNA 치료제 임상시험 참여 가능성을 전문의와 상담하세요.",
        related_genes=["VEGF", "KRAS", "BCL2", "MYC", "EGFR"],
    ),
    DiseaseInfo(
        name="ASO 효능 예측",
        name_en="ASO Efficacy Prediction",
        icd_code="Z51.1",
        description="안티센스 올리고뉴클레오타이드(ASO) 치료에 적합한 표적 서열입니다. ASO는 특정 RNA에 결합하여 스플라이싱을 조절하거나 mRNA를 분해합니다.",
        clinical_significance="ASO 기반 치료제(예: 누시너센, 에테플리르센)의 잠재적 적용 대상",
        recommendation="ASO 치료 가능성에 대해 신경과 또는 유전학 전문의와 상담하세요.",
        related_genes=["SMN1", "SMN2", "DMD", "HTT", "SOD1"],
    ),
    DiseaseInfo(
        name="UTR 변이 병원성",
        name_en="UTR Variant Pathogenicity",
        icd_code="Q99.8",
        description="5' 또는 3' 비번역 영역(UTR)에서 병원성 변이가 감지되었습니다. UTR 변이는 mRNA 안정성, 번역 효율, 또는 microRNA 결합에 영향을 줄 수 있습니다.",
        clinical_significance="유전자 발현 조절 이상으로 인한 질환 위험",
        recommendation="기능적 영향 평가를 위한 추가 검사를 권장합니다.",
        related_genes=["FMR1", "DMPK", "APP", "MAPT"],
    ),
    DiseaseInfo(
        name="유전성 근육 질환",
        name_en="Hereditary Muscle Disease",
        icd_code="G71.9",
        description="근이영양증, 근무력증, 또는 선천성 근병증과 관련된 RNA 이상이 감지되었습니다. 근육 단백질 생성 또는 기능에 영향을 미치는 변이입니다.",
        clinical_significance="진행성 근력 약화, 근위축, 호흡 기능 저하 위험",
        recommendation="신경근육 전문의 상담 및 유전자 검사를 권장합니다. 물리치료 및 호흡 관리가 필요할 수 있습니다.",
        related_genes=["DMD", "SMN1", "DMPK", "RYR1", "MTM1", "LMNA"],
    ),
    DiseaseInfo(
        name="신경퇴행성 질환",
        name_en="Neurodegenerative Disease",
        icd_code="G31.9",
        description="알츠하이머병, 파킨슨병, 헌팅턴병, 또는 근위축성측삭경화증(ALS)과 관련된 RNA 패턴이 감지되었습니다. 신경세포의 점진적 손실을 초래할 수 있습니다.",
        clinical_significance="인지 기능 저하, 운동 장애, 또는 행동 변화 위험",
        recommendation="신경과 전문의와 조기 상담을 권장합니다. 조기 개입이 질병 진행을 늦출 수 있습니다.",
        related_genes=["APP", "PSEN1", "PSEN2", "SNCA", "HTT", "SOD1", "C9orf72", "MAPT"],
    ),
    DiseaseInfo(
        name="암 관련 RNA 이상",
        name_en="Cancer-related RNA Abnormality",
        icd_code="C80.1",
        description="종양 억제 유전자 또는 암 유전자의 RNA에서 이상이 감지되었습니다. 비정상적인 발현, 스플라이싱 변이, 또는 융합 전사체가 포함될 수 있습니다.",
        clinical_significance="세포 증식, 세포사멸 회피, 또는 전이 위험 증가",
        recommendation="종양 전문의와 상담하여 추가 진단 검사(조직검사, 영상검사)를 고려하세요.",
        related_genes=["TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "ALK", "ROS1", "MYC", "BCL2"],
    ),
]

# Detailed disease information for predictions
DISEASE_DETAILS = {
    "정상/저위험": {
        "description": "분석된 RNA 서열이 정상 범위 내에 있으며, 알려진 병원성 변이가 감지되지 않았습니다.",
        "clinical_significance": "임상적으로 유의미한 이상 소견 없음",
        "recommendation": "정기적인 건강 검진을 권장합니다.",
        "related_genes": [],
    },
    "RNA 변이 관련 질환": {
        "description": "RNA 서열에서 단백질 기능에 영향을 줄 수 있는 변이가 감지되었습니다. 스플라이싱 이상, 코돈 변이, 또는 조절 영역 변화가 포함될 수 있습니다.",
        "clinical_significance": "유전자 발현 및 단백질 합성에 영향을 미칠 수 있는 변이",
        "recommendation": "유전 상담 및 추가 분자 진단 검사를 권장합니다.",
        "related_genes": ["BRCA1", "BRCA2", "TP53", "MLH1", "MSH2"],
    },
    "siRNA 치료 반응 예측": {
        "description": "이 RNA 서열은 siRNA(소간섭 RNA) 치료제의 표적이 될 수 있습니다. siRNA는 특정 mRNA를 분해하여 질병 유발 단백질의 생성을 억제합니다.",
        "clinical_significance": "RNA 간섭(RNAi) 기반 치료의 잠재적 표적",
        "recommendation": "siRNA 치료제 임상시험 참여 가능성을 전문의와 상담하세요.",
        "related_genes": ["VEGF", "KRAS", "BCL2", "MYC", "EGFR"],
    },
    "ASO 효능 예측": {
        "description": "안티센스 올리고뉴클레오타이드(ASO) 치료에 적합한 표적 서열입니다. ASO는 특정 RNA에 결합하여 스플라이싱을 조절하거나 mRNA를 분해합니다.",
        "clinical_significance": "ASO 기반 치료제(예: 누시너센, 에테플리르센)의 잠재적 적용 대상",
        "recommendation": "ASO 치료 가능성에 대해 신경과 또는 유전학 전문의와 상담하세요.",
        "related_genes": ["SMN1", "SMN2", "DMD", "HTT", "SOD1"],
    },
    "UTR 변이 병원성": {
        "description": "5' 또는 3' 비번역 영역(UTR)에서 병원성 변이가 감지되었습니다. UTR 변이는 mRNA 안정성, 번역 효율, 또는 microRNA 결합에 영향을 줄 수 있습니다.",
        "clinical_significance": "유전자 발현 조절 이상으로 인한 질환 위험",
        "recommendation": "기능적 영향 평가를 위한 추가 검사를 권장합니다.",
        "related_genes": ["FMR1", "DMPK", "APP", "MAPT"],
    },
    "유전성 근육 질환": {
        "description": "근이영양증, 근무력증, 또는 선천성 근병증과 관련된 RNA 이상이 감지되었습니다. 근육 단백질 생성 또는 기능에 영향을 미치는 변이입니다.",
        "clinical_significance": "진행성 근력 약화, 근위축, 호흡 기능 저하 위험",
        "recommendation": "신경근육 전문의 상담 및 유전자 검사를 권장합니다. 물리치료 및 호흡 관리가 필요할 수 있습니다.",
        "related_genes": ["DMD", "SMN1", "DMPK", "RYR1", "MTM1", "LMNA"],
    },
    "신경퇴행성 질환": {
        "description": "알츠하이머병, 파킨슨병, 헌팅턴병, 또는 근위축성측삭경화증(ALS)과 관련된 RNA 패턴이 감지되었습니다. 신경세포의 점진적 손실을 초래할 수 있습니다.",
        "clinical_significance": "인지 기능 저하, 운동 장애, 또는 행동 변화 위험",
        "recommendation": "신경과 전문의와 조기 상담을 권장합니다. 조기 개입이 질병 진행을 늦출 수 있습니다.",
        "related_genes": ["APP", "PSEN1", "PSEN2", "SNCA", "HTT", "SOD1", "C9orf72", "MAPT"],
    },
    "암 관련 RNA 이상": {
        "description": "종양 억제 유전자 또는 암 유전자의 RNA에서 이상이 감지되었습니다. 비정상적인 발현, 스플라이싱 변이, 또는 융합 전사체가 포함될 수 있습니다.",
        "clinical_significance": "세포 증식, 세포사멸 회피, 또는 전이 위험 증가",
        "recommendation": "종양 전문의와 상담하여 추가 진단 검사(조직검사, 영상검사)를 고려하세요.",
        "related_genes": ["TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "ALK", "ROS1", "MYC", "BCL2"],
    },
}


def normalize_sequence(sequence: str) -> str:
    """Normalize RNA sequence."""
    return sequence.upper().replace("T", "U")


def validate_sequence(sequence: str) -> tuple:
    """Validate RNA sequence.

    Returns:
        Tuple of (is_valid, normalized_sequence, invalid_chars)
    """
    normalized = normalize_sequence(sequence)
    valid_chars = set("AUGC")
    invalid_chars = [c for c in normalized if c not in valid_chars and not c.isspace()]

    # Remove spaces and newlines
    normalized = "".join(c for c in normalized if c in valid_chars)
    is_valid = len(normalized) > 0 and len(invalid_chars) == 0

    return is_valid, normalized, list(set(invalid_chars))


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content percentage."""
    if len(sequence) == 0:
        return 0.0
    gc_count = sequence.count("G") + sequence.count("C")
    return (gc_count / len(sequence)) * 100


def detect_rna_type(sequence: str) -> tuple:
    """Detect RNA type from sequence characteristics.

    Returns:
        Tuple of (rna_type, confidence)
    """
    length = len(sequence)
    gc_content = calculate_gc_content(sequence)

    # Simple heuristics for RNA type detection
    if length < 30:
        return "siRNA", 0.7
    elif length < 100:
        return "siRNA", 0.5 if gc_content > 30 else ("lncRNA", 0.4)
    elif length < 500:
        return "mRNA", 0.6
    else:
        return "mRNA", 0.5


def find_motifs(sequence: str) -> List[str]:
    """Find known RNA motifs in sequence."""
    motifs = []

    patterns = {
        "Kozak": r"[AG]CCAUGG",
        "Poly(A) Signal": r"AAUAAA",
        "AU-rich Element": r"AUUUA",
        "Iron-responsive Element": r"CAGUG",
    }

    for motif_name, pattern in patterns.items():
        if re.search(pattern, sequence):
            motifs.append(motif_name)

    return motifs


def get_disease_predictions_for_rna_type(rna_type: str, gc_content: float, length: int) -> List[DiseasePrediction]:
    """Generate disease predictions based on RNA type and sequence characteristics."""
    predictions = []

    # Base normal/low-risk prediction
    normal_details = DISEASE_DETAILS["정상/저위험"]
    normal_prob = 0.65 if gc_content >= 30 and gc_content <= 70 else 0.45

    predictions.append(DiseasePrediction(
        disease="정상/저위험",
        disease_en="Normal/Low Risk",
        icd_code="N/A",
        probability=normal_prob,
        confidence="high" if normal_prob > 0.6 else "medium",
        description=normal_details["description"],
        clinical_significance=normal_details["clinical_significance"],
        recommendation=normal_details["recommendation"],
        related_genes=normal_details["related_genes"],
    ))

    # RNA type specific predictions
    if rna_type == "mRNA":
        # mRNA - potential cancer or genetic disease markers
        cancer_details = DISEASE_DETAILS["암 관련 RNA 이상"]
        mutation_details = DISEASE_DETAILS["RNA 변이 관련 질환"]

        predictions.append(DiseasePrediction(
            disease="암 관련 RNA 이상",
            disease_en="Cancer-related RNA Abnormality",
            icd_code="C80.1",
            probability=0.18,
            confidence="medium",
            description=cancer_details["description"],
            clinical_significance=cancer_details["clinical_significance"],
            recommendation=cancer_details["recommendation"],
            related_genes=cancer_details["related_genes"],
        ))
        predictions.append(DiseasePrediction(
            disease="RNA 변이 관련 질환",
            disease_en="RNA Mutation Related Disease",
            icd_code="Q89.9",
            probability=0.12,
            confidence="low",
            description=mutation_details["description"],
            clinical_significance=mutation_details["clinical_significance"],
            recommendation=mutation_details["recommendation"],
            related_genes=mutation_details["related_genes"],
        ))

    elif rna_type == "siRNA":
        # siRNA - therapy response predictions
        sirna_details = DISEASE_DETAILS["siRNA 치료 반응 예측"]
        aso_details = DISEASE_DETAILS["ASO 효능 예측"]

        predictions.append(DiseasePrediction(
            disease="siRNA 치료 반응 예측",
            disease_en="siRNA Therapy Response Prediction",
            icd_code="Z51.1",
            probability=0.25,
            confidence="medium",
            description=sirna_details["description"],
            clinical_significance=sirna_details["clinical_significance"],
            recommendation=sirna_details["recommendation"],
            related_genes=sirna_details["related_genes"],
        ))
        predictions.append(DiseasePrediction(
            disease="ASO 효능 예측",
            disease_en="ASO Efficacy Prediction",
            icd_code="Z51.1",
            probability=0.10,
            confidence="low",
            description=aso_details["description"],
            clinical_significance=aso_details["clinical_significance"],
            recommendation=aso_details["recommendation"],
            related_genes=aso_details["related_genes"],
        ))

    elif rna_type == "circRNA":
        # circRNA - regulatory function disorders
        neuro_details = DISEASE_DETAILS["신경퇴행성 질환"]
        cancer_details = DISEASE_DETAILS["암 관련 RNA 이상"]

        predictions.append(DiseasePrediction(
            disease="신경퇴행성 질환",
            disease_en="Neurodegenerative Disease",
            icd_code="G31.9",
            probability=0.15,
            confidence="low",
            description=neuro_details["description"],
            clinical_significance=neuro_details["clinical_significance"],
            recommendation=neuro_details["recommendation"],
            related_genes=neuro_details["related_genes"],
        ))
        predictions.append(DiseasePrediction(
            disease="암 관련 RNA 이상",
            disease_en="Cancer-related RNA Abnormality",
            icd_code="C80.1",
            probability=0.12,
            confidence="low",
            description=cancer_details["description"],
            clinical_significance=cancer_details["clinical_significance"],
            recommendation=cancer_details["recommendation"],
            related_genes=cancer_details["related_genes"],
        ))

    elif rna_type == "lncRNA":
        # lncRNA - various regulatory disorders
        muscle_details = DISEASE_DETAILS["유전성 근육 질환"]
        cancer_details = DISEASE_DETAILS["암 관련 RNA 이상"]
        utr_details = DISEASE_DETAILS["UTR 변이 병원성"]

        predictions.append(DiseasePrediction(
            disease="암 관련 RNA 이상",
            disease_en="Cancer-related RNA Abnormality",
            icd_code="C80.1",
            probability=0.20,
            confidence="medium",
            description=cancer_details["description"],
            clinical_significance=cancer_details["clinical_significance"],
            recommendation=cancer_details["recommendation"],
            related_genes=cancer_details["related_genes"],
        ))
        predictions.append(DiseasePrediction(
            disease="유전성 근육 질환",
            disease_en="Hereditary Muscle Disease",
            icd_code="G71.9",
            probability=0.08,
            confidence="low",
            description=muscle_details["description"],
            clinical_significance=muscle_details["clinical_significance"],
            recommendation=muscle_details["recommendation"],
            related_genes=muscle_details["related_genes"],
        ))

    else:
        # Default predictions for unknown types
        mutation_details = DISEASE_DETAILS["RNA 변이 관련 질환"]
        predictions.append(DiseasePrediction(
            disease="RNA 변이 관련 질환",
            disease_en="RNA Mutation Related Disease",
            icd_code="Q89.9",
            probability=0.15,
            confidence="low",
            description=mutation_details["description"],
            clinical_significance=mutation_details["clinical_significance"],
            recommendation=mutation_details["recommendation"],
            related_genes=mutation_details["related_genes"],
        ))

    # Sort by probability descending
    predictions.sort(key=lambda x: x.probability, reverse=True)
    return predictions


def create_mock_prediction(sequence: str, rna_type: Optional[str] = None) -> RNAAnalysisResponse:
    """Create mock prediction when model is not available."""
    normalized = normalize_sequence(sequence)
    length = len(normalized)
    gc_content = calculate_gc_content(normalized)

    # Detect RNA type if not provided
    if rna_type and rna_type != "auto":
        detected_type = rna_type
        type_confidence = 0.9
    else:
        detected_type, type_confidence = detect_rna_type(normalized)

    # Find motifs
    motifs = find_motifs(normalized)

    # Create disease predictions based on RNA type
    disease_predictions = get_disease_predictions_for_rna_type(detected_type, gc_content, length)

    # Calculate risk score based on disease predictions
    # Weight non-normal disease probabilities (normal/low-risk is excluded)
    disease_weights = {
        "암 관련 RNA 이상": 3.0,
        "신경퇴행성 질환": 2.5,
        "유전성 근육 질환": 2.5,
        "UTR 변이 병원성": 2.0,
        "RNA 변이 관련 질환": 1.5,
        "siRNA 치료 반응 예측": 1.0,
        "ASO 효능 예측": 1.0,
    }

    weighted_risk = 0.0
    risk_factors = []
    for pred in disease_predictions:
        if pred.disease != "정상/저위험":
            weight = disease_weights.get(pred.disease, 1.0)
            contribution = pred.probability * weight * 100
            weighted_risk += contribution
            if pred.probability > 0.1:
                risk_factors.append(f"{pred.disease}: {pred.probability*100:.1f}%")

    # Scale to 0-100 range (max possible weighted risk is around 200)
    risk_score = min(100, max(0, weighted_risk / 2))

    # Determine risk level based on calculated risk
    if risk_score < 20:
        risk_level = "low"
        pathogenicity = "benign"
    elif risk_score < 40:
        risk_level = "moderate"
        pathogenicity = "likely_benign"
    elif risk_score < 60:
        risk_level = "high"
        pathogenicity = "uncertain"
    else:
        risk_level = "critical"
        pathogenicity = "likely_pathogenic"

    # Add analysis factors
    risk_factors.insert(0, f"GC 함량: {gc_content:.1f}%")
    risk_factors.insert(0, f"서열 길이: {length}nt")

    recommendations = []
    if risk_level in ["high", "critical"]:
        recommendations.append("전문의 상담을 권장합니다")
        recommendations.append("추가 유전자 검사 고려")
    elif risk_level == "moderate":
        recommendations.append("정기적인 모니터링 권장")

    risk_assessment = RNARiskAssessment(
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        pathogenicity=pathogenicity,
        pathogenicity_confidence=0.7 if risk_level == "low" else 0.5,
        factors=risk_factors,
        recommendations=recommendations,
    )

    return RNAAnalysisResponse(
        sequence_analysis=SequenceAnalysis(
            length=length,
            gc_content=round(gc_content, 2),
            detected_rna_type=detected_type,
            rna_type_confidence=type_confidence,
            motifs_found=motifs,
        ),
        disease_predictions=disease_predictions,
        risk_assessment=risk_assessment,
    )


@router.post("/analyze", response_model=RNAAnalysisResponse)
async def analyze_rna(request: RNAAnalysisRequest):
    """
    Analyze RNA sequence for disease prediction.

    Takes an RNA sequence (A, U, G, C nucleotides) and predicts:
    - RNA type (mRNA, siRNA, circRNA, lncRNA)
    - Related diseases
    - Risk assessment
    - Pathogenicity prediction

    **Note**: This is a demonstration system. Always consult a medical
    professional for actual diagnosis.
    """
    # Validate sequence
    is_valid, normalized, invalid_chars = validate_sequence(request.sequence)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RNA sequence. Found invalid characters: {invalid_chars}",
        )

    if len(normalized) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RNA sequence must be at least 10 nucleotides long",
        )

    # Try to use ML model
    predictor = get_rna_predictor()

    if predictor is not None:
        try:
            result = predictor.predict(normalized, request.rna_type)

            # Enhance predictions with detailed information
            enhanced_predictions = []
            for pred in result["disease_predictions"]:
                disease_name = pred.get("disease", "")
                details = DISEASE_DETAILS.get(disease_name, {})
                enhanced_pred = {
                    **pred,
                    "description": pred.get("description") or details.get("description"),
                    "clinical_significance": pred.get("clinical_significance") or details.get("clinical_significance"),
                    "recommendation": pred.get("recommendation") or details.get("recommendation"),
                    "related_genes": pred.get("related_genes") or details.get("related_genes", []),
                }
                enhanced_predictions.append(enhanced_pred)

            # Enhance risk assessment based on disease predictions
            disease_weights = {
                "암 관련 RNA 이상": 3.0,
                "신경퇴행성 질환": 2.5,
                "유전성 근육 질환": 2.5,
                "UTR 변이 병원성": 2.0,
                "RNA 변이 관련 질환": 1.5,
                "siRNA 치료 반응 예측": 1.0,
                "ASO 효능 예측": 1.0,
            }

            weighted_risk = 0.0
            risk_factors = []
            seq_analysis = result["sequence_analysis"]
            risk_factors.append(f"서열 길이: {seq_analysis.get('length', 0)}nt")
            risk_factors.append(f"GC 함량: {seq_analysis.get('gc_content', 0):.1f}%")

            for pred in enhanced_predictions:
                disease_name = pred.get("disease", "")
                prob = pred.get("probability", 0)
                if disease_name != "정상/저위험":
                    weight = disease_weights.get(disease_name, 1.0)
                    weighted_risk += prob * weight * 100
                    if prob > 0.1:
                        risk_factors.append(f"{disease_name}: {prob*100:.1f}%")

            # Calculate enhanced risk score
            enhanced_risk_score = min(100, max(0, weighted_risk / 2))

            # Determine risk level
            if enhanced_risk_score < 20:
                risk_level = "low"
                pathogenicity = "benign"
            elif enhanced_risk_score < 40:
                risk_level = "moderate"
                pathogenicity = "likely_benign"
            elif enhanced_risk_score < 60:
                risk_level = "high"
                pathogenicity = "uncertain"
            else:
                risk_level = "critical"
                pathogenicity = "likely_pathogenic"

            recommendations = []
            if risk_level in ["high", "critical"]:
                recommendations.append("전문의 상담을 권장합니다")
                recommendations.append("추가 유전자 검사 고려")
            elif risk_level == "moderate":
                recommendations.append("정기적인 모니터링 권장")

            enhanced_risk_assessment = RNARiskAssessment(
                risk_score=round(enhanced_risk_score, 2),
                risk_level=risk_level,
                pathogenicity=pathogenicity,
                pathogenicity_confidence=0.7 if risk_level == "low" else 0.5,
                factors=risk_factors,
                recommendations=recommendations,
            )

            # Convert to response model
            return RNAAnalysisResponse(
                sequence_analysis=SequenceAnalysis(**result["sequence_analysis"]),
                disease_predictions=[
                    DiseasePrediction(**pred) for pred in enhanced_predictions
                ],
                risk_assessment=enhanced_risk_assessment,
                disclaimer=result.get("disclaimer", "이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요."),
            )
        except Exception as e:
            print(f"[RNA Analysis] Model prediction failed: {e}")
            # Fall back to mock prediction

    # Use mock prediction if model not available
    return create_mock_prediction(normalized, request.rna_type)


@router.post("/batch", response_model=RNABatchResponse)
async def analyze_batch(request: RNABatchRequest):
    """
    Batch analysis for multiple RNA sequences.

    Analyzes up to 100 sequences at once and returns individual results
    along with summary statistics.
    """
    results = []

    for seq in request.sequences:
        try:
            single_request = RNAAnalysisRequest(
                sequence=seq.sequence,
                rna_type=seq.rna_type,
            )
            result = await analyze_rna(single_request)
            results.append(result)
        except HTTPException:
            # Create error result
            results.append(create_mock_prediction(seq.sequence, seq.rna_type))

    # Calculate summary
    total = len(results)
    high_risk = sum(1 for r in results if r.risk_assessment.risk_level in ["high", "critical"])

    summary = {
        "total_sequences": total,
        "high_risk_count": high_risk,
        "avg_gc_content": sum(r.sequence_analysis.gc_content for r in results) / total if total > 0 else 0,
        "rna_type_distribution": {},
    }

    # Count RNA types
    for r in results:
        rna_type = r.sequence_analysis.detected_rna_type
        summary["rna_type_distribution"][rna_type] = summary["rna_type_distribution"].get(rna_type, 0) + 1

    return RNABatchResponse(results=results, summary=summary)


@router.post("/validate", response_model=RNAValidationResponse)
async def validate_rna_sequence(request: RNAValidationRequest):
    """
    Validate RNA sequence format.

    Checks if the sequence contains only valid nucleotides (A, U, G, C)
    and provides normalized sequence with statistics.
    """
    is_valid, normalized, invalid_chars = validate_sequence(request.sequence)
    gc_content = calculate_gc_content(normalized) if normalized else 0

    warnings = []
    if len(normalized) < 10:
        warnings.append("서열이 너무 짧습니다 (최소 10 뉴클레오타이드 권장)")
    if gc_content < 20 or gc_content > 80:
        warnings.append(f"비정상적인 GC 함량: {gc_content:.1f}%")

    return RNAValidationResponse(
        is_valid=is_valid,
        normalized_sequence=normalized if is_valid else None,
        length=len(normalized),
        gc_content=round(gc_content, 2),
        invalid_chars=invalid_chars,
        warnings=warnings,
    )


@router.get("/diseases", response_model=List[DiseaseInfo])
async def list_diseases():
    """
    List all detectable RNA-related diseases.

    Returns a list of diseases that can be predicted from RNA sequence analysis.
    """
    return DISEASES


@router.get("/rna-types", response_model=List[RNATypeInfo])
async def list_rna_types():
    """
    List supported RNA types.

    Returns information about all supported RNA types for classification.
    """
    return RNA_TYPES


def parse_fasta(content: str) -> List[dict]:
    """Parse FASTA format content."""
    sequences = []
    current_id = None
    current_seq = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_id is not None:
                sequences.append({
                    "id": current_id,
                    "sequence": "".join(current_seq),
                    "rna_type": "auto"
                })
            current_id = line[1:].split()[0]  # Get ID before first space
            current_seq = []
        elif line and current_id is not None:
            current_seq.append(line)

    # Don't forget the last sequence
    if current_id is not None:
        sequences.append({
            "id": current_id,
            "sequence": "".join(current_seq),
            "rna_type": "auto"
        })

    return sequences


@router.post("/upload/json", response_model=RNAFileUploadResponse)
async def upload_json_file(file: UploadFile = File(...)):
    """
    Upload JSON file containing RNA sequences for batch analysis.

    Expected JSON format:
    ```json
    {
        "sequences": [
            {
                "id": "seq1",
                "sequence": "AUGGCCAUGG...",
                "rna_type": "mRNA",
                "name": "Sample 1"
            }
        ]
    }
    ```
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a JSON file (.json)"
        )

    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON format: {str(e)}"
        )

    sequences = data.get("sequences", [])
    if not sequences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No sequences found in JSON file"
        )

    # Analyze sequences
    results = []
    for seq in sequences[:100]:  # Limit to 100 sequences
        try:
            request = RNAAnalysisRequest(
                sequence=seq.get("sequence", ""),
                rna_type=seq.get("rna_type", "auto"),
            )
            result = await analyze_rna(request)
            results.append(result)
        except HTTPException:
            continue

    # Calculate summary
    total = len(results)
    high_risk = sum(1 for r in results if r.risk_assessment.risk_level in ["high", "critical"])

    summary = {
        "total_sequences": total,
        "high_risk_count": high_risk,
        "avg_gc_content": sum(r.sequence_analysis.gc_content for r in results) / total if total > 0 else 0,
        "rna_type_distribution": {},
    }

    for r in results:
        rna_type = r.sequence_analysis.detected_rna_type
        summary["rna_type_distribution"][rna_type] = summary["rna_type_distribution"].get(rna_type, 0) + 1

    return RNAFileUploadResponse(
        filename=file.filename,
        format="json",
        sequences_found=len(sequences),
        sequences_analyzed=len(results),
        results=results,
        summary=summary,
    )


@router.post("/upload/fasta", response_model=RNAFileUploadResponse)
async def upload_fasta_file(file: UploadFile = File(...)):
    """
    Upload FASTA file containing RNA sequences for batch analysis.

    Expected FASTA format:
    ```
    >sequence_id_1
    AUGGCCAUGGCGCCCAGAACUGAG
    >sequence_id_2
    GCUGACUCCAAAGCUCUGCUU
    ```
    """
    if not file.filename.endswith((".fasta", ".fa", ".fna")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a FASTA file (.fasta, .fa, .fna)"
        )

    try:
        content = await file.read()
        sequences = parse_fasta(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse FASTA file: {str(e)}"
        )

    if not sequences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No sequences found in FASTA file"
        )

    # Analyze sequences
    results = []
    for seq in sequences[:100]:  # Limit to 100 sequences
        try:
            request = RNAAnalysisRequest(
                sequence=seq.get("sequence", ""),
                rna_type=seq.get("rna_type", "auto"),
            )
            result = await analyze_rna(request)
            results.append(result)
        except HTTPException:
            continue

    # Calculate summary
    total = len(results)
    high_risk = sum(1 for r in results if r.risk_assessment.risk_level in ["high", "critical"])

    summary = {
        "total_sequences": total,
        "high_risk_count": high_risk,
        "avg_gc_content": sum(r.sequence_analysis.gc_content for r in results) / total if total > 0 else 0,
        "rna_type_distribution": {},
    }

    for r in results:
        rna_type = r.sequence_analysis.detected_rna_type
        summary["rna_type_distribution"][rna_type] = summary["rna_type_distribution"].get(rna_type, 0) + 1

    return RNAFileUploadResponse(
        filename=file.filename,
        format="fasta",
        sequences_found=len(sequences),
        sequences_analyzed=len(results),
        results=results,
        summary=summary,
    )


@router.get("/sample-data", response_model=RNASampleDataResponse)
async def get_sample_data():
    """
    Get sample RNA sequence data for testing.

    Returns a collection of sample RNA sequences that can be used
    to test the analysis endpoints.
    """
    # Load from sample file if exists
    # Try multiple locations for test_data directory
    sample_file = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "test_data" / "sample_rna_sequences.json"
        if candidate.exists():
            sample_file = candidate
            break
    if sample_file is None:
        sample_file = Path("/app/test_data/sample_rna_sequences.json")

    if sample_file.exists():
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RNASampleDataResponse(
                sequences=data.get("sequences", []),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            pass

    # Return default sample data
    return RNASampleDataResponse(
        sequences=[
            {
                "id": "sample_mRNA_001",
                "name": "Sample mRNA",
                "sequence": "AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA",
                "rna_type": "mRNA",
                "description": "샘플 mRNA 서열",
            },
            {
                "id": "sample_siRNA_001",
                "name": "Sample siRNA",
                "sequence": "GCUGACUCCAAAGCUCUGCUU",
                "rna_type": "siRNA",
                "description": "샘플 siRNA 서열 (21nt)",
            },
            {
                "id": "sample_circRNA_001",
                "name": "Sample circRNA",
                "sequence": "AGCUAGCUAGCUUUAAACCCGGGAUUUCCAAAGGGCCCUUUAAAGCUAGCU",
                "rna_type": "circRNA",
                "description": "샘플 원형 RNA 서열",
            },
        ],
        metadata={
            "version": "1.0",
            "description": "기본 샘플 데이터",
        },
    )
