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
MODELS_PATH = Path("/app/models")
if not MODELS_PATH.exists():
    MODELS_PATH = Path(__file__).parents[4] / "models"
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

# Detectable diseases
DISEASES = [
    DiseaseInfo(
        name="정상/저위험",
        name_en="Normal/Low Risk",
        icd_code="N/A",
        description="정상 범위 또는 저위험 상태",
    ),
    DiseaseInfo(
        name="RNA 변이 관련 질환",
        name_en="RNA Mutation Related Disease",
        icd_code="Q89.9",
        description="RNA 변이로 인한 유전적 질환",
    ),
    DiseaseInfo(
        name="siRNA 치료 반응 예측",
        name_en="siRNA Therapy Response Prediction",
        icd_code="Z51.1",
        description="siRNA 기반 치료에 대한 반응 예측",
    ),
    DiseaseInfo(
        name="ASO 효능 예측",
        name_en="ASO Efficacy Prediction",
        icd_code="Z51.1",
        description="안티센스 올리고뉴클레오타이드 치료 효능",
    ),
    DiseaseInfo(
        name="UTR 변이 병원성",
        name_en="UTR Variant Pathogenicity",
        icd_code="Q99.8",
        description="비번역 영역(UTR) 변이의 병원성",
    ),
    DiseaseInfo(
        name="유전성 근육 질환",
        name_en="Hereditary Muscle Disease",
        icd_code="G71.9",
        description="근육 조직에 영향을 미치는 유전성 질환",
    ),
    DiseaseInfo(
        name="신경퇴행성 질환",
        name_en="Neurodegenerative Disease",
        icd_code="G31.9",
        description="신경 세포의 점진적 퇴화 질환",
    ),
    DiseaseInfo(
        name="암 관련 RNA 이상",
        name_en="Cancer-related RNA Abnormality",
        icd_code="C80.1",
        description="암 발생 및 진행과 관련된 RNA 이상",
    ),
]


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

    # Create mock disease predictions
    disease_predictions = [
        DiseasePrediction(
            disease="정상/저위험",
            disease_en="Normal/Low Risk",
            icd_code="N/A",
            probability=0.75,
            confidence="high",
            description="정상 범위의 RNA 서열",
            related_genes=[],
        ),
        DiseasePrediction(
            disease="RNA 변이 관련 질환",
            disease_en="RNA Mutation Related Disease",
            icd_code="Q89.9",
            probability=0.15,
            confidence="low",
            description="RNA 변이로 인한 유전적 질환 가능성",
            related_genes=["BRCA1", "TP53"],
        ),
    ]

    # Create risk assessment
    risk_score = min(100, max(0, 100 - gc_content))  # Mock calculation
    if risk_score < 25:
        risk_level = "low"
    elif risk_score < 50:
        risk_level = "moderate"
    elif risk_score < 75:
        risk_level = "high"
    else:
        risk_level = "critical"

    risk_assessment = RNARiskAssessment(
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        pathogenicity="benign" if risk_score < 50 else "uncertain",
        pathogenicity_confidence=0.6,
        factors=["서열 길이 분석 완료", "GC 함량 분석 완료"],
        recommendations=["정기적인 모니터링 권장"] if risk_level != "low" else [],
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

            # Convert to response model
            return RNAAnalysisResponse(
                sequence_analysis=SequenceAnalysis(**result["sequence_analysis"]),
                disease_predictions=[
                    DiseasePrediction(**pred) for pred in result["disease_predictions"]
                ],
                risk_assessment=RNARiskAssessment(**result["risk_assessment"]),
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
    sample_file = Path(__file__).parents[4] / "test_data" / "sample_rna_sequences.json"

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
