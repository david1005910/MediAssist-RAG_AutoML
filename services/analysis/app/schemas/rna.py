"""RNA Analysis API Schemas."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class RNASequence(BaseModel):
    """Single RNA sequence input."""
    sequence: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="RNA sequence (A, U, G, C nucleotides)",
    )
    rna_type: Optional[str] = Field(
        default=None,
        description="RNA type hint: mRNA, siRNA, circRNA, lncRNA",
    )
    name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional sequence identifier",
    )


class RNAAnalysisRequest(BaseModel):
    """Request for RNA sequence analysis."""
    sequence: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="RNA sequence (A, U, G, C nucleotides)",
    )
    rna_type: Optional[str] = Field(
        default=None,
        description="RNA type hint: mRNA, siRNA, circRNA, lncRNA, or auto for detection",
    )
    include_structure_prediction: bool = Field(
        default=False,
        description="Include secondary structure prediction (slower)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAG",
                    "rna_type": "mRNA",
                    "include_structure_prediction": False,
                }
            ]
        }
    }


class RNABatchRequest(BaseModel):
    """Batch analysis request for multiple sequences."""
    sequences: List[RNASequence] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of RNA sequences to analyze",
    )


class RNAValidationRequest(BaseModel):
    """Request to validate RNA sequence."""
    sequence: str = Field(..., description="Sequence to validate")


class DiseasePrediction(BaseModel):
    """Single disease prediction."""
    disease: str = Field(..., description="Disease name (Korean)")
    disease_en: str = Field(..., description="Disease name (English)")
    icd_code: str = Field(..., description="ICD-10 code")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    description: Optional[str] = Field(default=None, description="Disease description")
    clinical_significance: Optional[str] = Field(
        default=None, description="Clinical significance of the finding"
    )
    recommendation: Optional[str] = Field(
        default=None, description="Medical recommendations"
    )
    related_genes: List[str] = Field(default_factory=list, description="Related genes")


class RNARiskAssessment(BaseModel):
    """Risk assessment for RNA sequence."""
    risk_score: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_level: str = Field(..., description="Risk level: low, moderate, high, critical")
    pathogenicity: str = Field(
        ...,
        description="Pathogenicity: benign, likely_benign, uncertain, likely_pathogenic, pathogenic",
    )
    pathogenicity_confidence: float = Field(
        ..., ge=0, le=1, description="Pathogenicity prediction confidence"
    )
    factors: List[str] = Field(default_factory=list, description="Risk factors identified")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations based on risk"
    )


class SequenceAnalysis(BaseModel):
    """Sequence quality and characteristics analysis."""
    length: int = Field(..., description="Sequence length in nucleotides")
    gc_content: float = Field(..., ge=0, le=100, description="GC content percentage")
    detected_rna_type: str = Field(..., description="Detected RNA type")
    rna_type_confidence: float = Field(
        ..., ge=0, le=1, description="RNA type detection confidence"
    )
    motifs_found: List[str] = Field(
        default_factory=list, description="Known RNA motifs found"
    )


class RNAAnalysisResponse(BaseModel):
    """Response from RNA sequence analysis."""
    sequence_analysis: SequenceAnalysis
    disease_predictions: List[DiseasePrediction]
    risk_assessment: RNARiskAssessment
    disclaimer: str = Field(
        default="이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요.",
        description="Medical disclaimer",
    )


class RNABatchResponse(BaseModel):
    """Batch analysis response."""
    results: List[RNAAnalysisResponse]
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )
    disclaimer: str = Field(
        default="이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요.",
    )


class RNAValidationResponse(BaseModel):
    """Sequence validation response."""
    is_valid: bool = Field(..., description="Whether sequence is valid RNA")
    normalized_sequence: Optional[str] = Field(
        default=None, description="Normalized sequence (uppercase, T->U)"
    )
    length: int = Field(..., description="Sequence length")
    gc_content: float = Field(..., description="GC content percentage")
    invalid_chars: List[str] = Field(
        default_factory=list, description="Invalid characters found"
    )
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class RNATypeInfo(BaseModel):
    """Information about an RNA type."""
    code: str = Field(..., description="RNA type code")
    name: str = Field(..., description="Full name")
    description: str = Field(..., description="Description")


class DiseaseInfo(BaseModel):
    """Information about a detectable disease."""
    name: str = Field(..., description="Disease name (Korean)")
    name_en: str = Field(..., description="Disease name (English)")
    icd_code: str = Field(..., description="ICD-10 code")
    description: Optional[str] = Field(default=None, description="Disease description")
    clinical_significance: Optional[str] = Field(
        default=None, description="Clinical significance"
    )
    recommendation: Optional[str] = Field(default=None, description="Recommendations")
    related_genes: List[str] = Field(default_factory=list, description="Related genes")


class RNAFileUploadResponse(BaseModel):
    """Response from file upload analysis."""
    filename: str = Field(..., description="Uploaded filename")
    format: str = Field(..., description="File format (json, fasta)")
    sequences_found: int = Field(..., description="Number of sequences found")
    sequences_analyzed: int = Field(..., description="Number of sequences analyzed")
    results: List[RNAAnalysisResponse]
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    disclaimer: str = Field(
        default="이 분석 결과는 참고용이며, 정확한 진단을 위해 반드시 의료 전문가와 상담하세요.",
    )


class RNASampleDataResponse(BaseModel):
    """Response with sample RNA data."""
    sequences: List[Dict[str, Any]] = Field(..., description="List of sample sequences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
