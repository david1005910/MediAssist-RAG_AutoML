"""Schemas for image analysis API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageQuality(BaseModel):
    """Image quality assessment."""
    resolution: List[int] = Field(description="Image resolution [height, width]")
    brightness: float = Field(description="Average brightness")
    contrast: float = Field(description="Image contrast (std)")
    is_acceptable: bool = Field(description="Whether quality is acceptable")


class Finding(BaseModel):
    """Single finding from image analysis."""
    condition: str = Field(description="Detected condition name (Korean)")
    condition_en: Optional[str] = Field(default=None, description="Detected condition name (English)")
    probability: float = Field(ge=0, le=1, description="Detection probability")
    confidence: str = Field(description="Confidence level: high/medium/low")
    description: Optional[str] = Field(default=None, description="Detailed description of the condition")
    clinical_significance: Optional[str] = Field(default=None, description="Clinical significance")
    recommendation: Optional[str] = Field(default=None, description="Medical recommendations")


class ImageAnalysisResponse(BaseModel):
    """Response from image analysis endpoint."""
    findings: List[Finding] = Field(description="List of detected conditions")
    image_quality: ImageQuality = Field(description="Image quality assessment")
    gradcam_available: bool = Field(default=False, description="Whether Grad-CAM is available")


class GradCAMResponse(BaseModel):
    """Response from Grad-CAM visualization endpoint."""
    condition: str = Field(description="Target condition for visualization")
    heatmap_base64: str = Field(description="Base64 encoded heatmap image")
    overlay_base64: str = Field(description="Base64 encoded overlay image")
