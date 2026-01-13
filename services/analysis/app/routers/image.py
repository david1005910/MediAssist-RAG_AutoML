"""Image analysis API router."""

import io
import base64
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np

from app.schemas.image import ImageAnalysisResponse, GradCAMResponse, Finding, ImageQuality

# Add models to path at module load time
MODELS_PATH = Path("/app/models")
if not MODELS_PATH.exists():
    MODELS_PATH = Path(__file__).parents[4] / "models"
if str(MODELS_PATH) not in sys.path:
    sys.path.insert(0, str(MODELS_PATH))

router = APIRouter(prefix="/api/v1/image", tags=["Image Analysis"])

# Global model instances (loaded on startup)
_image_analyzer = None
_gradcam = None


def get_analyzer():
    """Get or initialize image analyzer."""
    global _image_analyzer
    if _image_analyzer is None:
        from image_analyzer.model import ImageAnalyzer
        _image_analyzer = ImageAnalyzer()
    return _image_analyzer


def get_gradcam(model):
    """Get or initialize GradCAM."""
    global _gradcam
    if _gradcam is None:
        from image_analyzer.gradcam import GradCAM
        _gradcam = GradCAM(model.model, "backbone.features.denseblock4")
    return _gradcam


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    generate_gradcam: bool = Query(False, description="Generate Grad-CAM visualization"),
):
    """
    Analyze a chest X-ray image for medical conditions.

    Supported conditions:
    - 정상 (Normal)
    - 폐렴 (Pneumonia)
    - 결핵 (Tuberculosis)
    - 폐암 의심 (Suspected Lung Cancer)
    - 심비대 (Cardiomegaly)
    - 기흉 (Pneumothorax)
    - 폐부종 (Pulmonary Edema)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file to temp location
    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Analyze image
        analyzer = get_analyzer()
        result = analyzer.analyze(tmp_path)

        # Convert to response model
        findings = [
            Finding(
                condition=f["condition"],
                probability=f["probability"],
                confidence=f["confidence"]
            )
            for f in result["findings"]
        ]

        quality = ImageQuality(**result["image_quality"])

        return ImageAnalysisResponse(
            findings=findings,
            image_quality=quality,
            gradcam_available=generate_gradcam and len(findings) > 0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup temp file
        try:
            Path(tmp_path).unlink()
        except:
            pass


@router.post("/gradcam", response_model=GradCAMResponse)
async def generate_gradcam_visualization(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    target_condition: Optional[str] = Query(
        None,
        description="Target condition for visualization (default: highest probability)"
    ),
):
    """
    Generate Grad-CAM visualization for a chest X-ray image.

    Returns heatmap showing which regions the model focuses on for the diagnosis.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Get analyzer and run analysis first
        analyzer = get_analyzer()
        result = analyzer.analyze(tmp_path)

        if not result["findings"]:
            raise HTTPException(status_code=400, detail="No findings detected in image")

        # Determine target condition
        from image_analyzer.model import ChestXrayAnalyzer
        conditions = ChestXrayAnalyzer.CONDITIONS

        if target_condition:
            if target_condition not in conditions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid condition. Valid: {conditions}"
                )
            target_idx = conditions.index(target_condition)
            target_name = target_condition
        else:
            # Use highest probability finding
            target_name = result["findings"][0]["condition"]
            target_idx = conditions.index(target_name)

        # Generate Grad-CAM
        gradcam = get_gradcam(analyzer)

        # Prepare input tensor
        image = Image.open(tmp_path).convert("RGB")
        input_tensor = analyzer.transform(image).unsqueeze(0).to(analyzer.device)
        input_tensor.requires_grad = True

        # Generate heatmap
        heatmap = gradcam.generate(input_tensor, target_idx)

        # Generate overlay
        overlay_image = gradcam.overlay(tmp_path, heatmap, alpha=0.5)

        # Convert heatmap to image
        import cv2
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(heatmap, (224, 224))),
            cv2.COLORMAP_JET
        )
        heatmap_image = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))

        # Encode images to base64
        def image_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        return GradCAMResponse(
            condition=target_name,
            heatmap_base64=image_to_base64(heatmap_image),
            overlay_base64=image_to_base64(overlay_image)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")
    finally:
        try:
            Path(tmp_path).unlink()
        except:
            pass


@router.get("/conditions")
async def list_conditions():
    """List all detectable conditions."""
    from image_analyzer.model import ChestXrayAnalyzer
    return {
        "conditions": [
            {"name": c, "index": i}
            for i, c in enumerate(ChestXrayAnalyzer.CONDITIONS)
        ]
    }
