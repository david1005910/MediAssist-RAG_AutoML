"""Chest X-ray analysis model using DenseNet121."""

from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ChestXrayAnalyzer(nn.Module):
    """DenseNet121-based chest X-ray analyzer."""

    CONDITIONS = [
        "정상", "폐렴", "결핵", "폐암 의심",
        "심비대", "기흉", "폐부종"
    ]

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )

        self.gradients = None
        self.activations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def activations_hook(self, grad):
        """Hook for Grad-CAM."""
        self.gradients = grad


class ImageAnalyzer:
    """Image analysis service wrapper."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChestXrayAnalyzer().to(self.device)

        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def analyze(self, image_path: str) -> Dict:
        """Analyze a chest X-ray image.

        Args:
            image_path: Path to the image file.

        Returns:
            Analysis results including findings and quality assessment.
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        probabilities = outputs.cpu().numpy()[0]

        findings = []
        for condition, prob in zip(ChestXrayAnalyzer.CONDITIONS, probabilities):
            if prob > 0.3:
                findings.append({
                    "condition": condition,
                    "probability": float(prob),
                    "confidence": "high" if prob > 0.7 else "medium" if prob > 0.5 else "low",
                })

        findings.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "findings": findings,
            "image_quality": self._assess_quality(image_path),
        }

    def _assess_quality(self, image_path: str) -> Dict:
        """Assess image quality metrics."""
        try:
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return {
                "resolution": list(image.shape),
                "brightness": float(np.mean(image)),
                "contrast": float(np.std(image)),
                "is_acceptable": np.std(image) > 30,
            }
        except Exception:
            return {
                "resolution": [0, 0],
                "brightness": 0.0,
                "contrast": 0.0,
                "is_acceptable": False,
            }
