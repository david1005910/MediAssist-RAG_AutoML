"""Risk prediction model for patient severity assessment."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np


class RiskPredictorNetwork(nn.Module):
    """Neural network for risk score prediction."""

    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * 100  # Scale to 0-100


class RiskPredictor:
    """Risk prediction service."""

    RISK_LEVELS = {
        (0, 25): "low",
        (25, 50): "moderate",
        (50, 75): "high",
        (75, 100): "critical",
    }

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RiskPredictorNetwork().to(self.device)

        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        self.model.eval()

    def predict(
        self,
        features: np.ndarray,
        symptoms: List[Dict],
        patient_info: Optional[Dict] = None,
    ) -> Dict:
        """Predict risk score and level.

        Args:
            features: Feature vector from symptom classifier.
            symptoms: List of symptom dictionaries.
            patient_info: Optional patient metadata.

        Returns:
            Risk assessment with score, level, and contributing factors.
        """
        # Prepare input
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            score = self.model(input_tensor).item()

        # Determine risk level
        level = self._get_risk_level(score)

        # Identify contributing factors
        factors = self._identify_risk_factors(symptoms, patient_info, score)

        return {
            "score": round(score, 1),
            "level": level,
            "factors": factors,
        }

    def _get_risk_level(self, score: float) -> str:
        """Map score to risk level."""
        for (low, high), level in self.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return "critical"

    def _identify_risk_factors(
        self,
        symptoms: List[Dict],
        patient_info: Optional[Dict],
        score: float,
    ) -> List[str]:
        """Identify factors contributing to risk score."""
        factors = []

        # Check for high severity symptoms
        high_severity = [s for s in symptoms if s.get("severity", 0) >= 8]
        if high_severity:
            factors.append(f"고심각도 증상 {len(high_severity)}개")

        # Check for prolonged symptoms
        prolonged = [s for s in symptoms if s.get("duration_days", 0) >= 7]
        if prolonged:
            factors.append("장기간 지속되는 증상")

        # Check patient age
        if patient_info:
            age = patient_info.get("age", 0)
            if age >= 65:
                factors.append("고령 환자")
            elif age <= 5:
                factors.append("영유아 환자")

            # Check medical history
            history = patient_info.get("medical_history", [])
            chronic = ["당뇨", "고혈압", "심장질환", "폐질환"]
            matching = [h for h in history if any(c in h for c in chronic)]
            if matching:
                factors.append(f"기저질환: {', '.join(matching)}")

        return factors

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(self.model.state_dict(), f"{path}/risk_predictor.pt")

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model.load_state_dict(
            torch.load(f"{path}/risk_predictor.pt", map_location=self.device)
        )
