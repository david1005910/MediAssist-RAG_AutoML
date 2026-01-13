"""Symptom-based disease classification model."""

from typing import List, Dict, Optional
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from .ner import MedicalNER


class SymptomClassifier:
    """Ensemble classifier for disease prediction based on symptoms."""

    def __init__(self, model_path: Optional[str] = None):
        self.ner = MedicalNER()
        self.label_encoder = LabelEncoder()
        self.ensemble = None

        if model_path:
            self.load(model_path)

    def _extract_features(
        self,
        symptoms: List[Dict],
        patient_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """Extract feature vector from symptoms and patient info.

        Args:
            symptoms: List of symptom dictionaries with name, severity, duration.
            patient_info: Optional patient metadata (age, gender, history).

        Returns:
            Combined feature vector.
        """
        # Combine symptom text
        symptom_text = " ".join([s["name"] for s in symptoms])

        # Get BioBERT embeddings
        bert_embedding = self.ner.get_embeddings(symptom_text)

        # Patient info features
        patient_features = np.zeros(5)
        if patient_info:
            patient_features[0] = patient_info.get("age", 0) / 100
            patient_features[1] = 1 if patient_info.get("gender") == "male" else 0
            patient_features[2] = len(patient_info.get("medical_history", [])) / 10

        # Symptom numeric features
        symptom_features = np.array([
            np.mean([s.get("severity", 5) for s in symptoms]) / 10,
            np.mean([s.get("duration_days", 1) for s in symptoms]) / 30,
            len(symptoms) / 10,
        ])

        return np.concatenate([bert_embedding, patient_features, symptom_features])

    def train(self, X: List[Dict], y: List[str]) -> Dict[str, float]:
        """Train the ensemble model.

        Args:
            X: List of training samples with symptoms and patient_info.
            y: List of disease labels.

        Returns:
            Training metrics.
        """
        # Extract features
        X_features = np.array([
            self._extract_features(x["symptoms"], x.get("patient_info"))
            for x in X
        ])

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Build ensemble
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            random_state=42,
            eval_metric="mlogloss",
        )

        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )

        self.ensemble = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
            voting="soft",
            weights=[0.6, 0.4],
        )

        self.ensemble.fit(X_features, y_encoded)

        # Evaluate
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.ensemble, X_features, y_encoded, cv=5)

        return {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
        }

    def predict(
        self,
        symptoms: List[Dict],
        patient_info: Optional[Dict] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """Predict diseases from symptoms.

        Args:
            symptoms: List of symptom dictionaries.
            patient_info: Optional patient metadata.
            top_k: Number of top predictions to return.

        Returns:
            List of predictions with disease, probability, and confidence.
        """
        if self.ensemble is None:
            raise ValueError("Model not trained or loaded")

        features = self._extract_features(symptoms, patient_info).reshape(1, -1)
        probabilities = self.ensemble.predict_proba(features)[0]

        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]

            predictions.append({
                "disease": disease,
                "icd_code": self._get_icd_code(disease),
                "probability": float(prob),
                "confidence": self._get_confidence(prob),
            })

        return predictions

    def _get_confidence(self, prob: float) -> str:
        """Map probability to confidence level."""
        if prob >= 0.7:
            return "high"
        elif prob >= 0.4:
            return "medium"
        return "low"

    def _get_icd_code(self, disease: str) -> str:
        """Get ICD-10 code for a disease."""
        # TODO: Implement proper ICD-10 mapping
        icd_mapping = {
            "급성 상기도 감염": "J06.9",
            "인플루엔자": "J11",
            "폐렴": "J18.9",
        }
        return icd_mapping.get(disease, "R69")

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            "ensemble": self.ensemble,
            "label_encoder": self.label_encoder,
        }, f"{path}/classifier.pkl")

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(f"{path}/classifier.pkl")
        self.ensemble = data["ensemble"]
        self.label_encoder = data["label_encoder"]
