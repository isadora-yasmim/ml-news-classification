from pathlib import Path
from typing import Any

import joblib


MODEL_PATH = Path("models/best_model.pkl")


class NewsPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {self.model_path}. "
                "Treine e salve o modelo antes de executar a predição."
            )

        return joblib.load(self.model_path)

    @staticmethod
    def build_text(headline: str, short_description: str) -> str:
        headline = headline.strip()
        short_description = short_description.strip()

        if not headline and not short_description:
            raise ValueError("Headline e short_description não podem estar vazios.")

        return f"{headline} {short_description}".strip()

    def predict(self, headline: str, short_description: str) -> dict[str, Any]:
        text = self.build_text(headline, short_description)

        predicted_category = self.model.predict([text])[0]

        result = {
            "predicted_category": predicted_category,
            "confidence": None,
        }

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            result["confidence"] = round(float(probabilities.max()), 4)

        return result