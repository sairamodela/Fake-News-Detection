# src/predict.py
import os
import joblib
from typing import Dict
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

_pipeline = None  # cache model


def load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run src/train.py first.")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_news(text: str) -> Dict[str, float | str]:
    pipeline = load_model()

    cleaned = clean_text(text)
    proba = pipeline.predict_proba([cleaned])[0]

    fake_prob, real_prob = proba
    label = "Real" if real_prob >= 0.5 else "Fake"

    return {
        "label": label,
        "confidence": round(max(real_prob, fake_prob) * 100, 2),
        "real_prob": round(real_prob * 100, 2),
        "fake_prob": round(fake_prob * 100, 2),
    }