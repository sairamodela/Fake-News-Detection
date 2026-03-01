"""
Fake News Detection - Prediction Module
Author: Sairam Odela
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_model():
    model_path = os.path.join(MODEL_DIR, 'fake_news_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run src/train.py first.")
    return joblib.load(model_path)


def predict_news(text: str) -> dict:
    """
    Predict whether a news article is Real or Fake.

    Args:
        text: Raw news text (title + body or just body).

    Returns:
        {
          'label'     : 'Real' | 'Fake',
          'confidence': float (0–100),
          'real_prob' : float,
          'fake_prob' : float,
        }
    """
    pipeline = load_model()

    cleaned   = clean_text(text)
    proba     = pipeline.predict_proba([cleaned])[0]  # [P(fake), P(real)]
    fake_prob = proba[0]
    real_prob = proba[1]
    label     = 'Real' if real_prob >= 0.5 else 'Fake'
    confidence = max(real_prob, fake_prob) * 100

    return {
        'label'     : label,
        'confidence': round(confidence, 2),
        'real_prob' : round(real_prob * 100, 2),
        'fake_prob' : round(fake_prob * 100, 2),
    }


if __name__ == '__main__':
    sample = """
    NASA confirms water ice discovered on the moon's surface.
    Scientists at the agency announced today that direct evidence of water ice
    has been found in permanently shadowed craters near the lunar poles.
    """
    result = predict_news(sample)
    print(f"Prediction  : {result['label']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Real Prob   : {result['real_prob']}%")
    print(f"Fake Prob   : {result['fake_prob']}%")
