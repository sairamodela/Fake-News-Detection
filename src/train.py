"""
Fake News Detection - Model Training Pipeline
Dataset: Kaggle Fake/Real News Dataset
Algorithm: Logistic Regression with TF-IDF
Author: Sairam Odela
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from preprocess import clean_text

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_kaggle_dataset(data_dir: str) -> pd.DataFrame:
    """
    Expects two CSV files inside data/:
        Fake.csv  — columns: title, text, subject, date
        True.csv  — columns: title, text, subject, date

    From: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    fake_path = os.path.join(data_dir, 'Fake.csv')
    true_path = os.path.join(data_dir, 'True.csv')

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            "Dataset not found.\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "Place Fake.csv and True.csv inside the data/ folder."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0  # 0 = Fake
    true_df['label'] = 1  # 1 = Real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"  Total samples : {len(df)}")
    print(f"  Fake news     : {(df['label'] == 0).sum()}")
    print(f"  Real news     : {(df['label'] == 1).sum()}")
    return df


def train():
    print("=" * 55)
    print("  Fake News Detection — Training Pipeline")
    print("=" * 55)

    print("\n[1/5] Loading dataset …")
    df = load_kaggle_dataset(DATA_DIR)

    print("\n[2/5] Preprocessing text …")
    # Combine title + text for richer features
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    print(f"  Sample cleaned text: {df['content'].iloc[0][:120]} …")

    print("\n[3/5] Splitting dataset …")
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("\n[4/5] Training Logistic Regression with TF-IDF …")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
            min_df=2,
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    print("\n[5/5] Evaluating …")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model pipeline
    model_path = os.path.join(MODEL_DIR, 'fake_news_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\n  Model saved → {model_path}")
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
