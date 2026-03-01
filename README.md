# 📰 Fake News Detection Using Machine Learning

A web application that detects whether a news article is **Real** or **Fake** using Logistic Regression and TF-IDF vectorisation, trained on 40,000+ news articles.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Algorithm** | Logistic Regression |
| **Features** | TF-IDF (unigrams + bigrams, 50K features) |
| **Dataset** | Kaggle Fake and Real News Dataset (~44K articles) |
| **Interface** | Flask Web Application |
| **Language** | Python 3.10+ |

---

## 🗂️ Project Structure

```
fake-news-detection/
├── app.py                  # Flask web application
├── requirements.txt
├── src/
│   ├── preprocess.py       # Text cleaning pipeline
│   ├── train.py            # Model training
│   └── predict.py          # Prediction module
├── models/
│   └── fake_news_model.pkl # Trained pipeline (TF-IDF + LR)
├── templates/
│   └── index.html          # Bootstrap web UI
├── notebooks/
│   └── fake_news_detection.ipynb  # EDA + training walkthrough
└── data/
    ├── Fake.csv            # Fake news articles
    └── True.csv            # Real news articles
```

---

## 🧠 How It Works

### Text Preprocessing
1. Lowercase conversion
2. URL and HTML tag removal
3. Punctuation and digit removal
4. Whitespace normalisation

### Feature Engineering
- **TF-IDF Vectoriser** with 50,000 features
- **N-gram range**: unigrams + bigrams `(1, 2)`
- **Sublinear TF scaling** for better generalisation
- Title and body text are **combined** for richer context

### Model
- **Logistic Regression** — fast, interpretable, highly effective for text
- Trained as a **scikit-learn Pipeline** (TF-IDF → LR) for clean inference
- **5-fold cross-validation** for robust evaluation

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/sairamodela/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place in `data/`:
```
data/Fake.csv
data/True.csv
```

### 4. Train the model
```bash
python src/train.py
```

### 5. Run the web app
```bash
python app.py
```
Open your browser at `http://localhost:5001`

---

## 📊 Results

| Metric | Score |
|---|---|
| **Accuracy** | ~98.7% |
| **Precision (Real)** | ~99% |
| **Recall (Real)** | ~98% |
| **5-Fold CV Accuracy** | ~98.5% |

> Results are consistent with published benchmarks on this dataset.

---

## 🖥️ Web Application Features

- Paste any news article text
- One-click analysis with probability scores
- Real / Fake example articles to try instantly
- Clean Bootstrap 5 UI with colour-coded results
- Character counter and input validation

---

## 🛠️ Tech Stack

- **Python** — Core language
- **scikit-learn** — TF-IDF, Logistic Regression, Pipeline
- **Flask** — Web framework
- **Bootstrap 5** — Frontend UI
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn / WordCloud** — EDA visualisations

---

## 👤 Author

**Sairam Odela**  
[LinkedIn](https://www.linkedin.com/in/sairam-odela-801462250/) · [Portfolio](https://sairamodela.github.io/sai-portfolio/)
