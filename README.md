⸻

## Fake News Detection Using Machine Learning

A web application that classifies news articles as Real or Fake using Logistic Regression and TF-IDF text features, trained on a large public news dataset.


## Project Overview

Item	Detail
Algorithm	Logistic Regression
Features	TF-IDF (unigrams + bigrams)
Dataset	Kaggle Fake and Real News Dataset (~44K articles)
Interface	Flask Web Application
Language	Python 3.10+



## Project Structure

fake-news-detection/
├── app.py                  # Flask web application
├── requirements.txt
├── src/
│   ├── preprocess.py       # Text preprocessing
│   ├── train.py            # Model training
│   └── predict.py          # Prediction logic
├── models/                 # Trained model (generated locally)
├── templates/
│   └── index.html          # Web UI
├── notebooks/              # EDA and experiments
└── data/                   # Dataset (not tracked)



## How It Works

Text Preprocessing
	•	Lowercasing
	•	URL and HTML removal
	•	Punctuation and digit removal
	•	Whitespace normalization

Feature Engineering
	•	TF-IDF vectorization with unigrams and bigrams
	•	Title and body text combined for context

Model
	•	Logistic Regression implemented as a scikit-learn Pipeline
	•	Probability-based predictions for Real vs Fake


## Getting Started

1. Clone the repository

git clone https://github.com/sairamodela/fake-news-detection.git
cd fake-news-detection

2. Install dependencies

pip install -r requirements.txt

3. Download the dataset

Download from Kaggle and place files in data/:

data/Fake.csv
data/True.csv

4. Train the model

python src/train.py

5. Run the application

python app.py

Open http://localhost:5001


## Results

Metric	Score
Accuracy	~98–99%
Precision (Real)	~99%
Recall (Real)	~98%



## Web Application Features
	•	Paste news article text for prediction
	•	Real/Fake classification with probability scores
	•	Example inputs for quick testing
	•	Simple, responsive Bootstrap UI


## Tech Stack
	•	Python
	•	scikit-learn
	•	Flask
	•	Bootstrap
	•	Pandas, NumPy


## Author

Sairam Odela
LinkedIn: https://www.linkedin.com/in/sairam-odela-801462250/
Portfolio: https://sairamodela.github.io/sai-portfolio/