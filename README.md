# Fake News Detection System

## Overview

The Fake News Detection System is a machine learning-based project designed to identify whether a news article is real or fake. The system uses Python for data preprocessing, feature extraction, and model training, and Flask for serving predictions through a web application.

## Features

- **Data Preprocessing**: Cleans and prepares news data by removing punctuation, stop words, and applying TF-IDF vectorization.
- **Machine Learning Model**: Utilizes a logistic regression model to classify news articles as real or fake.
- **Web Application**: A Flask-based web app where users can input news articles and receive real-time predictions.
- **Model Persistence**: Uses `pickle` to save and load the trained model and vectorizer.

## Project Structure
Fake-News-Detection/
│
├── app.py                        # Main Flask application
├── requirements.txt              # List of dependencies
├── fake_news_model.pkl           # Serialized ML model using pickle
├── tfidf_vectorizer.pkl          # Serialized TF-IDF vectorizer using pickle
├── templates/                    # HTML template files for the Flask app
│   ├── index.html                # Home page with input form for news article
│   └── result.html               # Page displaying the prediction result
├── static/                       # Folder for static files like CSS, JS, images
│   └── styles.css                # Optional CSS file for styling
├── data/                         # Folder for datasets
│   ├── True.csv                  # Real news dataset
│   └── Fake.csv                  # Fake news dataset
├── model_training.ipynb          # Jupyter notebook for model training and evaluation
└── README.md                     # Project documentation and usage instructions


## Installation

Clone the repository:
   bash
   git clone https://github.com/yourusername/Fake-News-Detection.git
   cd Fake-News-Detection
   pip install -r requirements.txt
   Usage
Training the Model:

Open model_training.ipynb in Jupyter Notebook to train the model and save it using pickle.
Running the Flask Application:

Start the Flask application by running:
bash

python app.py
Navigate to http://127.0.0.1:5000/ in your web browser to access the application.
Interacting with the Web App:

On the home page (index.html), enter a news article in the provided form.
Submit the form to receive a prediction on whether the news is real or fake.

**Dependencies**
The project requires the following Python packages:
Flask
scikit-learn
pandas
nltk
pickle5
gunicorn (for production server)
Install them using the requirements.txt file provided


