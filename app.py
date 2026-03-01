"""
Fake News Detection - Flask Web Application
Author: Sairam Odela
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_news

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or not data.get('text', '').strip():
        return jsonify({'error': 'Please provide news text to analyse.'}), 400

    text = data['text'].strip()
    if len(text) < 30:
        return jsonify({'error': 'Text is too short. Please provide at least a sentence.'}), 400

    try:
        result = predict_news(text)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Please train the model first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
