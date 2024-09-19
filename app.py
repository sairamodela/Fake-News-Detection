# app.py
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the Logistic Regression model and the fitted
path = "D:/Fake News Detection/"
model = pickle.load(open(path + 'fake_news_model.pkl', 'rb'))
vec = pickle.load(open(path + 'vec.pkl', 'rb'))

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the news input from the form
        news = request.form['news']
        
        # Clean and transform the input using the fitted
        data = vec.transform([news])
        
        # Make the prediction using the Logistic Regression model
        prediction = model.predict(data)
        
        # Return 'Real' if 1, 'Fake' if 0
        result = 'Real' if prediction == 1 else 'Fake'
        
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
