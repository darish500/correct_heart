"""
Heart Attack Risk Prediction System using Flask.

This Flask app provides a REST API for predicting the risk of a heart attack
based on user-provided input features. The app dynamically downloads a
pre-trained machine learning model from Google Drive if it's not already
present locally.

Endpoints:
- `/` (GET): Returns a welcome message.
- `/predict` (POST): Accepts JSON input, preprocesses it, and returns a
  prediction result.

Author: Rasak Khalid
Date: January 2025
"""

import joblib
import numpy as np
import os
import gdown
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Google Drive file ID for the model
MODEL_URL = "https://drive.google.com/uc?id=1D6tor14R0jFqhvAhiOxPO_q2u26lK-kw"
MODEL_PATH = "heart_attack_risk_model.pkl"

# Function to download the model from Google Drive if it doesn't already exist
def download_model():
    """Download the trained model from Google Drive if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded.")

# Load the trained model
download_model()
model = joblib.load(MODEL_PATH)

def preprocess_input(data):
    """
    Preprocess the input data for model prediction.

    Parameters:
    - data (dict): A dictionary containing input features.

    Returns:
    - np.array: A numpy array formatted for model prediction.
    """
    input_features = [
        data['Age'],
        data['Gender'],
        data['Smoking'],
        data['Alcohol_Consumption'],
        data['Physical_Activity_Level'],
        data['BMI'],
        data['Diabetes'],
        data['Hypertension'],
        data['Cholesterol_Level'],
        data['Resting_BP'],
        data['Heart_Rate'],
        data['Family_History'],
        data['Stress_Level'],
        data['Chest_Pain_Type'],
        data['Thalassemia'],
        data['Fasting_Blood_Sugar'],
        data['ECG_Results'],
        data['Exercise_Induced_Angina'],
        data['Max_Heart_Rate_Achieved']
    ]
    return np.array(input_features).reshape(1, -1)

@app.route('/')
def home():
    """Home endpoint providing a welcome message."""
    return "Welcome to the Heart Attack Risk Prediction System!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.

    Accepts a POST request with JSON data, preprocesses the data,
    and returns a prediction result.

    Returns:
    - JSON: A dictionary containing the prediction or an error message.
    """
    try:
        data = request.json
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
