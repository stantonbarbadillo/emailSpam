import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from ml_model import classify_email, train_model, get_metrics, get_visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess the dataset
def load_dataset():
    df = pd.read_csv('email.csv')
    return df

@app.route('/')
def home():
    return 'Welcome to the Spam Classifier API!'

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    email_content = data.get('email_content')
    try:
        classification = classify_email(email_content)
        logging.info(f"Email classified as: {classification}")
        return jsonify({'classification': classification})
    except Exception as e:
        logging.error(f"Error classifying email: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    logging.info("Training model...")
    try:
        df = load_dataset()
        train_model(df)
        logging.info("Model trained successfully")
        return jsonify({'message': 'Model trained successfully'})
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        metrics = get_metrics()
        logging.info("Fetched model metrics")
        return jsonify(metrics)
    except Exception as e:
        logging.error(f"Error fetching metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations', methods=['GET'])
def visualizations():
    try:
        visualizations = get_visualizations()
        logging.info("Fetched visualizations")
        return jsonify(visualizations)
    except Exception as e:
        logging.error(f"Error fetching visualizations: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
