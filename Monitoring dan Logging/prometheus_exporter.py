"""
Model Serving with Prometheus Metrics
Author: Putu Linda Suryantini
Purpose: Kriteria 4 - MSML Submission
"""

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'model_prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['prediction_class']
)

ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Number of active prediction requests'
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

# Global variables
model = None
label_encoders = None
scaler = None


def load_model_and_artifacts():
    """
    Load trained model and preprocessing artifacts
    """
    global model, label_encoders, scaler

    start_time = time.time()

    try:
        # Load model from MLflow
        # For demo purposes, we'll load the latest run
        # In production, you would specify the exact model URI
        model_path = os.environ.get('MODEL_PATH', 'mlruns/0/latest/artifacts/model')

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = mlflow.sklearn.load_model(model_path)
        else:
            print("Model path not found, using dummy model for demo")
            # For demo: create a simple dummy model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # === PERBAIKAN: LATIH MODEL DUMMY SUPAYA TIDAK ERROR ===
            # Kita buat data latih palsu dengan 19 fitur (sesuai baris 131)
            print("Training dummy model...")
            X_dummy = [[0] * 19, [1] * 19]  # 2 sampel, 19 fitur
            y_dummy = [0, 1]                # 2 label target
            model.fit(X_dummy, y_dummy)     # <--- INI KUNCINYA (FIT)
            print("Dummy model fitted successfully!")
            # =======================================================


        # Load preprocessing artifacts
        artifacts_dir = 'preprocessing/artifacts'

        if os.path.exists(os.path.join(artifacts_dir, 'label_encoders.pkl')):
            with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'rb') as f:
                label_encoders = pickle.load(f)
            print("Label encoders loaded")

        if os.path.exists(os.path.join(artifacts_dir, 'scaler.pkl')):
            with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded")

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)

        print(f"Model and artifacts loaded successfully in {load_time:.2f} seconds")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    REQUEST_COUNT.labels(endpoint='/health', status='success').inc()

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint

    Expected input format:
    {
        "features": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            ...
        }
    }
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Get input data
        data = request.get_json()

        if not data or 'features' not in data:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Invalid input format'}), 400

        # For demo: if model is None, return dummy prediction
        if model is None:
            prediction = np.random.randint(0, 2)
            probability = np.random.random()
        else:
            # Preprocess input
            # In real scenario, you would apply the same preprocessing as training
            features = data['features']

            # For demo: create a simple feature vector
            # In production, this should match the preprocessing pipeline
            feature_vector = np.array([[0] * 19])  # Placeholder

            # Make prediction
            prediction = model.predict(feature_vector)[0]
            probability = model.predict_proba(feature_vector)[0][1]

        # Record prediction
        prediction_class = 'churn' if prediction == 1 else 'no_churn'
        PREDICTION_COUNTER.labels(prediction_class=prediction_class).inc()

        # Calculate latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
        ACTIVE_REQUESTS.dec()

        # Return response
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
            'probability': float(probability),
            'latency': latency
        }), 200

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint

    Expected input format:
    {
        "instances": [
            {"gender": "Female", "SeniorCitizen": 0, ...},
            {"gender": "Male", "SeniorCitizen": 1, ...},
            ...
        ]
    }
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Get input data
        data = request.get_json()

        if not data or 'instances' not in data:
            REQUEST_COUNT.labels(endpoint='/predict_batch', status='error').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Invalid input format'}), 400

        instances = data['instances']
        num_instances = len(instances)

        # For demo: return dummy predictions
        predictions = []
        for _ in range(num_instances):
            if model is None:
                prediction = np.random.randint(0, 2)
                probability = np.random.random()
            else:
                # In production, process and predict for each instance
                prediction = np.random.randint(0, 2)
                probability = np.random.random()

            prediction_class = 'churn' if prediction == 1 else 'no_churn'
            PREDICTION_COUNTER.labels(prediction_class=prediction_class).inc()

            predictions.append({
                'prediction': int(prediction),
                'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
                'probability': float(probability)
            })

        # Calculate latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        REQUEST_COUNT.labels(endpoint='/predict_batch', status='success').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({
            'predictions': predictions,
            'count': num_instances,
            'latency': latency
        }), 200

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/predict_batch', status='error').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({
            'error': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/info', methods=['GET'])
def info():
    """
    Model information endpoint
    """
    REQUEST_COUNT.labels(endpoint='/info', status='success').inc()

    return jsonify({
        'model_name': 'Telco Customer Churn Classifier',
        'model_type': 'Random Forest / Logistic Regression',
        'version': '1.0',
        'author': 'Nama Siswa',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Single prediction',
            '/predict_batch': 'Batch predictions',
            '/metrics': 'Prometheus metrics',
            '/info': 'Model information'
        }
    }), 200


if __name__ == '__main__':
    print("="*60)
    print("Starting Model Serving API")
    print("="*60)

    # Load model and artifacts
    load_model_and_artifacts()

    # Run Flask app
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  - Health check: GET http://localhost:5001/health")
    print("  - Prediction: POST http://localhost:5001/predict")
    print("  - Batch prediction: POST http://localhost:5001/predict_batch")
    print("  - Metrics: GET http://localhost:5001/metrics")
    print("  - Info: GET http://localhost:5001/info")
    print("\n" + "="*60)

    app.run(host='0.0.0.0', port=5001, debug=False)
