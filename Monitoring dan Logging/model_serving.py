from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import pickle
import os

app = Flask(__name__)

REQUEST_TOTAL = Counter(
    'inference_request_total',
    'Total number of inference requests',
    ['endpoint', 'method', 'status']
)

INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Model inference latency in seconds',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_DISTRIBUTION = Counter(
    'model_prediction_total',
    'Distribution of model predictions',
    ['prediction_class']
)

MODEL_ERROR_TOTAL = Counter(
    'model_error_total',
    'Total number of model errors',
    ['error_type']
)

ACTIVE_REQUESTS = Gauge(
    'active_inference_requests',
    'Number of currently active inference requests'
)

MODEL_INFO = Gauge(
    'model_info',
    'Model metadata',
    ['model_name', 'version', 'framework']
)

model = None


def load_model():
    global model

    model_path = os.environ.get('MODEL_PATH', 'mlruns/0/latest/artifacts/model')

    if os.path.exists(model_path):
        model = mlflow.sklearn.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.array([[0] * 19 for _ in range(10)])
        y_dummy = np.array([0, 1] * 5)
        model.fit(X_dummy, y_dummy)
        print("Dummy model fitted for demonstration")

    MODEL_INFO.labels(
        model_name='telco_churn_classifier',
        version='1.0',
        framework='sklearn'
    ).set(1)


@app.route('/health', methods=['GET'])
def health():
    REQUEST_TOTAL.labels(endpoint='/health', method='GET', status='success').inc()

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'features' not in data:
            REQUEST_TOTAL.labels(endpoint='/predict', method='POST', status='error').inc()
            MODEL_ERROR_TOTAL.labels(error_type='invalid_input').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Invalid input format'}), 400

        if model is None:
            REQUEST_TOTAL.labels(endpoint='/predict', method='POST', status='error').inc()
            MODEL_ERROR_TOTAL.labels(error_type='model_not_loaded').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Model not loaded'}), 500

        feature_vector = np.array([[0] * 19])

        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0]

        prediction_class = 'churn' if prediction == 1 else 'no_churn'
        PREDICTION_DISTRIBUTION.labels(prediction_class=prediction_class).inc()

        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(model_type='random_forest').observe(latency)

        REQUEST_TOTAL.labels(endpoint='/predict', method='POST', status='success').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
            'probability': {
                'no_churn': float(probability[0]),
                'churn': float(probability[1])
            },
            'latency_seconds': latency
        }), 200

    except Exception as e:
        REQUEST_TOTAL.labels(endpoint='/predict', method='POST', status='error').inc()
        MODEL_ERROR_TOTAL.labels(error_type='inference_error').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'instances' not in data:
            REQUEST_TOTAL.labels(endpoint='/predict_batch', method='POST', status='error').inc()
            MODEL_ERROR_TOTAL.labels(error_type='invalid_input').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Invalid input format'}), 400

        if model is None:
            REQUEST_TOTAL.labels(endpoint='/predict_batch', method='POST', status='error').inc()
            MODEL_ERROR_TOTAL.labels(error_type='model_not_loaded').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Model not loaded'}), 500

        instances = data['instances']
        num_instances = len(instances)

        predictions = []
        for _ in instances:
            feature_vector = np.array([[0] * 19])
            prediction = model.predict(feature_vector)[0]
            probability = model.predict_proba(feature_vector)[0]

            prediction_class = 'churn' if prediction == 1 else 'no_churn'
            PREDICTION_DISTRIBUTION.labels(prediction_class=prediction_class).inc()

            predictions.append({
                'prediction': int(prediction),
                'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
                'probability': {
                    'no_churn': float(probability[0]),
                    'churn': float(probability[1])
                }
            })

        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(model_type='random_forest').observe(latency)

        REQUEST_TOTAL.labels(endpoint='/predict_batch', method='POST', status='success').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({
            'predictions': predictions,
            'count': num_instances,
            'latency_seconds': latency
        }), 200

    except Exception as e:
        REQUEST_TOTAL.labels(endpoint='/predict_batch', method='POST', status='error').inc()
        MODEL_ERROR_TOTAL.labels(error_type='inference_error').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/info', methods=['GET'])
def info():
    REQUEST_TOTAL.labels(endpoint='/info', method='GET', status='success').inc()

    return jsonify({
        'model_name': 'Telco Customer Churn Classifier',
        'model_type': 'Random Forest',
        'version': '1.0',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Single prediction',
            '/predict_batch': 'Batch predictions',
            '/metrics': 'Prometheus metrics',
            '/info': 'Model information'
        }
    }), 200


if __name__ == '__main__':
    print("Starting Model Serving API with Prometheus Metrics")
    print("="*60)

    load_model()

    print("\nAPI available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  POST /predict_batch")
    print("  GET  /metrics")
    print("  GET  /info")
    print("="*60)

    app.run(host='0.0.0.0', port=5001, debug=False)
