"""
Test script for Model Serving API
Author: Putu Linda Suryantini
Purpose: Testing prediction endpoints
"""

import requests
import json
import time


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)

    response = requests.get("http://localhost:5001/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("Health check: PASSED")


def test_predict():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)

    payload = {
        "features": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0
        }
    }

    response = requests.post(
        "http://localhost:5001/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    print("Single prediction: PASSED")


def test_predict_batch():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict_batch endpoint")
    print("="*60)

    payload = {
        "instances": [
            {"gender": "Female", "SeniorCitizen": 0},
            {"gender": "Male", "SeniorCitizen": 1},
            {"gender": "Female", "SeniorCitizen": 0}
        ]
    }

    response = requests.post(
        "http://localhost:5001/predict_batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 3
    print("Batch prediction: PASSED")


def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*60)
    print("Testing /metrics endpoint")
    print("="*60)

    response = requests.get("http://localhost:5001/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Metrics sample (first 500 chars):")
    print(response.text[:500])

    assert response.status_code == 200
    assert "model_prediction_requests_total" in response.text
    print("Metrics endpoint: PASSED")


def test_info():
    """Test info endpoint"""
    print("\n" + "="*60)
    print("Testing /info endpoint")
    print("="*60)

    response = requests.get("http://localhost:5001/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("Info endpoint: PASSED")


def load_test():
    """Generate load for monitoring"""
    print("\n" + "="*60)
    print("Running Load Test (20 requests)")
    print("="*60)

    payload = {
        "features": {
            "gender": "Female",
            "SeniorCitizen": 0
        }
    }

    for i in range(20):
        response = requests.post(
            "http://localhost:5001/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Request {i+1}/20 - Status: {response.status_code} - "
              f"Prediction: {response.json().get('prediction_label', 'N/A')}")
        time.sleep(0.5)

    print("Load test completed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL SERVING API TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running at http://localhost:5001")
    print("Start with: python monitoring/model_serving.py")

    try:
        # Run all tests
        test_health()
        test_predict()
        test_predict_batch()
        test_metrics()
        test_info()
        load_test()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now:")
        print("1. View metrics: http://localhost:5001/metrics")
        print("2. Check Prometheus: http://localhost:9090")
        print("3. View Grafana dashboard: http://localhost:3000")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API")
        print("Please start the API first:")
        print("  python monitoring/model_serving.py")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")

    except Exception as e:
        print(f"\nERROR: {e}")
