import requests
import json
import time

def test_api_and_generate_metrics():
    base_url = "http://localhost:5001"

    print("="*60)
    print("Testing Model Serving API and Generating Metrics")
    print("="*60)

    print("\n1. Health Check")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    print("\n2. Generating Prediction Requests")
    for i in range(30):
        payload = {
            "features": {
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": i % 2,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12 + i,
                "PhoneService": "Yes",
                "MonthlyCharges": 65.0 + i,
                "TotalCharges": 780.0 + i * 10
            }
        }

        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Request {i+1}: {result['prediction_label']} (latency: {result['latency_seconds']:.4f}s)")
        else:
            print(f"Request {i+1}: ERROR {response.status_code}")

        time.sleep(0.2)

    print("\n3. Testing Batch Prediction")
    batch_payload = {
        "instances": [
            {"gender": "Female", "SeniorCitizen": 0},
            {"gender": "Male", "SeniorCitizen": 1},
            {"gender": "Female", "SeniorCitizen": 0}
        ]
    }

    response = requests.post(
        f"{base_url}/predict_batch",
        json=batch_payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Batch prediction status: {response.status_code}")
    print(f"Number of predictions: {response.json()['count']}")

    print("\n4. Fetching Prometheus Metrics")
    response = requests.get(f"{base_url}/metrics")

    metrics_lines = response.text.split('\n')

    print("\nKey Metrics:")
    print("-" * 60)

    for line in metrics_lines:
        if line.startswith('inference_request_total{'):
            print(line)
        elif line.startswith('inference_latency_seconds_bucket{'):
            print(line)
        elif line.startswith('model_prediction_total{'):
            print(line)
        elif line.startswith('model_error_total{'):
            print(line)
        elif line.startswith('active_inference_requests'):
            print(line)

    print("\n" + "="*60)
    print("Metrics Generation Complete")
    print("="*60)
    print("\nNext Steps:")
    print("1. Check Prometheus: http://localhost:9090")
    print("2. View alerts: http://localhost:9090/alerts")
    print("3. Check Alertmanager: http://localhost:9093")
    print("4. View Grafana: http://localhost:3000")

if __name__ == "__main__":
    try:
        test_api_and_generate_metrics()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API")
        print("Start the API first: python model_serving.py")
    except Exception as e:
        print(f"\nERROR: {e}")
