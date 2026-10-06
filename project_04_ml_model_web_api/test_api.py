"""
Automated Test Client Suite for FastAPI ML Microservice.
Uses FastAPI TestClient to send sample JSON HTTP requests to endpoints.
"""

import json
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    print("Testing GET /health...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print("Response Status Code: 200 OK")
    print(f"Health Response Data:\n{json.dumps(data, indent=2)}\n")
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_single_endpoint():
    print("Testing POST /predict (Single Prediction)...")
    sample_payload = {
        "square_feet": 2400.0,
        "bedrooms": 4,
        "bathrooms": 3,
        "age_years": 5.0,
        "distance_to_city_km": 6.5,
        "crime_rate_index": 1.1
    }
    print(f"Request Payload:\n{json.dumps(sample_payload, indent=2)}")
    
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    print(f"Response Payload:\n{json.dumps(data, indent=2)}\n")
    assert "predicted_price" in data
    assert data["currency"] == "USD"

def test_batch_predict_endpoint():
    print("Testing POST /batch-predict (Batch Prediction)...")
    sample_batch_payload = {
        "houses": [
            {
                "square_feet": 1200.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "age_years": 20.0,
                "distance_to_city_km": 15.0,
                "crime_rate_index": 3.5
            },
            {
                "square_feet": 3500.0,
                "bedrooms": 5,
                "bathrooms": 4,
                "age_years": 2.0,
                "distance_to_city_km": 4.0,
                "crime_rate_index": 0.5
            }
        ]
    }
    response = client.post("/batch-predict", json=sample_batch_payload)
    assert response.status_code == 200
    data = response.json()
    print(f"Batch Response Data:\n{json.dumps(data, indent=2)}\n")
    assert data["total_processed"] == 2

def main():
    print("=" * 60)
    print("   FASTAPI ML WEB API ENDPOINT TEST SUITE")
    print("=" * 60)
    test_health_endpoint()
    test_predict_single_endpoint()
    test_batch_predict_endpoint()
    print("🎉 All FastAPI endpoints tested successfully!")

if __name__ == "__main__":
    main()
