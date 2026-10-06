# ⚡ Scikit-Learn ML Model Web API (FastAPI Microservice)

> A production-grade **FastAPI microservice** that wraps a trained Scikit-Learn `RandomForestRegressor` and `StandardScaler` pipeline to serve high-performance JSON predictions over HTTP.

---

## 🏗️ Architecture & Features

- **Framework**: **FastAPI** with automatic OpenAPI (Swagger UI) documentation generation.
- **Data Validation**: Strict type-checking and schema validation powered by **Pydantic v2**.
- **Model Pipeline**: Serialized `RandomForestRegressor` model + `StandardScaler` preprocessor managed via `joblib`.
- **Endpoints**:
  - `GET /` — API metadata and documentation links.
  - `GET /health` — Health check, model status, and evaluation metrics (RMSE, $R^2$).
  - `POST /predict` — Single instance JSON housing price prediction.
  - `POST /batch-predict` — High-throughput batch JSON housing price predictions.
  - `GET /model/info` — Feature importances, hyperparameters, and model specs.

---

## 📩 Sample Request and Response

### 1. Single Instance Prediction (`POST /predict`)

#### **Sample HTTP Request Payload (JSON)**:
```json
{
  "square_feet": 2400.0,
  "bedrooms": 4,
  "bathrooms": 3,
  "age_years": 5.0,
  "distance_to_city_km": 6.5,
  "crime_rate_index": 1.1
}
```

#### **Sample cURL Command**:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "square_feet": 2400.0,
  "bedrooms": 4,
  "bathrooms": 3,
  "age_years": 5.0,
  "distance_to_city_km": 6.5,
  "crime_rate_index": 1.1
}'
```

#### **Sample HTTP JSON Response**:
```json
{
  "predicted_price": 532480.75,
  "currency": "USD",
  "model_version": "1.0.0",
  "input_summary": {
    "square_feet": 2400.0,
    "bedrooms": 4,
    "bathrooms": 3,
    "age_years": 5.0,
    "distance_to_city_km": 6.5,
    "crime_rate_index": 1.1
  }
}
```

---

### 2. Batch Predictions (`POST /batch-predict`)

#### **Sample HTTP Request Payload (JSON)**:
```json
{
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
```

#### **Sample HTTP JSON Response**:
```json
{
  "predictions": [
    {
      "predicted_price": 245120.40,
      "currency": "USD",
      "model_version": "1.0.0",
      "input_summary": {
        "square_feet": 1200.0,
        "bedrooms": 2,
        "bathrooms": 1,
        "age_years": 20.0,
        "distance_to_city_km": 15.0,
        "crime_rate_index": 3.5
      }
    },
    {
      "predicted_price": 782910.10,
      "currency": "USD",
      "model_version": "1.0.0",
      "input_summary": {
        "square_feet": 3500.0,
        "bedrooms": 5,
        "bathrooms": 4,
        "age_years": 2.0,
        "distance_to_city_km": 4.0,
        "crime_rate_index": 0.5
      }
    }
  ],
  "total_processed": 2
}
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Serialize the Scikit-Learn Model
Run the model trainer to generate `model.joblib`:
```bash
python train_model.py
```

### 3. Run FastAPI Microservice with Uvicorn
Start the production server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000/docs` in your web browser to explore interactive Swagger UI documentation!

### 4. Run Automated Test Suite
```bash
python test_api.py
```

---

## 📂 Project Structure

```
project_04_ml_model_web_api/
├── README.md           # API Documentation & sample request/response
├── requirements.txt    # Dependencies
├── train_model.py      # Scikit-Learn model training & joblib exporter
├── app.py              # FastAPI microservice endpoints & validation
├── test_api.py         # Automated TestClient runner
└── model.joblib        # Serialized model & scaler artifact
```
