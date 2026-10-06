"""
FastAPI Microservice for Serving Scikit-Learn Housing Price Predictions.
Provides JSON prediction endpoints with strict Pydantic schema validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Initialize FastAPI Application
app = FastAPI(
    title="Housing Price Predictive ML Microservice",
    description="Production-grade FastAPI service serving Scikit-Learn Random Forest predictions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model state artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
artifact: Dict[str, Any] = {}

def get_model_artifact():
    global artifact
    if not artifact:
        if not os.path.exists(MODEL_PATH):
            # Train model if not present
            from train_model import train_and_save_model
            artifact = train_and_save_model(MODEL_PATH)
        else:
            artifact = joblib.load(MODEL_PATH)
    return artifact

@app.on_event("startup")
def startup_event():
    get_model_artifact()
    print("FastAPI Microservice loaded model artifact successfully!")

# Pydantic Request & Response Schemas
class HouseFeatures(BaseModel):
    square_feet: float = Field(..., gt=100, lt=20000, example=2100.0, description="Total area in square feet")
    bedrooms: int = Field(..., ge=1, le=10, example=3, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=10, example=2, description="Number of bathrooms")
    age_years: float = Field(..., ge=0, le=150, example=8.5, description="Age of property in years")
    distance_to_city_km: float = Field(..., ge=0.1, le=200.0, example=5.2, description="Distance to city center in kilometers")
    crime_rate_index: float = Field(..., ge=0.0, le=100.0, example=1.2, description="Neighborhood crime rate score index")

class SinglePredictionResponse(BaseModel):
    predicted_price: float = Field(..., example=425800.50)
    currency: str = Field("USD", example="USD")
    model_version: str = Field("1.0.0", example="1.0.0")
    input_summary: Dict[str, float]

class BatchPredictRequest(BaseModel):
    houses: List[HouseFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    metrics: Dict[str, float]

# Endpoints
@app.get("/", summary="Root Endpoint")
def root():
    return {
        "service": "Housing Price Predictive ML Web API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check & Model Metadata")
def health_check():
    art = get_model_artifact()
    return {
        "status": "healthy",
        "model_loaded": art is not None,
        "model_version": art.get("version", "1.0.0"),
        "metrics": art.get("metrics", {})
    }

@app.post("/predict", response_model=SinglePredictionResponse, summary="Single House Price Prediction")
def predict_single(features: HouseFeatures):
    art = get_model_artifact()
    model = art["model"]
    scaler = art["scaler"]
    feature_names = art["feature_names"]
    
    # Format input array
    input_data = pd.DataFrame([[
        features.square_feet,
        features.bedrooms,
        features.bathrooms,
        features.age_years,
        features.distance_to_city_km,
        features.crime_rate_index
    ]], columns=feature_names)
    
    # Scale features & predict
    scaled_data = scaler.transform(input_data)
    pred_val = float(model.predict(scaled_data)[0])
    
    return {
        "predicted_price": round(pred_val, 2),
        "currency": "USD",
        "model_version": art.get("version", "1.0.0"),
        "input_summary": features.dict()
    }

@app.post("/batch-predict", response_model=BatchPredictionResponse, summary="Batch House Price Predictions")
def predict_batch(request: BatchPredictRequest):
    art = get_model_artifact()
    model = art["model"]
    scaler = art["scaler"]
    feature_names = art["feature_names"]
    
    rows = []
    for f in request.houses:
        rows.append([
            f.square_feet,
            f.bedrooms,
            f.bathrooms,
            f.age_years,
            f.distance_to_city_km,
            f.crime_rate_index
        ])
    
    input_df = pd.DataFrame(rows, columns=feature_names)
    scaled_data = scaler.transform(input_df)
    predictions = model.predict(scaled_data)
    
    results = []
    for idx, f in enumerate(request.houses):
        results.append({
            "predicted_price": round(float(predictions[idx]), 2),
            "currency": "USD",
            "model_version": art.get("version", "1.0.0"),
            "input_summary": f.dict()
        })
        
    return {
        "predictions": results,
        "total_processed": len(results)
    }

@app.get("/model/info", summary="Model Specifications & Features")
def model_info():
    art = get_model_artifact()
    model = art["model"]
    importances = dict(zip(art["feature_names"], [round(float(imp), 4) for imp in model.feature_importances_]))
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "feature_names": art["feature_names"],
        "feature_importances": importances,
        "evaluation_metrics": art["metrics"]
    }
