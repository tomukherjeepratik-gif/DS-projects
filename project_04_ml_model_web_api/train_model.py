"""
Model Training Script for FastAPI Web API Service.
Trains a Scikit-Learn RandomForestRegressor on a Real Estate / Housing dataset
and serializes the trained model artifact and feature metadata to joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generate_housing_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    square_feet = np.random.normal(1800, 500, n_samples).clip(600, 5000)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age_years = np.random.uniform(0, 50, n_samples)
    distance_to_city_km = np.random.uniform(1, 30, n_samples)
    crime_rate_index = np.random.uniform(0.1, 10.0, n_samples)
    
    # Target Formula: Base price + feature contributions + random noise
    base_price = 50000
    price = (
        base_price
        + square_feet * 180.0
        + bedrooms * 25000.0
        + bathrooms * 35000.0
        - age_years * 1200.0
        - distance_to_city_km * 3200.0
        - crime_rate_index * 4500.0
        + np.random.normal(0, 15000, n_samples)
    )
    
    df = pd.DataFrame({
        'square_feet': np.round(square_feet, 1),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': np.round(age_years, 1),
        'distance_to_city_km': np.round(distance_to_city_km, 1),
        'crime_rate_index': np.round(crime_rate_index, 2),
        'price': np.round(price, 2)
    })
    return df

def train_and_save_model(model_path="model.joblib"):
    print("Generating training dataset...")
    df = generate_housing_data(n_samples=1500, random_state=42)
    
    feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city_km', 'crime_rate_index']
    X = df[feature_names]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting StandardScaler and RandomForestRegressor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"  - RMSE: ${rmse:,.2f}")
    print(f"  - R² Score: {r2:.4f}")
    
    # Save combined artifact
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': {
            'rmse': float(rmse),
            'r2': float(r2)
        },
        'version': '1.0.0'
    }
    
    joblib.dump(artifact, model_path)
    print(f"Model artifact successfully serialized to '{model_path}'!")
    return artifact

if __name__ == "__main__":
    train_and_save_model()
