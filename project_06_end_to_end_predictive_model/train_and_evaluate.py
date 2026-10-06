"""
End-to-End Predictive Model: Housing Price Predictor.
Demonstrates feature scaling, Decision Trees, Random Forests,
hyperparameter tuning, and comprehensive evaluation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def generate_housing_dataset(n_samples=1500, seed=42):
    np.random.seed(seed)
    
    sqft = np.random.normal(1900, 520, n_samples).clip(650, 5200)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age_years = np.random.uniform(0, 50, n_samples)
    distance_to_city = np.random.uniform(1, 35, n_samples)
    crime_rate = np.random.uniform(0.1, 9.0, n_samples)
    school_rating = np.random.uniform(3.0, 10.0, n_samples)
    
    # Target formula with non-linear tree interaction features
    base = 45000
    price = (
        base
        + sqft * 185.0
        + bedrooms * 24000.0
        + bathrooms * 36000.0
        - age_years * 1150.0
        - distance_to_city * 3400.0
        - crime_rate * 4800.0
        + school_rating * 12000.0
        + np.random.normal(0, 16000, n_samples)
    )
    
    df = pd.DataFrame({
        'square_feet': np.round(sqft, 1),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': np.round(age_years, 1),
        'distance_to_city': np.round(distance_to_city, 1),
        'crime_rate': np.round(crime_rate, 2),
        'school_rating': np.round(school_rating, 1),
        'price': np.round(price, 2)
    })
    return df

def run_end_to_end_pipeline():
    print("=" * 70)
    print("  END-TO-END PREDICTIVE MODEL PIPELINE (HOUSING PRICE PREDICTOR)")
    print("=" * 70)
    
    # Step 1: Data Generation & Loading
    df = generate_housing_dataset(n_samples=1500, seed=42)
    print(f"\n[1] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    
    # Step 2: Feature Selection & Split
    feature_cols = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city', 'crime_rate', 'school_rating']
    X = df[feature_cols]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[2] Train/Test Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    # Step 3: Feature Scaling Comparison
    print("\n[3] Applying Feature Scaling...")
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train)
    X_test_std = scaler_std.transform(X_test)
    
    scaler_minmax = MinMaxScaler()
    X_train_mm = scaler_minmax.fit_transform(X_train)
    X_test_mm = scaler_minmax.transform(X_test)
    
    # Step 4: Model Training & Evaluation Function
    def evaluate_model(model, name, X_tr, X_te):
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return {
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'model_obj': model,
            'predictions': preds
        }

    results = []
    
    # Decision Tree (Unscaled vs Scaled)
    dt_raw = evaluate_model(DecisionTreeRegressor(max_depth=8, random_state=42), "Decision Tree (Unscaled)", X_train, X_test)
    dt_std = evaluate_model(DecisionTreeRegressor(max_depth=8, random_state=42), "Decision Tree (StandardScaler)", X_train_std, X_test_std)
    dt_mm = evaluate_model(DecisionTreeRegressor(max_depth=8, random_state=42), "Decision Tree (MinMaxScaler)", X_train_mm, X_test_mm)
    
    # Random Forest (Unscaled vs Scaled)
    rf_raw = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42), "Random Forest (Unscaled)", X_train, X_test)
    rf_std = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42), "Random Forest (StandardScaler)", X_train_std, X_test_std)
    rf_mm = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42), "Random Forest (MinMaxScaler)", X_train_mm, X_test_mm)
    
    results.extend([dt_raw, dt_std, dt_mm, rf_raw, rf_std, rf_mm])
    
    # Step 5: Hyperparameter Tuning via GridSearchCV
    print("\n[4] Running GridSearchCV Hyperparameter Tuning on Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 12, 16],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train_std, y_train)
    best_rf = grid_search.best_estimator_
    
    best_rf_res = evaluate_model(best_rf, "Random Forest (GridSearch Tuned + Scaled)", X_train_std, X_test_std)
    results.append(best_rf_res)
    
    # Step 6: Benchmark Results Table
    res_df = pd.DataFrame(results)[['Model', 'RMSE', 'MAE', 'R2']]
    print("\n[5] MODEL BENCHMARK COMPARISON TABLE:")
    print("-" * 75)
    print(res_df.to_string(index=False))
    print("-" * 75)
    
    # Step 7: Plot & Save Evaluation Charts
    print("\n[6] Generating Performance Evaluation Plots...")
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Feature Importances of Best Model
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    
    sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette="viridis", legend=False, ax=ax)
    ax.set_title("Feature Importances (Tuned Random Forest)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png", dpi=200)
    plt.close()
    
    # Plot 2: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    best_preds = best_rf_res['predictions']
    ax.scatter(y_test, best_preds, alpha=0.6, color="#2563eb", edgecolors='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
    ax.set_title("Actual vs Predicted House Prices ($ USD)")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=200)
    plt.close()
    
    print("Evaluation charts saved to 'plots/feature_importances.png' and 'plots/actual_vs_predicted.png'.")
    
    # Step 8: Save Model Artifact
    model_artifact = {
        'model': best_rf,
        'scaler': scaler_std,
        'feature_names': feature_cols,
        'metrics': {
            'rmse': best_rf_res['RMSE'],
            'mae': best_rf_res['MAE'],
            'r2': best_rf_res['R2']
        },
        'best_params': grid_search.best_params_
    }
    joblib.dump(model_artifact, "housing_rf_model.joblib")
    print("\n[7] Trained Model Pipeline serialized to 'housing_rf_model.joblib'.")
    print("\nPipeline Execution Finished Successfully!")

if __name__ == "__main__":
    run_end_to_end_pipeline()
