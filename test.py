import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("student.csv")

# Features and target
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale only for Linear Regression (not needed for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42, n_estimators=100)

# Train models
lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)  # NOTE: no scaling for RF

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

# Evaluation function
def evaluate_model(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{name} Performance:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return r2, mae, rmse

# Evaluate both models
lr_metrics = evaluate_model("Linear Regression", y_test, y_pred_lr)
rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf)

# Compare results
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_metrics[0], rf_metrics[0]],
    "MAE": [lr_metrics[1], rf_metrics[1]],
    "RMSE": [lr_metrics[2], rf_metrics[2]]
})

print("\nModel Comparison:\n")
print(results)

# Select best model based on R2 score
best_model = rf if rf_metrics[0] > lr_metrics[0] else lr

# Save best model + scaler (only if LR wins)
pickle.dump(best_model, open("best_model.pkl", "wb"))

if isinstance(best_model, LinearRegression):
    pickle.dump(scaler, open("scaler.pkl", "wb"))