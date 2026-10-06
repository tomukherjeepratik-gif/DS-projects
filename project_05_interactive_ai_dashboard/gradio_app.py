"""
Secondary Interactive Web Dashboard using Gradio.
Provides slider inputs and returns real-time house price predictions.
"""

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_default_model():
    np.random.seed(42)
    n = 1000
    sqft = np.random.normal(2000, 500, n).clip(700, 5000)
    beds = np.random.randint(1, 6, n)
    baths = np.random.randint(1, 4, n)
    age = np.random.uniform(0, 45, n)
    dist = np.random.uniform(1, 30, n)
    crime = np.random.uniform(0.1, 8.0, n)
    
    price = 50000 + sqft*190 + beds*23000 + baths*36000 - age*1100 - dist*3300 - crime*4800
    
    X = pd.DataFrame({
        'square_feet': sqft, 'bedrooms': beds, 'bathrooms': baths,
        'age_years': age, 'distance_to_city_km': dist, 'crime_rate_index': crime
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_scaled, price)
    return model, scaler

model, scaler = train_default_model()

def predict_price(sqft, beds, baths, age, dist, crime):
    input_df = pd.DataFrame([[sqft, beds, baths, age, dist, crime]], columns=[
        'square_feet', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city_km', 'crime_rate_index'
    ])
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    return f"${pred:,.2f} USD"

# Build Gradio Interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(700, 5000, value=2200, step=50, label="Square Feet"),
        gr.Slider(1, 6, value=3, step=1, label="Bedrooms"),
        gr.Slider(1, 5, value=2, step=1, label="Bathrooms"),
        gr.Slider(0, 50, value=10, step=1, label="Property Age (Years)"),
        gr.Slider(1, 35, value=8, step=1, label="Distance to City Center (km)"),
        gr.Slider(0.1, 10.0, value=1.5, step=0.1, label="Crime Index"),
    ],
    outputs=gr.Textbox(label="Estimated Property Price"),
    title="⚡ AI Housing Price Predictor (Gradio UI)",
    description="Adjust the sliders to get instant house price predictions powered by Random Forest ML."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
