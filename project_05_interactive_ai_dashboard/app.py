"""
Interactive AI ML Dashboard powered by Streamlit.
Provides real-time interactive model inference, what-if sensitivity analysis,
and dynamic model performance visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit Page Config
st.set_page_config(
    page_title="AI Interactive Predictive Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_housing_data(n_samples=1200, seed=42):
    np.random.seed(seed)
    sqft = np.random.normal(2000, 550, n_samples).clip(700, 5500)
    beds = np.random.randint(1, 6, n_samples)
    baths = np.random.randint(1, 5, n_samples)
    age = np.random.uniform(0, 45, n_samples)
    dist = np.random.uniform(1, 35, n_samples)
    crime = np.random.uniform(0.1, 8.0, n_samples)
    
    price = (
        60000
        + sqft * 195.0
        + beds * 22000.0
        + baths * 38000.0
        - age * 1100.0
        - dist * 3500.0
        - crime * 5000.0
        + np.random.normal(0, 18000, n_samples)
    )
    
    return pd.DataFrame({
        'square_feet': np.round(sqft, 1),
        'bedrooms': beds,
        'bathrooms': baths,
        'age_years': np.round(age, 1),
        'distance_to_city_km': np.round(dist, 1),
        'crime_rate_index': np.round(crime, 2),
        'price': np.round(price, 2)
    })

@st.cache_resource
def train_model(algorithm_name, scaling_type, n_estimators, max_depth):
    df = generate_housing_data()
    feature_cols = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city_km', 'crime_rate_index']
    X = df[feature_cols]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if scaling_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None
        
    if scaler:
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = X_train
        X_test_proc = X_test
        
    if algorithm_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif algorithm_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, scaler, feature_cols, df, X_test, y_test, y_pred, metrics

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=60)
st.sidebar.title("Dashboard Controls")

algorithm = st.sidebar.selectbox("Select ML Model", ["Random Forest", "Gradient Boosting", "Decision Tree"])
scaling = st.sidebar.selectbox("Feature Scaling", ["StandardScaler", "MinMaxScaler", "None"])

st.sidebar.subheader("Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 80, step=10) if algorithm != "Decision Tree" else 100
max_depth = st.sidebar.slider("Max Tree Depth", 2, 25, 10)

# Train model based on user selection
model, scaler, feature_cols, df_data, X_test, y_test, y_pred, metrics = train_model(
    algorithm, scaling, n_estimators, max_depth
)

# Header
st.markdown('<div class="main-title">⚡ Interactive Real-Time AI Predictive Dashboard</div>', unsafe_allow_html=True)
st.caption(f"Active Model: **{algorithm}** | Feature Scaler: **{scaling}** | Test Dataset $R^2$: **{metrics['r2']:.4f}**")

# Top Metric Banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model R² Accuracy", value=f"{metrics['r2']:.2%}")
with col2:
    st.metric(label="Root Mean Sq Error (RMSE)", value=f"${metrics['rmse']:,.0f}")
with col3:
    st.metric(label="Mean Absolute Error (MAE)", value=f"${metrics['mae']:,.0f}")
with col4:
    st.metric(label="Dataset Total Samples", value=f"{len(df_data):,}")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏡 Real-Time Predictor", 
    "📊 Model Analytics & Importance", 
    "🔍 What-If Sensitivity Explorer", 
    "📈 Dataset Explorer & Correlation"
])

# TAB 1: REAL-TIME PREDICTOR
with tab1:
    st.subheader("🏡 Interactive Property Input & Valuation")
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.write("Adjust property feature sliders to view instant valuation:")
        in_sqft = st.slider("Square Feet Area", 700, 5500, 2200, step=50)
        in_beds = st.slider("Bedrooms", 1, 6, 3)
        in_baths = st.slider("Bathrooms", 1, 5, 2)
        in_age = st.slider("Property Age (Years)", 0.0, 45.0, 8.0, step=0.5)
        in_dist = st.slider("Distance to City Center (km)", 1.0, 35.0, 7.5, step=0.5)
        in_crime = st.slider("Neighborhood Crime Index", 0.1, 8.0, 1.5, step=0.1)
        
    with col_result:
        raw_input = pd.DataFrame([[in_sqft, in_beds, in_baths, in_age, in_dist, in_crime]], columns=feature_cols)
        processed_input = scaler.transform(raw_input) if scaler else raw_input
        predicted_val = float(model.predict(processed_input)[0])
        
        lower_bound = predicted_val - metrics['mae']
        upper_bound = predicted_val + metrics['mae']
        
        st.markdown(f"""
        <div style="background-color: #0f172a; border-left: 5px solid #38bdf8; padding: 20px; border-radius: 8px;">
            <h4 style="color: #94a3b8; margin: 0;">ESTIMATED MARKET VALUE</h4>
            <h1 style="color: #38bdf8; font-size: 3rem; margin: 10px 0;">${predicted_val:,.2f} USD</h1>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Estimated Confidence Interval (± MAE):<br>
                <strong>${lower_bound:,.2f}</strong> — <strong>${upper_bound:,.2f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple Visual Indicator Bar
        fig, ax = plt.subplots(figsize=(6, 2.2))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.barh(["Estimated Price"], [predicted_val], color="#38bdf8", edgecolor="white", height=0.4)
        ax.set_xlim(0, 1200000)
        ax.set_xlabel("Valuation Range ($ USD)", color="#cbd5e1")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig)

# TAB 2: MODEL ANALYTICS & IMPORTANCE
with tab2:
    st.subheader("📊 Model Feature Importance & Residual Distribution")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Feature Importance Analysis**")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            fig_imp.patch.set_facecolor("#0f172a")
            ax_imp.set_facecolor("#0f172a")
            sns.barplot(x=importances, y=feature_cols, ax=ax_imp, palette="crest")
            ax_imp.set_title("Relative Feature Importance Scores", color="white")
            ax_imp.tick_params(colors="white")
            for spine in ax_imp.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig_imp)
        else:
            st.info("Feature importances not available for this model configuration.")
            
    with col_b:
        st.write("**Actual vs Predicted Prices Scatter Plot**")
        fig_scat, ax_scat = plt.subplots(figsize=(6, 4))
        fig_scat.patch.set_facecolor("#0f172a")
        ax_scat.set_facecolor("#0f172a")
        ax_scat.scatter(y_test, y_pred, alpha=0.5, color="#818cf8")
        ax_scat.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_scat.set_xlabel("Actual Price ($)", color="white")
        ax_scat.set_ylabel("Predicted Price ($)", color="white")
        ax_scat.set_title("Actual vs Predicted", color="white")
        ax_scat.tick_params(colors="white")
        for spine in ax_scat.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig_scat)

# TAB 3: WHAT-IF SENSITIVITY EXPLORER
with tab3:
    st.subheader("🔍 Sensitivity Analysis (Varying Square Footage)")
    st.write("Observe how changing property size (sq ft) influences house valuation holding other parameters constant:")
    
    sqft_range = np.linspace(800, 5000, 50)
    sim_inputs = []
    for sq in sqft_range:
        sim_inputs.append([sq, 3, 2, 10.0, 8.0, 1.5])
    sim_df = pd.DataFrame(sim_inputs, columns=feature_cols)
    sim_proc = scaler.transform(sim_df) if scaler else sim_df
    sim_preds = model.predict(sim_proc)
    
    fig_sens, ax_sens = plt.subplots(figsize=(10, 4))
    fig_sens.patch.set_facecolor("#0f172a")
    ax_sens.set_facecolor("#0f172a")
    ax_sens.plot(sqft_range, sim_preds, color="#34d399", linewidth=2.5, label="Predicted Valuation ($)")
    ax_sens.set_xlabel("Property Area (Square Feet)", color="white")
    ax_sens.set_ylabel("Predicted Valuation ($ USD)", color="white")
    ax_sens.set_title("Valuation Curve vs Square Footage", color="white")
    ax_sens.grid(True, linestyle="--", alpha=0.2)
    ax_sens.tick_params(colors="white")
    for spine in ax_sens.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig_sens)

# TAB 4: DATASET EXPLORER
with tab4:
    st.subheader("📈 Housing Dataset Overview")
    st.dataframe(df_data.head(10), use_container_width=True)
    st.write("Summary Statistics:")
    st.dataframe(df_data.describe().round(2), use_container_width=True)
