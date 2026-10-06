# ⚡ Interactive AI Predictive Dashboard (Streamlit & Gradio)

> An interactive machine learning web dashboard supporting **real-time model inference, what-if sensitivity analysis, feature importance visualizer, and dynamic dataset analytics** built using **Streamlit** (and optional **Gradio** interface).

---

## 🌟 Key Features

1. **🏡 Real-Time Valuation Predictor**:
   - Interactive sliders for property features (Square Footage, Bedrooms, Bathrooms, Age, Distance to City, Crime Index).
   - Instant live price prediction with confidence interval range and visual progress gauge.

2. **📊 Model Performance & Feature Analytics**:
   - Dynamic selection between **Random Forest**, **Gradient Boosting**, and **Decision Tree** models.
   - Interactive toggle for **StandardScaler**, **MinMaxScaler**, or **Raw Features**.
   - Feature Importance Bar Chart and Actual vs. Predicted scatter plot.

3. **🔍 What-If Sensitivity Explorer**:
   - Interactive curve tracking how changing square footage continuously impacts house price predictions.

4. **📈 Dataset Explorer**:
   - Browse raw training data rows and instant summary statistics.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Dashboard (Primary App)
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

### 3. Run Gradio Dashboard (Alternative Web UI)
Alternatively, launch the lightweight Gradio web interface:
```bash
python gradio_app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 📂 Project Structure

```
project_05_interactive_ai_dashboard/
├── README.md           # Dashboard overview and execution guide
├── requirements.txt    # Dependency requirements
├── app.py              # Main Streamlit Dashboard application
└── gradio_app.py       # Alternative Gradio Web UI interface
```
