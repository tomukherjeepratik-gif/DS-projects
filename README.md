# 🎓 ACM Student Chapter — AI & Machine Learning Projects Repository

> A comprehensive collection of Artificial Intelligence, Deep Learning, Machine Learning, and Web API projects developed under the **ACM Student Chapter**.

---

## 📂 Project Suite Directory

| Project Folder | Domain / Task | Key Technologies | Status |
| :--- | :--- | :--- | :---: |
| 🎓 **[`project_02_nexus_rag_assistant/`](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_02_nexus_rag_assistant/README.md)** | Retrieval-Augmented Generation (RAG) College Assistant | React 19, TypeScript, TF-IDF + BM25 Vector Engine, Python | ✅ Ready |
| 🖼️ **[`project_03_image_captioning/`](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_03_image_captioning/README.md)** | Deep Learning Image Caption Generator (Encoder–Decoder) | PyTorch, ResNet-50 CNN, LSTM RNN, Torchvision | ✅ Ready |
| ⚡ **[`project_04_ml_model_web_api/`](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_04_ml_model_web_api/README.md)** | Scikit-Learn Predictive Model Web API Microservice | FastAPI, Uvicorn, Scikit-Learn, Pydantic v2 | ✅ Ready |
| 🎛️ **[`project_05_interactive_ai_dashboard/`](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_05_interactive_ai_dashboard/README.md)** | Real-Time Interactive Machine Learning Dashboard | Streamlit, Gradio, Scikit-Learn, Plotly/Matplotlib | ✅ Ready |
| 🏡 **[`project_06_end_to_end_predictive_model/`](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_06_end_to_end_predictive_model/README.md)** | End-to-End Housing Price Predictor & Benchmark | Decision Trees, Random Forests, Feature Scaling, Jupyter | ✅ Ready |

---

## 🌟 Overview of Included Projects

### 2. 🎓 [Nexus RAG College Assistant](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_02_nexus_rag_assistant/README.md) (`project_02_nexus_rag_assistant/`)
Production-grade Retrieval-Augmented Generation (RAG) assistant for a college campus. Features hybrid TF-IDF + BM25 vector search, precise chunk attribution modal, interactive chunk inspector, and a zero-dependency Python CLI RAG pipeline (`rag_assistant.py`).

### 3. 🖼️ [Image Captioning using Deep Learning](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_03_image_captioning/README.md) (`project_03_image_captioning/`)
Deep-learning-based image caption generator using a PyTorch **CNN Encoder (ResNet-50)** and **RNN Decoder (LSTM)**. Automatically converts visual representations into natural language descriptions (e.g. `"A dog is playing with a ball in the park."`). Includes trained weights (`encoder_decoder_captioning.pt`) and runnable demo (`demo.py`).

### 4. ⚡ [ML Model Web API](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_04_ml_model_web_api/README.md) (`project_04_ml_model_web_api/`)
High-performance **FastAPI microservice** serving Scikit-Learn `RandomForestRegressor` and `StandardScaler` predictions. Features Pydantic request/response validation, `/predict` and `/batch-predict` JSON endpoints, interactive Swagger UI (`/docs`), and automated `TestClient` test runner (`test_api.py`).

### 5. 🎛️ [Interactive AI Dashboard](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_05_interactive_ai_dashboard/README.md) (`project_05_interactive_ai_dashboard/`)
Interactive ML web application deployed using **Streamlit** (plus secondary **Gradio** web UI). Provides real-time sliders, valuation confidence intervals, model hyperparameter tuning, feature importance charts, and what-if sensitivity curves.

### 6. 🏡 [End-to-End Predictive Model](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_06_end_to_end_predictive_model/README.md) (`project_06_end_to_end_predictive_model/`)
Complete machine learning pipeline using **Decision Trees** and **Random Forests** with **Feature Scaling (StandardScaler vs. MinMaxScaler)**. Includes interactive Jupyter Notebook (`predictive_model.ipynb`), executable CLI pipeline (`train_and_evaluate.py`), GridSearchCV optimization, and evaluation metrics.

---

## 🛠️ Tech Stack Across Repository

- **Frontend & Web UIs**: React 19, TypeScript 5.8, Streamlit 1.65, Gradio 6.29, FastAPI 0.115
- **Deep Learning & Computer Vision**: PyTorch 2.14, Torchvision 0.29, ResNet-50, LSTM, PIL
- **Machine Learning & Data Science**: Scikit-Learn 1.7, Pandas 2.2, NumPy 2.2, Matplotlib, Seaborn, Joblib, Jupyter
