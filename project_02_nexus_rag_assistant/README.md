# 🎓 Nexus RAG College Assistant

> A production-grade **Retrieval-Augmented Generation (RAG)** AI Assistant for a fictional college: **Nexus Institute of Technology (NIT)**.

Built with **React, TypeScript, Lucide Icons, and Vanilla Glassmorphism CSS**, plus a standalone **Python CLI RAG pipeline** (`rag_assistant.py`).

---

## 🌟 Key Features

1. **📚 Grounded College Knowledge Base**
   - Pre-loaded with official markdown documents covering:
     - **Courses & Syllabus**: B.Tech CSE (CS101, CS204, AI301, AI405 LLMs & RAG), Data Science, M.Tech Autonomous Systems, and Grading scales.
     - **Tuition Fees & Scholarships**: Semester fee structures, hostel/mess charges, late payment fines, 100% Academic Excellence Scholarship, 30% Women in STEM waiver.
     - **Clubs & Societies**: ACM Student Chapter (weekly meetings, **NexusHacks 36-Hour Hackathon**), Robotics Club, CyberSec Guild, Cultural Society.
     - **Timetables & Calendar**: Lecture time slots, academic calendar, exam dates, 24/7 library exam hours.
     - **Faculty Directory**: Contact info, offices, research areas, and student consultation hours for Dr. Elena Vance (HoD), Prof. Marcus Sterling, Prof. Aris Thorne.
     - **Rules & Campus Policies**: Minimum 75% attendance policy, anti-plagiarism & AI usage rules, hostel curfew (10:30 PM).

2. **🔍 Dual Dense/Sparse Hybrid Vector Search**
   - Computes **TF-IDF normalized vector space cosine similarity** combined with **BM25 keyword relevance** for high-precision retrieval.
   - Smart semantic document parser with customizable **Chunk Size (words)** and **Chunk Overlap**.

3. **📍 Precise Source & Chunk Attribution**
   - Every generated answer includes clickable citation pills (`[Source: Fees & Scholarships | ID: FEES-01]`).
   - Clicking any citation opens a **Source Inspector Modal** displaying the exact raw chunk text, token highlights, cosine similarity %, and term score breakdown.

4. **💬 Multi-Turn Conversation History (Bonus Feature)**
   - Remembers previous query context (e.g. asking "Tell me about the ACM Chapter" followed by "Who is the faculty sponsor for it?").
   - History depth badge, session export to `.txt`, and one-click history clearing.

5. **🔬 Interactive Chunking & Vector Inspector Tab**
   - Live visual grid of all document vector chunks.
   - Interactive search simulator: type any query to calculate cosine vector similarity heatmaps in real-time.

6. **📊 Automated RAG Benchmark & Test Suite Tab**
   - Includes 6+ pre-loaded test queries evaluating courses, fees, clubs, timetables, faculty, and campus rules.
   - One-click **"Run All RAG Evaluation Tests"** measuring pass accuracy %, average similarity score %, and query latency (ms).

7. **⚡ Zero External Dependency Python RAG Assistant (`rag_assistant.py`)**
   - Provides an interactive command-line interface and test runner written purely in standard Python 3.

---

## 🚀 Getting Started

### 1. Web Application (React + Vite UI)

```bash
# Install dependencies
npm install

# Start local development server
npm run dev
```

Open your browser at `http://localhost:5173` to explore the RAG Assistant.

To build the production bundle:
```bash
npm run build
```

---

### 2. Python CLI Assistant (`rag_assistant.py`)

Run the self-contained Python RAG test suite:
```bash
python3 rag_assistant.py --test
```

Or start the interactive Python command-line assistant:
```bash
python3 rag_assistant.py
```

---

## 🧪 RAG Architecture & Core Concepts

### 1. Document Chunking & Overlap
Large markdown documents are parsed into overlapping chunks (default: **140 words** per chunk with **30 words overlap**).
- **Why Chunking?** LLMs have context limits, and retrieving focused chunks reduces noise.
- **Why Overlap?** Overlap prevents cutting sentences or key entities (like course codes `CS204` or fee amounts `$4,500`) in half across chunk boundaries.

### 2. Vector Embeddings & Hybrid Retrieval
- **TF-IDF Vector Space**: Words are tokenized, stop-words removed, and sublinear term frequency ($1 + \log(\text{tf})$) scaled by smooth Inverse Document Frequency ($\text{IDF}$). Vectors are L2 normalized.
- **Cosine Similarity**: Measures the angle between query and document vectors:
  $$\text{CosineSim}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}$$
- **Hybrid Fusion**: Blends dense cosine similarity with BM25 term frequency matching for pinpoint accuracy on specific keywords.

### 3. Grounded Prompt Engineering (Zero Hallucination)
The RAG engine injects top-$K$ retrieved chunks into a strict system prompt:
> *"Answer using ONLY the provided document chunks. Always cite the exact Chunk ID."*
If no chunks meet the similarity threshold, the assistant refrains from guessing and explicitly states that the context is unavailable.

---

## 📋 Sample Test Queries & Expected Outcomes

| Question | Target Document | Key Retrieved Facts / Citations |
| :--- | :--- | :--- |
| **"What is the tuition fee for B.Tech and what scholarships exist?"** | `fees_and_scholarships.md` | `$4,500/semester` (₹1,25,000 INR), 100% Academic Excellence Scholarship (CGPA >= 9.50), 30% Women in STEM. |
| **"When does the ACM Student Chapter meet?"** | `clubs_and_societies.md` | Every **Wednesday at 5:00 PM** in Lab 304 (Building A), Faculty Sponsor: Prof. Aris Thorne, Flagship event: **NexusHacks**. |
| **"Who is the HOD of Computer Science and her office hours?"** | `faculty_directory.md` | **Dr. Elena Vance** (Building A - Room 402, `evance@nexus.edu`), Office Hours: Mon & Wed 2:00 – 4:00 PM. |
| **"What is the minimum attendance required to write exams?"** | `rules_regulations_policies.md` | Minimum **75% attendance** required; medical condonation allowed between 65%-74%. |
| **"What are the prerequisites for AI405 LLMs & RAG?"** | `courses_and_syllabus.md` | Prerequisite: **AI301** (Machine Learning). Taught by Dr. Elena Vance. |
| **"What time does the library close on regular days vs exam weeks?"** | `timetables_and_calendar.md` | Regular: **08:00 AM – 11:00 PM** (Mon-Fri); Exam periods: **24 Hours / 7 Days Open**. |


---

## 🚀 ACM Student Chapter AI & ML Project Suite

In addition to the **Nexus RAG College Assistant**, this repository contains four dedicated deep learning and machine learning projects located in separate project folders:

### 3. 🖼️ [Image Captioning using Deep Learning](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_03_image_captioning/README.md) (`project_03_image_captioning/`)
- **Task**: Deep-learning-based image caption generator using an **Encoder–Decoder architecture** (ResNet-50 CNN Encoder + LSTM RNN Decoder).
- **Objective**: Converts visual feature representations into natural-language captions (e.g., `"A dog is playing with a ball in the park."`).
- **Features**: Includes PyTorch models (`model.py`), vocabulary builder (`vocabulary.py`), synthetic benchmark image generator (`generate_samples.py`), trained model checkpoint, and runnable demo script (`demo.py`).

### 4. ⚡ [ML Model Web API](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_04_ml_model_web_api/README.md) (`project_04_ml_model_web_api/`)
- **Task**: Production-grade **FastAPI microservice** wrapping a trained Scikit-Learn `RandomForestRegressor` and `StandardScaler` pipeline serving JSON predictions over HTTP.
- **Features**: Strict Pydantic request validation, single (`POST /predict`) and batch (`POST /batch-predict`) prediction endpoints, `/health` endpoint, interactive Swagger UI (`/docs`), and automated `TestClient` suite (`test_api.py`).

### 5. 🎛️ [Interactive AI Dashboard](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_05_interactive_ai_dashboard/README.md) (`project_05_interactive_ai_dashboard/`)
- **Task**: Real-time interactive ML web dashboard built using **Streamlit** (plus secondary **Gradio** web UI).
- **Features**: Dynamic property feature sliders, live valuation confidence range, hyperparameter tuner (Random Forest, Gradient Boosting, Decision Tree), feature importance charts, and what-if sensitivity curves.

### 6. 🏡 [End-to-End Predictive Model](file:///Users/pratikmukherjee/Documents/Pratik%20Mukherjee/Codes/Project_ACM_StudentChapter/project_06_end_to_end_predictive_model/README.md) (`project_06_end_to_end_predictive_model/`)
- **Task**: End-to-end Housing Price Predictor comparing **Decision Trees** and **Random Forests** with **Feature Scaling (StandardScaler vs. MinMaxScaler)**.
- **Features**: Complete Jupyter Notebook (`predictive_model.ipynb`), executable CLI pipeline (`train_and_evaluate.py`), GridSearchCV hyperparameter optimization, model benchmarking table ($R^2$, RMSE, MAE), and evaluation plots.

---

## 📂 Project Structure

```
Project_ACM_StudentChapter/
├── README.md                                 # Main Project suite documentation
├── rag_assistant.py                          # Standalone Python CLI RAG Assistant
├── data/                                     # College Knowledge Base documents
├── src/                                      # React 19 + TypeScript RAG Assistant UI
├── project_03_image_captioning/              # Project 3: CNN-LSTM Image Caption Generator
│   ├── README.md
│   ├── model.py
│   ├── vocabulary.py
│   ├── generate_samples.py
│   └── demo.py
├── project_04_ml_model_web_api/              # Project 4: FastAPI Microservice
│   ├── README.md
│   ├── train_model.py
│   ├── app.py
│   ├── test_api.py
│   └── model.joblib
├── project_05_interactive_ai_dashboard/      # Project 5: Streamlit & Gradio Dashboard
│   ├── README.md
│   ├── app.py
│   └── gradio_app.py
└── project_06_end_to_end_predictive_model/   # Project 6: Decision Tree & Random Forest Pipeline
    ├── README.md
    ├── predictive_model.ipynb
    └── train_and_evaluate.py
```

---

## 🛠️ Tech Stack

- **Frontend & Web UIs**: React 19, TypeScript 5.8, Streamlit 1.65, Gradio 6.29, FastAPI 0.115
- **Deep Learning & Computer Vision**: PyTorch 2.14, Torchvision 0.29, ResNet-50, LSTM, PIL
- **Machine Learning & Data Science**: Scikit-Learn 1.7, Pandas 2.2, NumPy 2.2, Matplotlib, Seaborn, Joblib

