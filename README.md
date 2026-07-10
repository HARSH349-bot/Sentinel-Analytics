# Sentinel Analytics: AI-Powered Credit Card Fraud Detection Suite

Sentinel Analytics is an end-to-end Machine Learning Capstone project designed to classify, analyze, and mitigate fraudulent credit card transactions in real-time. The system integrates advanced ensemble models (XGBoost, Random Forest) and Deep Learning (TensorFlow/Keras Multi-Layer Perceptrons) with a highly interactive Flask Single-Page Application (SPA) dashboard styled using Tailwind CSS and Plotly.js.

---

## 📊 System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data Pipeline
        CC[creditcard_demo.csv] --> Pre[Data Preprocessing]
        Pre --> Scalers[Amount Scaler / Time StandardScaler]
        Scalers --> Split[Train/Test Stratified Split]
        Split --> SMOTE[SMOTE Minority Oversampling]
    end

    subgraph Model Training
        SMOTE --> XGB[XGBoost Classifier]
        SMOTE --> RF[Random Forest]
        SMOTE --> LR[Logistic Regression]
        SMOTE --> NN[TensorFlow Neural Net]
        
        XGB --> SaveXGB[xgboost_model.pkl / best_classical_model.pkl]
        RF --> SaveRF[random_forest_model.pkl]
        LR --> SaveLR[logistic_regression_model.pkl]
        NN --> SaveNN[neural_network_model.keras]
    end

    subgraph Web App Backend (Flask)
        SaveXGB & SaveRF & SaveLR & SaveNN & Scalers --> Flask[Flask Web Server: app.py]
        CC --> Flask
    end

    subgraph Frontend Dashboard (SPA)
        Flask -->|Serve UI| SPA[HTML/JS/Tailwind Dashboard]
        SPA -->|Single Prediction Request| API_Pred[/api/predict]
        SPA -->|Batch CSV Upload| API_Batch[/api/predict-batch]
        SPA -->|Visual Analytics Fetch| API_Stats[/api/dataset-distributions & /api/stats]
        
        API_Pred & API_Batch & API_Stats --> Flask
    end
```

---

## 🛠️ ML Lifecycle & Workflows

### 1. Exploratory Data Analysis & Imbalance Resolution
- **Class Imbalance:** Fraudulent transactions represent only **0.49%** of the raw records (40 fraud vs. 8,000 genuine).
- **Oversampling (SMOTE):** Synthetic Minority Over-sampling Technique is applied during the training phase to generate synthetic fraud samples and prevent classifiers from defaulting to genuine verdicts.

### 2. Feature Preprocessing
- **Anonymization (V1 - V28):** Principal Component Analysis (PCA) components are fed directly into the model pipelines.
- **Scaling:** Features `Amount` and `Time` are normalized using fitted training parameters:
  - `Amount_scaled = (Amount - 42.177772) / 86.698764`
  - `Time_scaled = StandardScaler().fit_transform(Time)` (via `scaler.pkl`)

### 3. Model Performance Comparison

| Model | Accuracy | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Best Classical)** | **99.25%** | **14.29%** | **0.7981** | **Selected (Default)** |
| Neural Network (Deep Learning) | 99.38% | 16.67% | 0.5378 | Alternative |
| Random Forest | 98.76% | 0.00% | 0.7375 | Alternative |
| Logistic Regression | 80.16% | 1.24% | 0.6130 | Baseline |


---


## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10 or higher.
- Git.

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install flask pandas numpy scikit-learn xgboost tensorflow joblib
```

### 2. Run the Application
Start the local web server:
```bash
python app.py
```

### 3. Access the Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure

```
Sentinel-Analytics/
├── app.py                      # Flask Server (API endpoints & model loading)
├── app_streamlit.py            # Streamlit Legacy Backup
├── creditcard_demo.csv          # Validation dataset sample (8,040 rows)
├── scaler.pkl                  # Fitted Time StandardScaler
├── xgboost_model.pkl           # Trained XGBoost model
├── random_forest_model.pkl     # Trained Random Forest model
├── logistic_regression_model.pkl # Trained Logistic Regression baseline
├── best_classical_model.pkl    # Identical to xgboost_model.pkl (Best classical model)
├── neural_network_model.keras  # Trained TensorFlow Keras model
├── templates/
│   └── index.html              # SPA Dashboard UI (HTML/JS/Plotly.js)
└── .gitignore                  # Git untracked pattern list
```

---
*Created as part of the AI Capstone Project Suite.*
