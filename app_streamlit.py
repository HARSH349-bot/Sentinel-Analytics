import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9fb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allowed_html=True)

# Define exact feature order expected by the models
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = PCA_FEATURES + ["Amount_scaled", "Time_scaled"]

# Verified training scaling parameters (Amount parameters + Time scaler)
AMOUNT_MEAN = 42.177772
AMOUNT_STD = 86.698764

# Cache data loading
@st.cache_data
def load_data():
    if os.path.exists("creditcard_demo.csv"):
        return pd.read_csv("creditcard_demo.csv")
    return None

# Cache model loading
@st.cache_resource
def load_models():
    models = {}
    try:
        models["scaler"] = joblib.load("scaler.pkl")
        models["best_classical"] = joblib.load("best_classical_model.pkl")
        models["lr"] = joblib.load("logistic_regression_model.pkl")
        models["rf"] = joblib.load("random_forest_model.pkl")
        models["xgb"] = joblib.load("xgboost_model.pkl")
        
        # Load Keras model
        from tensorflow.keras.models import load_model
        models["nn"] = load_model("neural_network_model.keras")
    except Exception as e:
        st.error(f"Error loading models/scaler: {e}")
    return models

# Load resources
df = load_data()
models_dict = load_models()

# Model Metadata (Metrics from training notebook)
MODEL_METADATA = {
    "XGBoost (Best Classical Model)": {
        "key": "xgb",
        "desc": "Extreme Gradient Boosting model trained on SMOTE resampled data. Highly sensitive to fraud patterns.",
        "metrics": {"Accuracy": "99.25%", "F1-Score": "14.29%", "ROC-AUC": "0.7981"}
    },
    "Random Forest": {
        "key": "rf",
        "desc": "Ensemble of decision trees. Balanced structure for classification.",
        "metrics": {"Accuracy": "98.76%", "F1-Score": "0.00%", "ROC-AUC": "0.7375"}
    },
    "Logistic Regression": {
        "key": "lr",
        "desc": "Standard linear classification baseline trained with resampled data.",
        "metrics": {"Accuracy": "80.16%", "F1-Score": "1.24%", "ROC-AUC": "0.6130"}
    },
    "Neural Network (Deep Learning)": {
        "key": "nn",
        "desc": "Multi-layer Perceptron (Keras) model optimized with binary cross-entropy.",
        "metrics": {"Accuracy": "99.38%", "F1-Score": "16.67%", "ROC-AUC": "0.5378"}
    }
}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield-with-key.png", width=80)
    st.title("Fraud Guard Settings")
    st.write("Configure model deployment details.")
    
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Choose Inference Model:",
        options=list(MODEL_METADATA.keys())
    )
    
    selected_model_info = MODEL_METADATA[model_name]
    st.write(selected_model_info["desc"])
    
    st.subheader("Model Validation Metrics")
    for key, val in selected_model_info["metrics"].items():
        st.metric(label=key, value=val)

# ----------------- MAIN LAYOUT -----------------
st.title("💳 Credit Card Fraud Detection Hub")
st.write("Analyze and flag fraudulent transactions in real time using Machine Learning and Deep Learning.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Transaction Predictor", 
    "📊 Batch Predictor (CSV)", 
    "📈 Model Performance", 
    "🗃️ Dataset Explorer"
])

# ----------------- TAB 1: SINGLE PREDICTOR -----------------
with tab1:
    st.header("Single Transaction Inference")
    st.write("Load a template transaction from the dataset or adjust features manually to predict fraud.")
    
    # Preset selection
    if df is not None:
        col_preset, _ = st.columns([2, 2])
        with col_preset:
            preset_type = st.selectbox(
                "Quick Test: Load Sample Transaction",
                options=["Manual Input", "Genuine Transaction Sample", "Fraudulent Transaction Sample"]
            )
        
        # Load preset logic
        preset_data = {}
        if preset_type == "Genuine Transaction Sample":
            genuine_samples = df[df['Class'] == 0]
            if len(genuine_samples) > 0:
                preset_data = genuine_samples.iloc[0].to_dict()
        elif preset_type == "Fraudulent Transaction Sample":
            fraud_samples = df[df['Class'] == 1]
            if len(fraud_samples) > 0:
                preset_data = fraud_samples.iloc[0].to_dict()
    else:
        preset_type = "Manual Input"
        preset_data = {}

    col_inputs_1, col_inputs_2 = st.columns([1, 1])
    
    with col_inputs_1:
        st.subheader("Transaction Metadata")
        time_val = st.number_input(
            "Transaction Time (seconds from start):",
            value=float(preset_data.get("Time", 0.0)),
            min_value=0.0,
            step=100.0
        )
        amount_val = st.number_input(
            "Transaction Amount ($):",
            value=float(preset_data.get("Amount", 1.0)),
            min_value=0.0,
            step=5.0
        )

    with col_inputs_2:
        st.subheader("PCA Anonymized Features")
        with st.expander("Modify V1 - V28 Features"):
            st.write("These represent anonymized transaction components.")
            pca_inputs = {}
            # Display sliders in 4 columns inside the expander
            cols = st.columns(4)
            for idx, feature in enumerate(PCA_FEATURES):
                col_idx = idx % 4
                default_val = float(preset_data.get(feature, 0.0))
                pca_inputs[feature] = cols[col_idx].number_input(
                    feature,
                    value=default_val,
                    format="%.4f",
                    step=0.1
                )

    st.markdown("---")
    
    # Prediction Section
    col_pred_btn, col_result = st.columns([1, 2])
    
    with col_pred_btn:
        st.write("Run model to evaluate transaction risk.")
        predict_clicked = st.button("Run Fraud Analysis", type="primary", use_container_width=True)

    if predict_clicked:
        # Preprocess input data
        # 1. Scale Time
        if "scaler" in models_dict:
            time_scaled = models_dict["scaler"].transform([[time_val]])[0][0]
        else:
            time_scaled = (time_val - 85533.50709) / 49916.04121257  # fallback

        # 2. Scale Amount
        amount_scaled = (amount_val - AMOUNT_MEAN) / AMOUNT_STD

        # 3. Create DataFrame in correct order
        features = {}
        for feature in PCA_FEATURES:
            features[feature] = pca_inputs[feature]
        features["Amount_scaled"] = amount_scaled
        features["Time_scaled"] = time_scaled

        input_df = pd.DataFrame([features])[FEATURE_ORDER]

        # Predict
        model_key = selected_model_info["key"]
        model = models_dict[model_key]

        if model_key == "nn":
            prob = float(model.predict(input_df)[0][0])
            pred = 1 if prob >= 0.5 else 0
        else:
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(model.predict(input_df)[0])

        with col_result:
            res_col_1, res_col_2 = st.columns([1, 1])
            with res_col_1:
                if pred == 1:
                    st.error("### 🚨 FRAUD DETECTED")
                    st.write(f"The transaction is flagged as **HIGH RISK**. Fraud Probability is high.")
                else:
                    st.success("### ✅ GENUINE TRANSACTION")
                    st.write("The transaction is marked as **LOW RISK** and safe to process.")
            
            with res_col_2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Fraud Probability (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if pred == 1 else "#10b981"},
                        'steps': [
                            {'range': [0, 30], 'color': "#e8f5e9"},
                            {'range': [30, 70], 'color': "#fff3e0"},
                            {'range': [70, 100], 'color': "#ffebee"}
                        ],
                    }
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: BATCH PREDICTOR -----------------
with tab2:
    st.header("Batch Prediction via CSV Upload")
    st.write("Upload a CSV file containing transactions. It must contain the columns `Time`, `Amount`, and `V1` to `V28`.")
    
    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success("CSV Uploaded successfully!")
        
        # Validate columns
        required_cols = ["Time", "Amount"] + PCA_FEATURES
        missing_cols = [c for c in required_cols if c not in batch_df.columns]
        
        if len(missing_cols) > 0:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            # Preprocess
            batch_proc = batch_df.copy()
            
            # Scale
            if "scaler" in models_dict:
                # Time scaling
                batch_proc["Time_scaled"] = models_dict["scaler"].transform(batch_proc[["Time"]])
            else:
                batch_proc["Time_scaled"] = (batch_proc["Time"] - 85533.50709) / 49916.04121257
            
            batch_proc["Amount_scaled"] = (batch_proc["Amount"] - AMOUNT_MEAN) / AMOUNT_STD
            
            # Predict
            model_key = selected_model_info["key"]
            model = models_dict[model_key]
            
            features_df = batch_proc[FEATURE_ORDER]
            
            if model_key == "nn":
                probs = model.predict(features_df).ravel()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = model.predict_proba(features_df)[:, 1]
                preds = model.predict(features_df)
                
            batch_df["Fraud Probability (%)"] = np.round(probs * 100, 2)
            batch_df["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in preds]
            
            # Display Metrics
            total_cases = len(batch_df)
            fraud_cases = int(sum(preds))
            genuine_cases = total_cases - fraud_cases
            
            m_col_1, m_col_2, m_col_3 = st.columns(3)
            m_col_1.metric("Total Transactions", total_cases)
            m_col_2.metric("Genuine Flagged", genuine_cases)
            m_col_3.metric("Fraud Flagged", fraud_cases, delta=f"{fraud_cases/total_cases*100:.2f}%", delta_color="inverse")
            
            # Plots
            st.subheader("Batch Analysis Visualizations")
            col_chart_1, col_chart_2 = st.columns([1, 1])
            
            with col_chart_1:
                fig_pie = px.pie(
                    names=["Genuine", "Fraud"],
                    values=[genuine_cases, fraud_cases],
                    color=["Genuine", "Fraud"],
                    color_discrete_map={"Genuine": "#10b981", "Fraud": "#ef4444"},
                    title="Transaction Risk Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart_2:
                fig_hist = px.histogram(
                    batch_df, 
                    x="Fraud Probability (%)",
                    nbins=20,
                    title="Fraud Probabilities Distribution",
                    color_discrete_sequence=["#4f46e5"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            st.subheader("Prediction Results Table")
            st.dataframe(batch_df.head(100))
            
            # Download file
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="transaction_predictions.csv",
                mime="text/csv"
            )

# ----------------- TAB 3: PERFORMANCE -----------------
with tab3:
    st.header("Model Performance & Comparison")
    st.write("Metrics compiled from the training runs on `creditcard_demo.csv` dataset (8,040 samples with 0.49% fraud class imbalance).")
    
    comp_data = []
    for name, info in MODEL_METADATA.items():
        comp_data.append({
            "Model": name,
            "Accuracy": float(info["metrics"]["Accuracy"].replace("%", "")) / 100.0,
            "F1-Score": float(info["metrics"]["F1-Score"].replace("%", "")) / 100.0,
            "ROC-AUC": float(info["metrics"]["ROC-AUC"])
        })
    comp_df = pd.DataFrame(comp_data)
    
    col_tbl, col_chart = st.columns([1, 1])
    
    with col_tbl:
        st.write("### Model Metrics Summary Table")
        st.dataframe(comp_df.style.format({
            "Accuracy": "{:.2%}",
            "F1-Score": "{:.2%}",
            "ROC-AUC": "{:.4f}"
        }))
        
        st.info("""
        **Why is the F1-Score low?**
        
        Because the dataset is extremely imbalanced (only 40 fraud cases out of 8,040 records).
        While **Accuracy** is high across all models, **ROC-AUC** is a much more robust indicator of performance in this context, where **XGBoost** excels at finding fraud cases.
        """)
        
    with col_chart:
        fig_bar = px.bar(
            comp_df,
            x="Model",
            y=["ROC-AUC", "F1-Score"],
            barmode="group",
            title="ROC-AUC vs F1-Score Comparison",
            color_discrete_sequence=["#4f46e5", "#10b981"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- TAB 4: DATASET EXPLORER -----------------
with tab4:
    st.header("Dataset Explorer")
    if df is not None:
        st.write(f"The `creditcard_demo.csv` file loaded successfully. Shape: {df.shape}")
        
        st.subheader("Data Summary & Distribution")
        col_ed_1, col_ed_2 = st.columns([1, 1])
        
        with col_ed_1:
            fig_class = px.histogram(
                df, 
                x="Class", 
                title="Class Count (0 = Genuine, 1 = Fraud)",
                color="Class",
                color_discrete_sequence=["#10b981", "#ef4444"]
            )
            st.plotly_chart(fig_class, use_container_width=True)
            
        with col_ed_2:
            fig_amount = px.box(
                df, 
                y="Amount", 
                x="Class", 
                title="Amount Boxplot by Class",
                color="Class",
                color_discrete_sequence=["#10b981", "#ef4444"]
            )
            st.plotly_chart(fig_amount, use_container_width=True)
            
        st.subheader("Explore Raw Dataset Rows")
        row_limit = st.slider("Select row count to display:", 5, 200, 15)
        st.dataframe(df.head(row_limit))
    else:
        st.warning("`creditcard_demo.csv` was not found in the workspace. Upload a CSV file or copy it to the workspace directory to enable this tab.")
