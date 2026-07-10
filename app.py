from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_curve, confusion_matrix
import sys

# Suppress tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Constants
AMOUNT_MEAN = 42.177772
AMOUNT_STD = 86.698764
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = PCA_FEATURES + ["Amount_scaled", "Time_scaled"]

# Global cache for data and models
DATA = None
MODELS = {}
ROC_DATA = {}
CONFUSION_MATRICES = {}
FEATURE_IMPORTANCE = {}
STATS = {}

def load_resources():
    global DATA, MODELS, ROC_DATA, CONFUSION_MATRICES, FEATURE_IMPORTANCE, STATS
    print("Loading datasets and models...", flush=True)
    
    # 1. Load Data
    if os.path.exists("creditcard_demo.csv"):
        DATA = pd.read_csv("creditcard_demo.csv")
        # Compute basic dataset stats
        total = len(DATA)
        fraud = int(DATA['Class'].sum())
        genuine = total - fraud
        avg_amount = float(DATA['Amount'].mean())
        STATS = {
            "total_transactions": total,
            "fraud_count": fraud,
            "genuine_count": genuine,
            "fraud_rate": round((fraud / total) * 100, 4),
            "avg_amount": round(avg_amount, 2),
            "max_amount": round(float(DATA['Amount'].max()), 2)
        }
    else:
        print("Warning: creditcard_demo.csv not found!", flush=True)
        STATS = {
            "total_transactions": 284807,
            "fraud_count": 492,
            "genuine_count": 284315,
            "fraud_rate": 0.1727,
            "avg_amount": 88.35,
            "max_amount": 25691.16
        }

    # 2. Load Models
    try:
        MODELS["scaler"] = joblib.load("scaler.pkl")
    except Exception as e:
        print(f"Error loading scaler: {e}", flush=True)
        MODELS["scaler"] = None

    model_files = {
        "xgb": "xgboost_model.pkl",
        "rf": "random_forest_model.pkl",
        "lr": "logistic_regression_model.pkl",
        "best_classical": "best_classical_model.pkl"
    }

    for key, filename in model_files.items():
        try:
            if os.path.exists(filename):
                MODELS[key] = joblib.load(filename)
                print(f"Loaded {filename}", flush=True)
            else:
                print(f"Warning: {filename} not found", flush=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}", flush=True)

    # Load Keras Model
    try:
        if os.path.exists("neural_network_model.keras"):
            from tensorflow.keras.models import load_model
            MODELS["nn"] = load_model("neural_network_model.keras")
            print("Loaded neural_network_model.keras", flush=True)
        else:
            print("Warning: neural_network_model.keras not found", flush=True)
    except Exception as e:
        print(f"Error loading neural network: {e}", flush=True)

    # 3. Precompute ROC and Confusion Matrix if Data and Models are available
    if DATA is not None:
        # Preprocess features of demo dataset for evaluation
        eval_df = DATA.copy()
        
        # Scale Time
        if MODELS.get("scaler") is not None:
            eval_df["Time_scaled"] = MODELS["scaler"].transform(eval_df[["Time"]])
        else:
            eval_df["Time_scaled"] = (eval_df["Time"] - 85533.50709) / 49916.04121257
            
        # Scale Amount
        eval_df["Amount_scaled"] = (eval_df["Amount"] - AMOUNT_MEAN) / AMOUNT_STD
        
        X_eval = eval_df[FEATURE_ORDER]
        y_eval = eval_df["Class"]

        for m_key in ["xgb", "rf", "lr", "nn"]:
            model = MODELS.get(m_key)
            if model is None:
                continue

            try:
                # Predict probabilities
                if m_key == "nn":
                    y_probs = model.predict(X_eval, verbose=0).ravel()
                    y_preds = (y_probs >= 0.5).astype(int)
                else:
                    y_probs = model.predict_proba(X_eval)[:, 1]
                    y_preds = model.predict(X_eval)

                # Compute ROC Curve
                fpr, tpr, thresholds = roc_curve(y_eval, y_probs)
                
                # Downsample ROC points for performance (max 100 points)
                if len(fpr) > 100:
                    indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                    fpr = fpr[indices]
                    tpr = tpr[indices]
                
                ROC_DATA[m_key] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

                # Compute Confusion Matrix
                cm = confusion_matrix(y_eval, y_preds)
                CONFUSION_MATRICES[m_key] = {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1])
                }
            except Exception as e:
                print(f"Error precomputing metrics for {m_key}: {e}", flush=True)

        # Compute Feature Importance for XGB (best classical)
        xgb_model = MODELS.get("xgb")
        if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            feat_imp = sorted(zip(FEATURE_ORDER, importances.tolist()), key=lambda x: x[1], reverse=True)
            FEATURE_IMPORTANCE["xgb"] = {
                "features": [x[0] for x in feat_imp[:12]],
                "importances": [round(x[1], 5) for x in feat_imp[:12]]
            }

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(STATS)

@app.route("/api/dataset-sample", methods=["GET"])
def get_dataset_sample():
    if DATA is None:
        return jsonify([])
    
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)
    
    sample_df = DATA.iloc[offset:offset+limit].copy()
    
    # Return as list of records
    return jsonify(sample_df.to_dict(orient="records"))



@app.route("/api/dataset-distributions", methods=["GET"])
def get_dataset_distributions():
    """Return precomputed histogram bins, correlation matrix, and distribution data
    for all visual analytics charts matching the notebook graphs."""
    if DATA is None:
        return jsonify({"error": "Dataset not loaded"}), 400

    result = {}

    # 1. Amount distribution histogram (50 bins)
    amount_vals = DATA['Amount'].dropna()
    counts_a, bin_edges_a = np.histogram(amount_vals, bins=50)
    result['amount_hist'] = {
        'bin_centers': [round(float((bin_edges_a[i] + bin_edges_a[i+1]) / 2), 2) for i in range(len(counts_a))],
        'counts': counts_a.tolist()
    }

    # 2. Time distribution histogram (50 bins)
    time_vals = DATA['Time'].dropna()
    counts_t, bin_edges_t = np.histogram(time_vals, bins=50)
    result['time_hist'] = {
        'bin_centers': [round(float((bin_edges_t[i] + bin_edges_t[i+1]) / 2), 2) for i in range(len(counts_t))],
        'counts': counts_t.tolist()
    }

    # 3. Amount by class (for box-plot / violin)
    genuine = DATA[DATA['Class'] == 0]['Amount']
    fraud = DATA[DATA['Class'] == 1]['Amount']
    result['amount_by_class'] = {
        'genuine': genuine.tolist()[:500],  # cap for transfer size
        'fraud': fraud.tolist()[:500]
    }

    # 4. Time by class
    genuine_time = DATA[DATA['Class'] == 0]['Time']
    fraud_time = DATA[DATA['Class'] == 1]['Time']
    result['time_by_class'] = {
        'genuine': genuine_time.tolist()[:500],
        'fraud': fraud_time.tolist()[:500]
    }

    # 5. Correlation matrix for top features (V1-V14 + Amount + Time)
    corr_features = ['V1','V2','V3','V4','V5','V6','V7','V10','V11','V12','V14','V17','Amount','Time','Class']
    corr_df = DATA[corr_features].corr()
    result['correlation'] = {
        'labels': corr_features,
        'matrix': corr_df.values.tolist()
    }

    # 6. Model comparison metrics (for bar chart)
    result['model_comparison'] = {
        'models': list(MODEL_METADATA.keys()),
        'accuracy': [float(v['metrics']['Accuracy'].replace('%',''))/100 for v in MODEL_METADATA.values()],
        'f1': [float(v['metrics']['F1-Score'].replace('%',''))/100 for v in MODEL_METADATA.values()],
        'roc_auc': [float(v['metrics']['ROC-AUC']) for v in MODEL_METADATA.values()]
    }

    # 7. Neural Network Training Curves (from notebook logs)
    result['nn_history'] = {
        'epochs': list(range(1, 16)),
        'loss': [0.5831, 0.3669, 0.2393, 0.1659, 0.1219, 0.0827, 0.0654, 0.0560, 0.0433, 0.0357, 0.0285, 0.0236, 0.0237, 0.0178, 0.0178],
        'val_loss': [0.4003, 0.1625, 0.0798, 0.0393, 0.0185, 0.0103, 0.0074, 0.0048, 0.0032, 0.0028, 0.0020, 0.0015, 0.0013, 0.0009, 0.0007],
        'auc': [0.7572, 0.9213, 0.9670, 0.9838, 0.9909, 0.9958, 0.9971, 0.9975, 0.9987, 0.9991, 0.9993, 0.9994, 0.9993, 0.9997, 0.9994],
        'val_auc': [0.7012, 0.8524, 0.9210, 0.9634, 0.9810, 0.9892, 0.9915, 0.9930, 0.9942, 0.9951, 0.9960, 0.9968, 0.9971, 0.9975, 0.9980]
    }

    return jsonify(result)

@app.route("/api/model-performance", methods=["GET"])
def get_model_performance():
    return jsonify({
        "metadata": MODEL_METADATA,
        "roc_data": ROC_DATA,
        "confusion_matrices": CONFUSION_MATRICES,
        "feature_importance": FEATURE_IMPORTANCE
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        req_data = request.get_json()
        model_name = req_data.get("model", "XGBoost (Best Classical Model)")
        
        # Resolve model key
        model_key = "xgb"
        for name, info in MODEL_METADATA.items():
            if name == model_name:
                model_key = info["key"]
                break
                
        model = MODELS.get(model_key)
        if model is None:
            return jsonify({"error": f"Model {model_name} is not loaded."}), 400
            
        time_val = float(req_data.get("Time", 0.0))
        amount_val = float(req_data.get("Amount", 0.0))
        
        # Scale Time
        if MODELS.get("scaler") is not None:
            time_scaled = float(MODELS["scaler"].transform([[time_val]])[0][0])
        else:
            time_scaled = (time_val - 85533.50709) / 49916.04121257
            
        # Scale Amount
        amount_scaled = (amount_val - AMOUNT_MEAN) / AMOUNT_STD
        
        features = {}
        for feature in PCA_FEATURES:
            features[feature] = float(req_data.get(feature, 0.0))
            
        features["Amount_scaled"] = amount_scaled
        features["Time_scaled"] = time_scaled
        
        input_df = pd.DataFrame([features])[FEATURE_ORDER]
        
        # Run Predict
        if model_key == "nn":
            prob = float(model.predict(input_df, verbose=0)[0][0])
            pred = 1 if prob >= 0.5 else 0
        else:
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(model.predict(input_df)[0])
            
        return jsonify({
            "prediction": pred,
            "probability": prob,
            "verdict": "Fraudulent" if pred == 1 else "Genuine",
            "model_used": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        model_name = request.form.get("model", "XGBoost (Best Classical Model)")
        model_key = "xgb"
        for name, info in MODEL_METADATA.items():
            if name == model_name:
                model_key = info["key"]
                break
                
        model = MODELS.get(model_key)
        if model is None:
            return jsonify({"error": f"Model {model_name} is not loaded."}), 400

        batch_df = pd.read_csv(file)
        
        # Validate columns
        required_cols = ["Time", "Amount"] + PCA_FEATURES
        missing_cols = [c for c in required_cols if c not in batch_df.columns]
        if len(missing_cols) > 0:
            return jsonify({"error": f"Missing required columns in CSV: {missing_cols}"}), 400
            
        # Preprocess
        batch_proc = batch_df.copy()
        if MODELS.get("scaler") is not None:
            batch_proc["Time_scaled"] = MODELS["scaler"].transform(batch_proc[["Time"]])
        else:
            batch_proc["Time_scaled"] = (batch_proc["Time"] - 85533.50709) / 49916.04121257
            
        batch_proc["Amount_scaled"] = (batch_proc["Amount"] - AMOUNT_MEAN) / AMOUNT_STD
        
        features_df = batch_proc[FEATURE_ORDER]
        
        # Predict
        if model_key == "nn":
            probs = model.predict(features_df, verbose=0).ravel()
            preds = (probs >= 0.5).astype(int)
        else:
            probs = model.predict_proba(features_df)[:, 1]
            preds = model.predict(features_df)
            
        batch_df["Fraud Probability (%)"] = np.round(probs * 100, 2)
        batch_df["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in preds]
        
        total_cases = len(batch_df)
        fraud_cases = int(sum(preds))
        genuine_cases = total_cases - fraud_cases
        
        # Convert first 100 results to list of dicts to display in frontend
        preview_rows = batch_df.head(100).to_dict(orient="records")
        # Keep full output in a static variable to download later if they click download (or send full CSV back)
        # To keep it simple, we can return the full CSV content encoded as string or we can send the whole file data.
        # Let's send the list of predicted rows (all of them) as JSON, so frontend can build the CSV download fully in JS!
        full_rows = batch_df.to_dict(orient="records")
        
        return jsonify({
            "total_transactions": total_cases,
            "genuine_count": genuine_cases,
            "fraud_count": fraud_cases,
            "fraud_rate": round((fraud_cases / total_cases) * 100, 2) if total_cases > 0 else 0.0,
            "preview_rows": preview_rows,
            "full_rows": full_rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

load_resources()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
