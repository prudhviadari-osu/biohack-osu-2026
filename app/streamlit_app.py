# -----------------------------
# streamlit_app.py
# -----------------------------
import os
import webbrowser
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kidney Pre-Perfusion Risk Dashboard", layout="wide")
st.title("Kidney Pre-Perfusion Risk Predictor")

# -----------------------------
# Paths
# -----------------------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_folder = os.path.join(repo_root, "models")

def find_data_dir(base_dir, keywords, max_depth=2):
    for root, dirs, files in os.walk(base_dir):
        rel_depth = os.path.relpath(root, base_dir).count(os.sep)
        if rel_depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith(".csv") and any(k in f.lower() for k in keywords):
                return root
    return base_dir

data_folder = find_data_dir(os.path.join(repo_root, "data"), ["referral", "opo"])

# -----------------------------
# Load models
# -----------------------------
left_model_path = os.path.join(models_folder, "left_kidney_model.pkl")
right_model_path = os.path.join(models_folder, "right_kidney_model.pkl")

if not os.path.exists(left_model_path) or not os.path.exists(right_model_path):
    st.error("Kidney models not found. Run `python setup` first to train them.")
    st.stop()

model_left = joblib.load(left_model_path)
model_right = joblib.load(right_model_path)

# -----------------------------
# Helper: load CSV by keyword
# -----------------------------
def load_csv_by_keyword(keyword):
    for f in os.listdir(data_folder):
        if keyword.lower() in f.lower() and f.lower().endswith(".csv"):
            return pd.read_csv(os.path.join(data_folder, f))
    st.warning(f"CSV for '{keyword}' not found in data folder.")
    return pd.DataFrame()

# -----------------------------
# Load referral data dynamically
# -----------------------------
referrals = load_csv_by_keyword("referral")
if referrals.empty:
    referrals = load_csv_by_keyword("opo")
if referrals.empty:
    st.error("No referral/OPO CSV found. Make sure ORCHID data is extracted properly.")
    st.stop()

referrals.columns = [c.strip() for c in referrals.columns]
if "patient_id" in referrals.columns:
    referrals["patient_id"] = referrals["patient_id"].astype(str)
else:
    st.error("Missing 'patient_id' column in referrals data.")
    st.stop()

# -----------------------------
# Column discovery
# -----------------------------
col_lower = {c.lower(): c for c in referrals.columns}

candidate_demographic_cols = []
for key in ["race", "ethnicity", "sex", "gender", "age", "blood_type"]:
    for c in referrals.columns:
        if key in c.lower():
            candidate_demographic_cols.append(c)

candidate_demographic_cols = list(dict.fromkeys(candidate_demographic_cols))

numeric_cols = referrals.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# Input sections
# -----------------------------
st.subheader("Basic Information")

basic_inputs = {}
if "patient_id" in referrals.columns:
    selected_patient_id = st.selectbox(
        "Select a patient ID (optional)",
        ["(custom)"] + referrals["patient_id"].tolist(),
    )
else:
    selected_patient_id = "(custom)"

if selected_patient_id != "(custom)":
    patient_row = referrals[referrals["patient_id"] == selected_patient_id].iloc[0]
    for col in candidate_demographic_cols:
        basic_inputs[col] = patient_row.get(col, None)
else:
    patient_row = None

for col in candidate_demographic_cols:
    if col in numeric_cols:
        col_min = float(referrals[col].min()) if pd.notnull(referrals[col].min()) else 0.0
        col_max = float(referrals[col].max()) if pd.notnull(referrals[col].max()) else 100.0
        default = float(basic_inputs.get(col, (col_min + col_max) / 2)) if basic_inputs.get(col) is not None else (col_min + col_max) / 2
        basic_inputs[col] = st.slider(col, min_value=col_min, max_value=col_max, value=default)
    else:
        options = sorted(referrals[col].dropna().astype(str).unique().tolist())
        default = str(basic_inputs.get(col, options[0] if options else ""))
        basic_inputs[col] = st.selectbox(col, options=options, index=options.index(default) if default in options else 0)

st.subheader("Advanced Information (Lab Stats)")

advanced_inputs = {}
feature_cols = []
if hasattr(model_left, "feature_names_in_"):
    feature_cols = list(model_left.feature_names_in_)

lab_like = [c for c in feature_cols if "score" in c.lower() or "lab" in c.lower()]

with st.expander("Edit lab-related features", expanded=True):
    if not lab_like:
        st.info("No explicit lab-score features detected in the model; showing numeric features instead.")
        lab_like = feature_cols

    for col in lab_like:
        default_val = 0.0
        if patient_row is not None and col in patient_row.index and pd.notnull(patient_row[col]):
            default_val = float(patient_row[col])
        advanced_inputs[col] = st.slider(col, min_value=0.0, max_value=1.0, value=float(default_val), step=0.01)

# -----------------------------
# Build feature vector
# -----------------------------
input_features = {}

if patient_row is not None:
    input_features.update(patient_row.to_dict())

input_features.update(basic_inputs)
input_features.update(advanced_inputs)

X = pd.DataFrame([input_features])

if feature_cols:
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

X = X.select_dtypes(include=[np.number]).fillna(0)

# -----------------------------
# Predict risk scores
# -----------------------------
left_risk = float(model_left.predict_proba(X)[:, 1][0])
right_risk = float(model_right.predict_proba(X)[:, 1][0])

risk_color = "green" if (left_risk + right_risk) / 2 < 0.33 else "orange" if (left_risk + right_risk) / 2 < 0.66 else "red"

st.subheader("Risk Scores")
col1, col2 = st.columns(2)
with col1:
    st.metric("Left Kidney Risk", f"{left_risk:.2f}")
with col2:
    st.metric("Right Kidney Risk", f"{right_risk:.2f}")

st.markdown(f"**Overall Risk Level:** :{risk_color}[{risk_color.upper()}]")

# -----------------------------
# Similar demographic scatterplot
# -----------------------------
st.subheader("Similar Demographic Scatterplot")

if numeric_cols:
    x_col = st.selectbox("X-axis", options=numeric_cols, index=0)
    y_col = st.selectbox("Y-axis", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    filtered = referrals.copy()
    for col in candidate_demographic_cols:
        if col in filtered.columns and col not in numeric_cols:
            val = basic_inputs.get(col)
            if val is not None and val != "":
                filtered = filtered[filtered[col].astype(str) == str(val)]

    fig, ax = plt.subplots()
    ax.scatter(filtered[x_col], filtered[y_col], alpha=0.5, label="Similar cohort")

    if patient_row is not None and x_col in patient_row.index and y_col in patient_row.index:
        ax.scatter([patient_row[x_col]], [patient_row[y_col]], color="red", s=80, label="Selected patient")
    else:
        ax.scatter([X.get(x_col, pd.Series([0])).iloc[0]], [X.get(y_col, pd.Series([0])).iloc[0]], color="red", s=80, label="Modeled patient")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No numeric columns available for scatterplot.")
