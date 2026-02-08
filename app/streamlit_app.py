# -----------------------------
# streamlit_app.py
# -----------------------------
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import plotly.express as px
import json

if get_script_run_ctx() is None:
    raise RuntimeError("Run this app with: streamlit run app/streamlit_app.py")

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

effective_feature_path = os.path.join(models_folder, "effective_features.json")
effective_features = []
if os.path.exists(effective_feature_path):
    with open(effective_feature_path, "r") as f:
        data = json.load(f)
    effective_features = sorted(set(data.get("left", []) + data.get("right", [])))

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

basic_keys = ["race", "gender", "sex", "age", "blood_type", "height", "weight", "bmi"]
candidate_demographic_cols = []
for key in basic_keys:
    for c in referrals.columns:
        if key in c.lower():
            candidate_demographic_cols.append(c)

candidate_demographic_cols = list(dict.fromkeys(candidate_demographic_cols))

numeric_cols = referrals.select_dtypes(include=[np.number]).columns.tolist()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Organ Info (Basic)")

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
            if col.lower() == "age":
                col_max = 90.0
            else:
                col_max = float(referrals[col].max()) if pd.notnull(referrals[col].max()) else 100.0
            default = float(basic_inputs.get(col, (col_min + col_max) / 2)) if basic_inputs.get(col) is not None else (col_min + col_max) / 2
            basic_inputs[col] = st.slider(col, min_value=col_min, max_value=col_max, value=default)
        else:
            options = sorted(referrals[col].dropna().astype(str).unique().tolist())
            default = str(basic_inputs.get(col, options[0] if options else ""))
            basic_inputs[col] = st.selectbox(col, options=options, index=options.index(default) if default in options else 0)

    st.subheader("Organ Info (Advanced)")

    advanced_inputs = {}
    feature_cols = []
    if hasattr(model_left, "feature_names_in_"):
        feature_cols = list(model_left.feature_names_in_)

    basic_lower = {c.lower() for c in candidate_demographic_cols}
    if effective_features:
        lab_like = [c for c in effective_features if c in feature_cols and c.lower() not in basic_lower]
    else:
        lab_like = [c for c in feature_cols if ("score" in c.lower() or "lab" in c.lower()) and c.lower() not in basic_lower]

    with st.expander("Edit lab-related features", expanded=True):
        if not lab_like:
            st.info("No explicit lab-score features detected in the model; showing numeric features instead.")
            lab_like = feature_cols

        # Build feature stats from cohort data to normalize sliders
        feature_stats = {}
        if feature_cols:
            cohort_stats = referrals.reindex(columns=feature_cols)
            cohort_stats = cohort_stats.select_dtypes(include=[np.number])
            for col in lab_like:
                if col in cohort_stats.columns:
                    series = pd.to_numeric(cohort_stats[col], errors="coerce")
                    if series.notna().any():
                        feature_stats[col] = {
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "median": float(series.median()),
                        }

        for col in lab_like:
            stats = feature_stats.get(col)
            if stats:
                col_min = 0.0
                col_max = stats["max"]
                default_val = max(0.0, stats["median"])
            else:
                col_min = 0.0
                col_max = 1.0
                default_val = 0.0

            if patient_row is not None and col in patient_row.index and pd.notnull(patient_row[col]):
                default_val = float(patient_row[col])

            if col_min >= col_max:
                col_max = col_min + 1.0

            advanced_inputs[col] = st.slider(
                col,
                min_value=float(col_min),
                max_value=float(col_max),
                value=float(default_val),
            )

            if stats and (advanced_inputs[col] < stats["min"] or advanced_inputs[col] > stats["max"]):
                st.warning(f"{col} is outside observed training range ({stats['min']:.3g}–{stats['max']:.3g}).")

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
        X = X.reindex(columns=feature_cols, fill_value=0)

    X = X.select_dtypes(include=[np.number]).fillna(0)

    #
    # Risk score definition:
    #   risk = P(not transplantable) = 1 - P(transplanted)
    #
    # Model probabilities are for the positive class (Transplanted).
    # -----------------------------
    left_prob_transplanted = float(model_left.predict_proba(X)[:, 1][0])
    right_prob_transplanted = float(model_right.predict_proba(X)[:, 1][0])
    left_risk = 1.0 - left_prob_transplanted
    right_risk = 1.0 - right_prob_transplanted

    avg_risk = (left_risk + right_risk) / 2
    risk_color = "green" if avg_risk < 0.33 else "orange" if avg_risk < 0.66 else "red"

    st.subheader("Risk Scores")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Left Kidney Risk", f"{left_risk:.2%}")
    with col2:
        st.metric("Right Kidney Risk", f"{right_risk:.2%}")

    st.markdown(f"**Overall Risk Level:** :{risk_color}[{risk_color.upper()}]")

with right_col:
    st.subheader("Similar Demographic Scatterplot")

    # Build a cohort-level risk score for each patient row
    cohort = referrals.copy()
    if feature_cols:
        cohort_X = cohort.reindex(columns=feature_cols, fill_value=0)
    else:
        cohort_X = cohort.copy()

    cohort_X = cohort_X.select_dtypes(include=[np.number]).fillna(0)

    if len(cohort_X) > 0:
        left_prob_all = model_left.predict_proba(cohort_X)[:, 1]
        right_prob_all = model_right.predict_proba(cohort_X)[:, 1]
        cohort["risk_left"] = 1.0 - left_prob_all
        cohort["risk_right"] = 1.0 - right_prob_all
        cohort["risk_avg"] = (cohort["risk_left"] + cohort["risk_right"]) / 2
    else:
        cohort["risk_left"] = np.nan
        cohort["risk_right"] = np.nan
        cohort["risk_avg"] = np.nan

    if numeric_cols:
        x_col = st.selectbox("X-axis", options=numeric_cols, index=0, key="x_axis")
        risk_view = st.radio("Risk axis", ["Average", "Left", "Right"], horizontal=True)
        age_tolerance = st.slider("Age tolerance (± years)", min_value=0, max_value=30, value=5)
        bmi_tolerance = 2.0

        if risk_view == "Left":
            y_col = "risk_left"
            y_label = "Left Kidney Risk"
            patient_y = left_risk
        elif risk_view == "Right":
            y_col = "risk_right"
            y_label = "Right Kidney Risk"
            patient_y = right_risk
        else:
            y_col = "risk_avg"
            y_label = "Average Risk"
            patient_y = avg_risk

        filtered = cohort.copy()
        for col in candidate_demographic_cols:
            if col in filtered.columns and col not in numeric_cols:
                val = basic_inputs.get(col)
                if val is not None and val != "":
                    filtered = filtered[filtered[col].astype(str) == str(val)]
            elif col in filtered.columns and col in numeric_cols and col.lower() == "age":
                val = basic_inputs.get(col)
                if val is not None:
                    filtered = filtered[
                        (filtered[col] >= float(val) - age_tolerance)
                        & (filtered[col] <= float(val) + age_tolerance)
                    ]
            elif col in filtered.columns and col in numeric_cols and "bmi" in col.lower():
                val = basic_inputs.get(col)
                if val is not None:
                    filtered = filtered[
                        (filtered[col] >= float(val) - bmi_tolerance)
                        & (filtered[col] <= float(val) + bmi_tolerance)
                    ]

    hover_cols = ["patient_id"] + [c for c in candidate_demographic_cols if c in filtered.columns]
    hover_cols = list(dict.fromkeys(hover_cols))

    fig = px.scatter(
        filtered,
        x=x_col,
        y=y_col,
        hover_data=hover_cols + [y_col],
        opacity=0.6,
        labels={x_col: x_col, y_col: y_label},
        title="Similar cohort",
        height=350,
    )

    if patient_row is not None and x_col in patient_row.index:
        modeled_x = patient_row[x_col]
        label = "Selected patient"
    else:
        modeled_x = basic_inputs.get(x_col, X.get(x_col, pd.Series([0])).iloc[0])
        label = "Modeled patient"

    fig.add_scatter(
        x=[modeled_x],
        y=[patient_y],
        mode="markers",
        marker=dict(color="red", size=10),
        name=label,
        hoverinfo="skip",
    )

    st.plotly_chart(fig, width=650)
