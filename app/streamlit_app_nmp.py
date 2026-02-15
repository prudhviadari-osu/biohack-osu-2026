# -----------------------------
# streamlit_app_nmp.py
# -----------------------------
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    raise RuntimeError("Run this app with: streamlit run app/streamlit_app_nmp.py")

st.set_page_config(page_title="NMPulse: NMP Homeostasis Model", layout="wide")
st.title("NMPulse: NMP Homeostasis Model")

# -----------------------------
# Paths
# -----------------------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(repo_root, "data")

def find_data_dir(base_dir, keywords, max_depth=3):
    for root, dirs, files in os.walk(base_dir):
        rel_depth = os.path.relpath(root, base_dir).count(os.sep)
        if rel_depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith(".csv") and any(k in f.lower() for k in keywords):
                return root
    return base_dir

data_folder = find_data_dir(data_root, ["referral", "opo"])

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_referrals(folder):
    for f in os.listdir(folder):
        if "referral" in f.lower() and f.lower().endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f), low_memory=False)
            return df
    for f in os.listdir(folder):
        if "opo" in f.lower() and f.lower().endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f), low_memory=False)
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_lab_events(folder, kind):
    fname = {
        "hemo": "HemoEvents.csv",
        "chem": "ChemistryEvents.csv",
        "abg": "ABGEvents.csv",
        "cbc": "CBCEvents.csv",
    }[kind]
    path = os.path.join(folder, fname)
    df = pd.read_csv(path, low_memory=False)
    if kind == "hemo":
        df = df.rename(columns={"measurement_name": "lab_name", "time_event_start": "time_event"})
    elif kind == "chem":
        df = df.rename(columns={"chem_name": "lab_name"})
    elif kind == "abg":
        df = df.rename(columns={"abg_name": "lab_name"})
    elif kind == "cbc":
        df = df.rename(columns={"cbc_name": "lab_name"})
    df["time_event"] = pd.to_datetime(df["time_event"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.dropna(subset=["lab_name", "value"])
    return df[["patient_id", "time_event", "lab_name", "value"]]

referrals = load_referrals(data_folder)
if referrals.empty:
    st.error("No referral/OPO CSV found. Make sure ORCHID data is extracted properly.")
    st.stop()

referrals.columns = [c.strip() for c in referrals.columns]
if "patient_id" not in referrals.columns:
    st.error("Missing 'patient_id' column in referrals data.")
    st.stop()
referrals["patient_id"] = referrals["patient_id"].astype(str)

candidate_demographic_cols = []
for key in ["race", "ethnicity", "sex", "gender", "age", "blood_type", "height", "weight", "bmi"]:
    for c in referrals.columns:
        if key in c.lower():
            candidate_demographic_cols.append(c)
candidate_demographic_cols = list(dict.fromkeys(candidate_demographic_cols))
numeric_cols = referrals.select_dtypes(include=[np.number]).columns.tolist()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Donor & Organ Context")
    selected_patient_id = st.selectbox(
        "Select patient ID",
        referrals["patient_id"].tolist(),
    )

    patient_row = referrals[referrals["patient_id"] == selected_patient_id].iloc[0]
    basic_inputs = {}
    for col in candidate_demographic_cols:
        basic_inputs[col] = patient_row.get(col, None)

    for col in candidate_demographic_cols:
        if col in numeric_cols:
            if col.lower() == "age":
                col_min = 0.0
                col_max = 90.0
            else:
                col_min = float(referrals[col].min()) if pd.notnull(referrals[col].min()) else 0.0
                col_max = float(referrals[col].max()) if pd.notnull(referrals[col].max()) else 100.0
            default = float(basic_inputs.get(col, (col_min + col_max) / 2)) if basic_inputs.get(col) is not None else (col_min + col_max) / 2
            basic_inputs[col] = st.slider(col, min_value=col_min, max_value=col_max, value=default)
        else:
            options = sorted(referrals[col].dropna().astype(str).unique().tolist())
            default = str(basic_inputs.get(col, options[0] if options else ""))
            basic_inputs[col] = st.selectbox(col, options=options, index=options.index(default) if default in options else 0)

    st.subheader("Lab Targeting")
    lab_kind = st.selectbox("Lab category", ["chem", "abg", "cbc", "hemo"], index=0)
    lab_df = load_lab_events(data_folder, lab_kind)
    lab_names = sorted(lab_df["lab_name"].dropna().unique().tolist())
    if not lab_names:
        st.error("No lab names found for this category.")
        st.stop()
    lab_name = st.selectbox("Lab name", lab_names)

    donor_rows_preview = lab_df[(lab_df["patient_id"] == selected_patient_id) & (lab_df["lab_name"] == lab_name)]
    max_window = max(2, len(donor_rows_preview))
    min_window = 1 if max_window <= 2 else 2
    rolling_n = st.slider(
        "Rolling window size (donor-specific)",
        min_value=min_window,
        max_value=max_window,
        value=max_window,
    )
    k_std = st.slider("Range width (k * std)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    age_tolerance = st.slider("Age tolerance (± years)", min_value=0, max_value=30, value=5)
    bmi_tolerance = 2.0

    def compute_velocity_default(df):
        df = df.dropna(subset=["time_event", "value"]).sort_values(["patient_id", "time_event"])
        velocities = []
        for _, grp in df.groupby("patient_id"):
            if len(grp) < 2:
                continue
            v_prev = grp["value"].shift(1)
            t_prev = grp["time_event"].shift(1)
            t_curr = grp["time_event"]
            hours = (t_curr - t_prev).dt.total_seconds() / 3600.0
            valid = hours > 0
            vel = (grp["value"] - v_prev) / hours
            velocities.extend(vel[valid].abs().dropna().tolist())
        if not velocities:
            return 0.5
        return float(np.percentile(velocities, 90))

    lab_vel_default = compute_velocity_default(
        lab_df[lab_df["lab_name"] == lab_name]
    )
    vel_threshold = st.slider(
        "Velocity threshold (abs change per hour)",
        min_value=0.0,
        max_value=max(10.0, lab_vel_default * 2),
        value=lab_vel_default,
        step=0.1,
    )
    
    with right_col:
        st.subheader("Organ Simulation & Reference")
    
        # Load and display the biohack.jpg image
        image_path = os.path.join(os.path.dirname(__file__), "images", "biohack.jpg")
    
        if os.path.exists(image_path):
            st.image(image_path, caption="Conceptual NMP System Design", use_container_width=True)
            st.info('''The conceptual design above illustrates a self-contained NMP transport unit. For future steps, we would try to integrate the homeostasis model’s logic directly into this hardware, allowing for automated monitoring and "at-a-glance" stability alerts during perfusion.''')
        else:
            st.error(f"Image not found at {image_path}. Please ensure biohack.jpg is in the app folder.")

        # Compute donor-specific stats and population stats for hybrid blending
        donor_rows = lab_df[(lab_df["patient_id"] == selected_patient_id) & (lab_df["lab_name"] == lab_name)].copy()
        donor_rows = donor_rows.sort_values("time_event")
        donor_rows = donor_rows.dropna(subset=["value"])

        if donor_rows.empty:
            st.warning("No donor-specific labs found for this selection. Falling back to population estimates.")

        recent = donor_rows.tail(rolling_n)
        donor_mean = recent["value"].mean() if not recent.empty else np.nan
        donor_std = recent["value"].std(ddof=0) if len(recent) >= 2 else np.nan
        donor_n = len(recent)

        # Velocity from last two points
        velocity = np.nan
        if len(recent) >= 2:
            v_prev, v_last = recent["value"].iloc[-2], recent["value"].iloc[-1]
            t_prev, t_last = recent["time_event"].iloc[-2], recent["time_event"].iloc[-1]
            if pd.notnull(t_prev) and pd.notnull(t_last):
                hours = max((t_last - t_prev).total_seconds() / 3600.0, 1e-6)
            else:
                hours = 1.0
            velocity = (v_last - v_prev) / hours

        # Population baseline
        pop = lab_df[lab_df["lab_name"] == lab_name].merge(referrals, on="patient_id", how="left")
        for col in candidate_demographic_cols:
            if col in pop.columns and col not in numeric_cols:
                val = basic_inputs.get(col)
                if val is not None and val != "":
                    pop = pop[pop[col].astype(str) == str(val)]
            elif col in pop.columns and col in numeric_cols and col.lower() == "age":
                val = basic_inputs.get(col)
                if val is not None:
                    pop = pop[(pop[col] >= float(val) - age_tolerance) & (pop[col] <= float(val) + age_tolerance)]
            elif col in pop.columns and col in numeric_cols and "bmi" in col.lower():
                val = basic_inputs.get(col)
                if val is not None:
                    pop = pop[(pop[col] >= float(val) - bmi_tolerance) & (pop[col] <= float(val) + bmi_tolerance)]

        pop_mean = pop["value"].mean() if not pop.empty else np.nan
        pop_std = pop["value"].std(ddof=0) if len(pop) >= 2 else np.nan

        # Hybrid blend
        if donor_n >= 2 and not np.isnan(pop_mean):
            w = min(1.0, donor_n / float(rolling_n))
            mean = (w * donor_mean) + ((1.0 - w) * pop_mean)
            std = (w * (donor_std if not np.isnan(donor_std) else 0.0)) + ((1.0 - w) * (pop_std if not np.isnan(pop_std) else 0.0))
        elif donor_n >= 2:
            mean, std = donor_mean, donor_std
        else:
            mean, std = pop_mean, pop_std

        if np.isnan(std) or std == 0:
            std = max(std if not np.isnan(std) else 0.0, 1e-6)

        lower = mean - k_std * std if not np.isnan(mean) else np.nan
        upper = mean + k_std * std if not np.isnan(mean) else np.nan
        current = donor_rows["value"].iloc[-1] if not donor_rows.empty else np.nan

        status = "unknown"
        if not np.isnan(current) and not np.isnan(lower) and not np.isnan(upper):
            if current < lower:
                status = "low"
            elif current > upper:
                status = "high"
            else:
                status = "ok"
            if status == "ok" and not np.isnan(velocity) and abs(velocity) > vel_threshold:
                status = "unstable"

            #Risk flags and suggestions
            st.subheader("Risk Flags")
            st.markdown(f"**Status:** `{status}`")
            suggestion = None
            if status == "unstable":
                st.warning(
                    f"Attention: rapid change detected (|velocity| = {abs(velocity):.3g} per hour "
                    f"exceeds threshold {vel_threshold:.3g}). Review perfusion settings and trends."
                )
                suggestion = f"Suggested action: verify recent changes affecting `{lab_name}` and re-check stability before adjustment."
            elif status in {"low", "high"}:
                st.warning(
                    f"Attention: value is {status} relative to target range "
                    f"({lower:.3g}–{upper:.3g}). Consider reviewing perfusion targets."
                )
                if status == "low":
                    suggestion = f"Suggested action: consider increasing support for `{lab_name}` if clinically appropriate."
                else:
                    suggestion = f"Suggested action: consider reducing drivers of `{lab_name}` if clinically appropriate."

            if suggestion:
                st.info(suggestion)

            st.markdown(f"**Velocity (per hour):** `{velocity:.4g}`" if not np.isnan(velocity) else "**Velocity (per hour):** `n/a`")

            #Target ranges and stats used for comparsion (accountability)
            st.subheader("Target Ranges")
            st.markdown(f"**Lab:** `{lab_name}`")
            st.markdown(f"**Current value:** `{current:.4g}`" if not np.isnan(current) else "**Current value:** `n/a`")
            st.markdown(f"**Target range:** `{lower:.4g} – {upper:.4g}`" if not np.isnan(lower) else "**Target range:** `n/a`")

            st.divider()
            st.markdown("**Donor-specific stats (rolling window)**")
            st.write(
                {
                    "n_points": donor_n,
                    "mean": None if np.isnan(donor_mean) else float(donor_mean),
                    "std": None if np.isnan(donor_std) else float(donor_std),
                }
            )

            st.markdown("**Population stats (demographic filtered)**")
            st.write(
                {
                    "n_points": int(len(pop)),
                    "mean": None if np.isnan(pop_mean) else float(pop_mean),
                    "std": None if np.isnan(pop_std) else float(pop_std),
                }
            )

# Display recent donor measurements for transparency and context (accountability)
with left_col:
    if not recent.empty:
            st.markdown("**Recent donor measurements**")
            st.dataframe(recent[["time_event", "value"]].tail(rolling_n), width=650)
