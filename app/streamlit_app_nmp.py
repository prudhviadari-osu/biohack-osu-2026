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

st.set_page_config(page_title="NMP Homeostasis Dashboard", layout="wide")
st.title("NMP Homeostasis Model")

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
    st.subheader("Organ Simulation")

    nmp_html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>NMP Kidney Visualization</title>
<style>
  body { margin: 0; background: #0b0d10; overflow: hidden; }
  canvas { display: block; }
  #label {
    position: absolute;
    padding: 6px 8px;
    background: rgba(15,15,15,0.9);
    color: #eee;
    border-radius: 8px;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    pointer-events: none;
    opacity: 0;
    transform: translate(-50%, -120%);
    transition: opacity 0.15s ease;
  }
  #stage { position: relative; width: 100%; height: 100%; }
</style>
</head>
<body>
<div id="stage"></div>
<div id="label">Label</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
// Scene
const stage = document.getElementById('stage');
const label = document.getElementById('label');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
scene.fog = new THREE.Fog(0x0b0d10, 10, 22);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2.2, 7.5);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);

// Lights
const key = new THREE.DirectionalLight(0xffffff, 1.0);
key.position.set(5, 6, 4);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
scene.add(key);

const fill = new THREE.DirectionalLight(0xffb3b3, 0.4);
fill.position.set(-5, -3, 2);
scene.add(fill);

const ambient = new THREE.AmbientLight(0x404040, 0.7);
scene.add(ambient);

const rim = new THREE.PointLight(0xff8888, 0.5, 15);
rim.position.set(-4, 2, -4);
scene.add(rim);

// Enclosure (rounded-ish box using scaled box + bevel illusion)
const enclosureGeo = new THREE.BoxGeometry(8, 4.5, 4);
const enclosureMat = new THREE.MeshStandardMaterial({ color: 0x0a0b0d, metalness: 0.6, roughness: 0.6 });
const enclosure = new THREE.Mesh(enclosureGeo, enclosureMat);
enclosure.castShadow = true;
enclosure.receiveShadow = true;
scene.add(enclosure);

// Glass front
const glassGeo = new THREE.BoxGeometry(7.2, 3.8, 0.15);
const glassMat = new THREE.MeshPhysicalMaterial({
  color: 0x88aaff,
  transmission: 0.6,
  thickness: 0.4,
  roughness: 0.05,
  metalness: 0.0,
  clearcoat: 1.0,
  clearcoatRoughness: 0.05,
  opacity: 0.35,
  transparent: true
});
const glass = new THREE.Mesh(glassGeo, glassMat);
glass.position.z = 2.05;
scene.add(glass);

// Kidney (lathe + noise)
const pts = [];
for (let i = 0; i < 24; i++) {
  const x = Math.sin(i * 0.26) * 0.65 + 0.45;
  const y = (i - 12) * 0.07;
  pts.push(new THREE.Vector2(x, y));
}
const kidneyGeo = new THREE.LatheGeometry(pts, 80);
const kidneyMat = new THREE.MeshStandardMaterial({
  color: 0xd26b6b,
  roughness: 0.5,
  metalness: 0.1,
  emissive: 0x2a0000,
  emissiveIntensity: 0.25
});
const kidney = new THREE.Mesh(kidneyGeo, kidneyMat);
kidney.rotation.z = Math.PI / 2;
kidney.position.set(0, 0.2, 0.4);
kidney.castShadow = true;
kidney.userData.label = 'Organ Perfusion';
scene.add(kidney);

// Tubing loop
const curve = new THREE.CatmullRomCurve3([
  new THREE.Vector3(-2.6, 0.8, 0.8),
  new THREE.Vector3(-1.4, 1.2, 0.2),
  new THREE.Vector3(0.4, 1.1, 0.0),
  new THREE.Vector3(1.8, 0.6, 0.4),
  new THREE.Vector3(2.5, -0.6, 0.9),
  new THREE.Vector3(0.9, -1.2, 0.4),
  new THREE.Vector3(-1.8, -0.9, 0.8),
], true);
const tubeGeo = new THREE.TubeGeometry(curve, 200, 0.08, 14, true);
const tubeMat = new THREE.MeshPhysicalMaterial({
  color: 0xff6b6b,
  transmission: 0.6,
  thickness: 0.2,
  roughness: 0.2,
  metalness: 0.0,
  emissive: 0x3a0000,
  emissiveIntensity: 0.6
});
const tube = new THREE.Mesh(tubeGeo, tubeMat);
tube.userData.label = 'Flow Circuit';
scene.add(tube);

// Flow particles
const particles = [];
const pGeo = new THREE.SphereGeometry(0.03, 8, 8);
const pMat = new THREE.MeshStandardMaterial({ color: 0xffb3b3, emissive: 0x550000 });
for (let i = 0; i < 90; i++) {
  const p = new THREE.Mesh(pGeo, pMat);
  p.userData.t = i / 90;
  particles.push(p);
  scene.add(p);
}

// Reservoir
const reservoirGeo = new THREE.CylinderGeometry(0.5, 0.5, 1.3, 24);
const reservoirMat = new THREE.MeshPhysicalMaterial({
  color: 0x4a0f15,
  transmission: 0.2,
  roughness: 0.4,
  metalness: 0.1,
  emissive: 0x2a0000,
  emissiveIntensity: 0.6
});
const reservoir = new THREE.Mesh(reservoirGeo, reservoirMat);
reservoir.position.set(-3.0, 0.6, 0.9);
reservoir.userData.label = 'Reservoir';
scene.add(reservoir);

// Gas canister
const gasGeo = new THREE.CylinderGeometry(0.35, 0.35, 1.5, 24);
const gasMat = new THREE.MeshStandardMaterial({ color: 0x59667a, metalness: 0.8, roughness: 0.3 });
const gas = new THREE.Mesh(gasGeo, gasMat);
gas.position.set(3.0, 0.8, -0.6);
gas.userData.label = 'Oxygenation';
scene.add(gas);

// Pump
const pumpGeo = new THREE.TorusGeometry(0.35, 0.12, 16, 32);
const pumpMat = new THREE.MeshStandardMaterial({ color: 0x2b313b, metalness: 0.6, roughness: 0.4 });
const pump = new THREE.Mesh(pumpGeo, pumpMat);
pump.position.set(1.2, -1.3, 0.9);
pump.userData.label = 'Pump / Flow Control';
scene.add(pump);

// Hover labels
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
window.addEventListener('mousemove', (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects([kidney, tube, reservoir, gas, pump], false);
  if (hits.length > 0) {
    label.textContent = hits[0].object.userData.label || 'Component';
    label.style.left = (event.clientX - rect.left) + 'px';
    label.style.top = (event.clientY - rect.top) + 'px';
    label.style.opacity = 1;
  } else {
    label.style.opacity = 0;
  }
});

// Animation
let pulse = 0;
function animate() {
  requestAnimationFrame(animate);
  pulse += 0.02;
  const scale = 1 + Math.sin(pulse) * 0.03;
  kidney.scale.set(scale, scale, scale);
  kidney.rotation.y += 0.003;

  const health = (Math.sin(pulse * 0.4) + 1) / 2;
  kidney.material.color.setHSL(0.0, 0.55, 0.42 + health * 0.12);
  kidney.material.emissiveIntensity = 0.2 + health * 0.25;

  for (const p of particles) {
    p.userData.t = (p.userData.t + 0.002) % 1;
    const v = curve.getPointAt(p.userData.t);
    p.position.copy(v);
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""
    components.html(nmp_html, height=500)

    st.subheader("Target Ranges & Risk Flags")

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

    st.markdown(f"**Lab:** `{lab_name}`")
    st.markdown(f"**Current value:** `{current:.4g}`" if not np.isnan(current) else "**Current value:** `n/a`")
    st.markdown(f"**Target range:** `{lower:.4g} – {upper:.4g}`" if not np.isnan(lower) else "**Target range:** `n/a`")
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

    if not recent.empty:
        st.markdown("**Recent donor measurements**")
        st.dataframe(recent[["time_event", "value"]].tail(rolling_n), width=650)
