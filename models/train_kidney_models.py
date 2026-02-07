# -----------------------------
# train_kidney_models.py
# -----------------------------
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn / pandas warnings

# -----------------------------
# Paths
# -----------------------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_CSVS = [
    "HemoEvents.csv",
    "ChemistryEvents.csv",
    "FluidBalanceEvents.csv",
    "ABGEvents.csv",
    "CBCEvents.csv",
    "OPOReferrals.csv",
]

def find_data_dir(base_dir, required_files, max_depth=2):
    def has_required_files(dir_path):
        return all(os.path.isfile(os.path.join(dir_path, f)) for f in required_files)
    if has_required_files(base_dir):
        return base_dir
    for root, dirs, _ in os.walk(base_dir):
        rel_depth = os.path.relpath(root, base_dir).count(os.sep)
        if rel_depth > max_depth:
            dirs[:] = []
            continue
        if has_required_files(root):
            return root
    return base_dir

data_folder = find_data_dir(os.path.join(repo_root, "data"), REQUIRED_CSVS)
models_folder = os.path.join(repo_root, "models")
os.makedirs(models_folder, exist_ok=True)

# -----------------------------
# Feature aggregation functions
# -----------------------------
def aggregate_hemo(hemo_df):
    df = hemo_df[['patient_id','measurement_name','measurement_type','value']].copy()
    df_pivot = df.pivot_table(index='patient_id', columns=['measurement_name','measurement_type'], values='value', aggfunc='mean')
    df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
    def compute_hemo_score(row):
        map_avg = row.get('MAP_Average', np.nan)
        hr_max = row.get('HeartRate_High', np.nan)
        hr_min = row.get('HeartRate_Low', np.nan)
        hr_var = hr_max - hr_min if pd.notnull(hr_max) and pd.notnull(hr_min) else 0
        score = 0
        if pd.notnull(map_avg) and map_avg < 65: score += 1
        if hr_var > 40: score += 1
        return score / 2
    df_pivot['hemo_score'] = df_pivot.apply(compute_hemo_score, axis=1)
    return df_pivot[['hemo_score']].reset_index()

def aggregate_chem(chem_df):
    df_pivot = chem_df.pivot_table(index='patient_id', columns='chem_name', values='value', aggfunc='mean')
    def compute_chem_score(row):
        score = 0
        creat = row.get('Creatinine', 0)
        na = row.get('Sodium', 140)
        k = row.get('Potassium', 4)
        if creat > 2.5: score += 1
        elif creat > 1.5: score += 0.5
        if na < 135 or na > 145: score += 1
        if k < 3.5 or k > 5.5: score += 1
        return min(score/3,1)
    df_pivot['chem_score'] = df_pivot.apply(compute_chem_score, axis=1)
    return df_pivot[['chem_score']].reset_index()

def aggregate_fluid(fluid_df):
    urine = fluid_df[fluid_df['fluid_type']=='Output'].groupby('patient_id')['amount'].sum().reset_index()
    def compute_fluid_score(row):
        urine_out = row['amount']
        if urine_out < 500: return 1
        elif urine_out < 1000: return 0.5
        else: return 0
    urine['fluid_score'] = urine.apply(compute_fluid_score, axis=1)
    return urine[['patient_id','fluid_score']]

def aggregate_abg(abg_df):
    df_pivot = abg_df.pivot_table(index='patient_id', columns='abg_name', values='value', aggfunc='mean')
    def compute_abg_score(row):
        ph = row.get('pH', 7.4)
        po2 = row.get('pO2', 90)
        score = 0
        if ph < 7.35 or ph > 7.45: score += 1
        if po2 < 80: score += 1
        return score / 2
    df_pivot['abg_score'] = df_pivot.apply(compute_abg_score, axis=1)
    return df_pivot[['abg_score']].reset_index()

def aggregate_cbc(cbc_df):
    df_pivot = cbc_df.pivot_table(index='patient_id', columns='cbc_name', values='value', aggfunc='mean')
    def compute_cbc_score(row):
        wbc = row.get('WBC', 7)
        return 0 if 4 <= wbc <= 11 else 1
    df_pivot['cbc_score'] = df_pivot.apply(compute_cbc_score, axis=1)
    return df_pivot[['cbc_score']].reset_index()

# -----------------------------
# Load CSVs
# -----------------------------
hemo_df = pd.read_csv(os.path.join(data_folder,'HemoEvents.csv'))
chem_df = pd.read_csv(os.path.join(data_folder,'ChemistryEvents.csv'))
fluid_df = pd.read_csv(os.path.join(data_folder,'FluidBalanceEvents.csv'))
abg_df = pd.read_csv(os.path.join(data_folder,'ABGEvents.csv'))
cbc_df = pd.read_csv(os.path.join(data_folder,'CBCEvents.csv'))
referrals = pd.read_csv(os.path.join(data_folder,'OPOReferrals.csv'))

# -----------------------------
# Aggregate features
# -----------------------------
hemo_scores = aggregate_hemo(hemo_df)
chem_scores = aggregate_chem(chem_df)
fluid_scores = aggregate_fluid(fluid_df)
abg_scores = aggregate_abg(abg_df)
cbc_scores = aggregate_cbc(cbc_df)

all_features = referrals.copy()
all_features = all_features.merge(hemo_scores, on='patient_id', how='left') \
                           .merge(chem_scores, on='patient_id', how='left') \
                           .merge(fluid_scores, on='patient_id', how='left') \
                           .merge(abg_scores, on='patient_id', how='left') \
                           .merge(cbc_scores, on='patient_id', how='left')

# -----------------------------
# Filter out very old patients (age > 89) if an age column exists
# -----------------------------
age_col = None
for c in all_features.columns:
    if c.lower() == "age":
        age_col = c
        break
if age_col is None:
    for c in all_features.columns:
        if "age" in c.lower():
            age_col = c
            break

if age_col is not None:
    all_features[age_col] = pd.to_numeric(all_features[age_col], errors="coerce")
    all_features = all_features[all_features[age_col].isna() | (all_features[age_col] <= 90)]

# -----------------------------
# Train left kidney
# -----------------------------
left_data = all_features.dropna(subset=['outcome_kidney_left'])
left_data['patient_id'] = left_data['patient_id'].astype(str)
X_left = left_data.drop(columns=['patient_id','outcome_kidney_left','outcome_kidney_right'], errors='ignore')
X_left = X_left.select_dtypes(include=[np.number])
y_left = left_data['outcome_kidney_left'].apply(lambda x: 1 if x=='Transplanted' else 0)

model_left = RandomForestClassifier(n_estimators=100, random_state=42)
model_left.fit(X_left, y_left)
joblib.dump(model_left, os.path.join(models_folder,'left_kidney_model.pkl'))

# -----------------------------
# Train right kidney
# -----------------------------
right_data = all_features.dropna(subset=['outcome_kidney_right'])
right_data['patient_id'] = right_data['patient_id'].astype(str)
X_right = right_data.drop(columns=['patient_id','outcome_kidney_left','outcome_kidney_right'], errors='ignore')
X_right = X_right.select_dtypes(include=[np.number])
y_right = right_data['outcome_kidney_right'].apply(lambda x: 1 if x=='Transplanted' else 0)

model_right = RandomForestClassifier(n_estimators=100, random_state=42)
model_right.fit(X_right, y_right)
joblib.dump(model_right, os.path.join(models_folder,'right_kidney_model.pkl'))

print("✅ Left & Right kidney models trained and saved successfully!")
