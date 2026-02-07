import pandas as pd
import numpy as np
import os
import joblib
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# --- Paths & Data Loading ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(repo_root, "data")
models_folder = os.path.join(repo_root, "models")
os.makedirs(models_folder, exist_ok=True)

REQUIRED_FILES = {"OPOReferrals.csv", "HemoEvents.csv", "ChemistryEvents.csv", "ABGEvents.csv", "CBCEvents.csv", "FluidBalanceEvents.csv"}

def _find_data_dir(search_root):
    for root, _, files in os.walk(search_root):
        if REQUIRED_FILES.issubset(set(files)): return root
    return None

def _ensure_data_dir(base_data_folder):
    data_dir = _find_data_dir(base_data_folder)
    if data_dir: return data_dir
    zip_candidates = [os.path.join(base_data_folder, f) for f in os.listdir(base_data_folder) if f.lower().endswith(".zip")]
    if not zip_candidates: raise FileNotFoundError("Dataset zip not found.")
    extract_root = os.path.join(base_data_folder, "extracted_data")
    os.makedirs(extract_root, exist_ok=True)
    with zipfile.ZipFile(zip_candidates[0], "r") as zf: zf.extractall(extract_root)
    return _find_data_dir(extract_root)

def aggregate_raw(df, value_col, name_col):
    pivot = df.pivot_table(index='patient_id', columns=name_col, values=value_col, aggfunc=['mean', 'min', 'max'])
    pivot.columns = [f"{c[1]}_{c[0]}" for c in pivot.columns]
    return pivot.reset_index()

# --- Load and Preprocess Data ---
data_dir = _ensure_data_dir(data_folder)
referrals = pd.read_csv(os.path.join(data_dir, 'OPOReferrals.csv'))
hemo = aggregate_raw(pd.read_csv(os.path.join(data_dir, 'HemoEvents.csv')), 'value', 'measurement_name')
chem = aggregate_raw(pd.read_csv(os.path.join(data_dir, 'ChemistryEvents.csv')), 'value', 'chem_name')
abg = aggregate_raw(pd.read_csv(os.path.join(data_dir, 'ABGEvents.csv')), 'value', 'abg_name')
cbc = aggregate_raw(pd.read_csv(os.path.join(data_dir, 'CBCEvents.csv')), 'value', 'cbc_name')
fluid = pd.read_csv(os.path.join(data_dir, 'FluidBalanceEvents.csv'))
urine = fluid[fluid['fluid_type']=='Output'].groupby('patient_id')['amount'].sum().reset_index()

all_features = referrals.copy()
for df_raw in [hemo, chem, abg, cbc, urine]:
    all_features = all_features.merge(df_raw, on='patient_id', how='left')

# --- Risk Tiering Logic ---
def get_risk_tier(score):
    risk_score = 1 - score 
    if risk_score < 0.3: return "LOW (Ideal)"
    if risk_score < 0.7: return "MODERATE"
    return "HIGH (Critical)"

def train_risk_model(side, iterations=10):  # Updated to 10 iterations
    target_col = f'outcome_kidney_{side}'
    data = all_features.dropna(subset=[target_col])
    X = data.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore').fillna(0)
    y = data[target_col].apply(lambda x: 1 if x == 'Transplanted' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results_history = []
    final_model = None

    for i in range(iterations):
        n_trees = (i + 1) * 25
        model = xgb.XGBClassifier(
            n_estimators=n_trees,
            learning_rate=0.075,  
            max_depth=4,
            min_child_weight=2,
            random_state=42,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        tier_counts = pd.Series([get_risk_tier(p) for p in probs]).value_counts()
        
        results_history.append({
            "Step": i + 1,
            "Trees": n_trees,
            "AUC_Discrim": round(auc, 3),
            "Brier_Error": round(brier, 4),
            "Ideal_N": tier_counts.get("LOW (Ideal)", 0),
            "Critical_N": tier_counts.get("HIGH (Critical)", 0)
        })
        final_model = model

    joblib.dump(final_model, os.path.join(models_folder, f'{side}_kidney_model.pkl'))
    return results_history

# --- Execution ---
print("\n" + "🚀 RUNNING 10-ITERATION RISK MODEL TRAINING (LR=0.1)")
print("="*75)

hist_l = train_risk_model('left', iterations=10)
hist_r = train_risk_model('right', iterations=10)

def print_performance_table(side, history):
    print(f"\n📈 {side.upper()} KIDNEY: RISK CALIBRATION REPORT")
    print("-" * 75)
    df = pd.DataFrame(history)
    print(df.to_string(index=False))

print_performance_table('left', hist_l)
print_performance_table('right', hist_r)

# --- Summary Comparison ---
print("\n" + "="*75)
print(f"{'10-STEP IMPROVEMENT SUMMARY':^75}")
print("-" * 75)
print(f"Left Kidney AUC:  Initial {hist_l[0]['AUC_Discrim']} ➔ Final {hist_l[-1]['AUC_Discrim']}")
print(f"Right Kidney AUC: Initial {hist_r[0]['AUC_Discrim']} ➔ Final {hist_r[-1]['AUC_Discrim']}")
print("="*75)

print("\n✅ Risk models saved. Ready for deployment.")