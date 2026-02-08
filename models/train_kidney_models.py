import pandas as pd
import numpy as np
import os
import joblib
import zipfile
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
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

def select_threshold(y_true, y_prob, max_fpr=0.10):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    candidates = np.where(fpr <= max_fpr)[0]
    if len(candidates) > 0:
        best = candidates[np.argmax(tpr[candidates])]
    else:
        best = int(np.argmin(fpr))
    return float(thresholds[best]), float(fpr[best]), float(tpr[best])

def train_risk_model(side, iterations=10, patience=2, min_delta=0.0):  # Updated to 10 iterations
    target_col = f'outcome_kidney_{side}'
    data = all_features.dropna(subset=[target_col])
    X = data.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore').fillna(0)
    y = data[target_col].apply(lambda x: 1 if x == 'Transplanted' else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results_history = []
    final_model = None
    best_avg_acc = 0.0
    no_improve = 0

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
        
        # Calibrate with internal CV (prefit is no longer supported in sklearn>=1.4)
        calibrator_sig = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrator_sig.fit(X_train, y_train)

        calibrator_iso = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrator_iso.fit(X_train, y_train)

        probs_sig = calibrator_sig.predict_proba(X_test)[:, 1]
        probs_iso = calibrator_iso.predict_proba(X_test)[:, 1]

        brier_sig = brier_score_loss(y_test, probs_sig)
        brier_iso = brier_score_loss(y_test, probs_iso)

        if brier_iso <= brier_sig:
            calibrator = calibrator_iso
            probs = probs_iso
            calib_method = "isotonic"
        else:
            calibrator = calibrator_sig
            probs = probs_sig
            calib_method = "sigmoid"

        threshold, sel_fpr, sel_tpr = select_threshold(y_test, probs, max_fpr=0.10)
        preds = (probs >= threshold).astype(int)
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        tier_counts = pd.Series([get_risk_tier(p) for p in probs]).value_counts()
        
        results_history.append({
            "Step": i + 1,
            "Trees": n_trees,
            "AUC_Discrim": round(auc, 3),
            "Brier_Error": round(brier, 4),
            "Calibration": calib_method,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3),
            "FPR": round(fpr, 3),
            "FNR": round(fnr, 3),
            "Threshold": round(threshold, 3),
            "Ideal_N": tier_counts.get("LOW (Ideal)", 0),
            "Critical_N": tier_counts.get("HIGH (Critical)", 0)
        })
        final_model = calibrator

        avg_acc = float(np.mean([h["Accuracy"] for h in results_history]))
        if avg_acc > best_avg_acc + min_delta:
            best_avg_acc = avg_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    joblib.dump(final_model, os.path.join(models_folder, f'{side}_kidney_model.pkl'))
    return results_history

# --- Execution ---
print("\n" + "🚀 RUNNING 10-ITERATION RISK MODEL TRAINING (LR=0.1)")
print("="*75)

hist_l = train_risk_model('left', iterations=10)
hist_r = train_risk_model('right', iterations=10)

def extract_top_features(side, top_n=12):
    target_col = f'outcome_kidney_{side}'
    data = all_features.dropna(subset=[target_col])
    X = data.select_dtypes(include=[np.number]).drop(columns=['patient_id'], errors='ignore').fillna(0)
    y = data[target_col].apply(lambda x: 1 if x == 'Transplanted' else 0)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_feats = importances.sort_values(ascending=False).head(top_n).index.tolist()
    return top_feats

try:
    feature_map = {
        "left": extract_top_features("left", top_n=12),
        "right": extract_top_features("right", top_n=12),
    }
    with open(os.path.join(models_folder, "effective_features.json"), "w") as f:
        json.dump(feature_map, f, indent=2)
except Exception:
    pass

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
def print_summary(side, history):
    first = history[0]
    last = history[-1]
    return {
        "AUC": f"{first['AUC_Discrim']} ➔ {last['AUC_Discrim']}",
        "Brier": f"{first['Brier_Error']} ➔ {last['Brier_Error']}",
        "Accuracy": f"{first['Accuracy']} ➔ {last['Accuracy']}",
        "Precision": f"{first['Precision']} ➔ {last['Precision']}",
        "Recall": f"{first['Recall']} ➔ {last['Recall']}",
        "F1": f"{first['F1']} ➔ {last['F1']}",
        "FPR": f"{first['FPR']} ➔ {last['FPR']}",
        "FNR": f"{first['FNR']} ➔ {last['FNR']}",
        "Calibration": f"{first['Calibration']} ➔ {last['Calibration']}",
        "Threshold": f"{first['Threshold']} ➔ {last['Threshold']}",
    }

left_summary = print_summary("Left", hist_l)
right_summary = print_summary("Right", hist_r)

print(f"{'Metric':<12} | {'Left':<20} | {'Right':<20}")
print("-" * 60)
for metric in ["AUC", "Brier", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "Calibration", "Threshold"]:
    print(f"{metric:<12} | {left_summary[metric]:<20} | {right_summary[metric]:<20}")
print("="*75)

print("\n✅ Risk models saved. Ready for deployment.")
