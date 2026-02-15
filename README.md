# BioHack OSU 2026: Kidney Risk + NMP Homeostasis Dashboard

A hackathon project for organ perfusion decision support built with Python, XGBoost, and Streamlit.

This repository includes two interactive apps:

- **Kidney Pre-Perfusion Risk Predictor**: estimates left/right kidney non-transplant risk from donor demographics and lab-derived features.
- **NMPulse Homeostasis Model**: tracks donor organ lab stability, generates target ranges, and flags unstable trends during normothermic machine perfusion (NMP) workflows for ease-of-use during organ transportation and maintenance without having to worry about cold ischemia
and the complications of real-world NMP applications that require 24/7 monitoring with trained professionals.

## Why this project

Kidney utilization decisions are time-sensitive and data-heavy. This project explores how ML + transparent dashboards can help teams:

- surface transplantability risk earlier right out of the donor with varying levels of data present,
- compare a donor against demographically similar cohorts for basic guidelines to inform urgent decisions,
- and monitor lab trajectory stability rather than single-point values to create thresholds normalized to the unique donor organ over time.

## Key Features

- Iterative XGBoost model training for **left** and **right** kidney outcomes.
- Probability calibration (`sigmoid` / `isotonic`) with metric tracking.
- Risk scoring shown as:
  - `risk = 1 - P(transplanted)`
- Streamlit UI with:
  - patient-level controls,
  - advanced lab-related feature controls,
  - cohort scatterplot with selected patient overlay,
  - color-coded overall risk level.
- NMP monitoring view with:
  - rolling donor-specific lab statistics,
  - demographic-filtered population baselines,
  - hybrid target range blending,
  - velocity-based instability alerts,
  - suggested action prompts.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- Streamlit
- Plotly
- joblib

## Repository Structure

```text
biohack-osu-2026/
├── app/
│   ├── streamlit_app_nmp.py              # NMP homeostasis dashboard
│   ├── streamlit_app.py                  # Kidney risk dashboard
│   └── images/biohack.jpg
├── data/
│   └── ORCHID_data.zip               # expected raw dataset zip
├── models/
│   ├── train_kidney_models.py        # training + calibration pipeline
│   ├── left_kidney_model.pkl         # generated artifact
│   ├── right_kidney_model.pkl        # generated artifact
│   └── effective_features.json       # generated top features
├── setup                             # one-command environment + training script
└── README.md
```

## Data Requirements

If not already present in the data file, place the ORCHID dataset zip file in `data/` (example: `data/ORCHID_data.zip`) and ensure it is named ORCHID_data.zip (case-sensitive).

The training script extracts these CSVs from the .zip file if done right.

- `OPOReferrals.csv`
- `HemoEvents.csv`
- `ChemistryEvents.csv`
- `ABGEvents.csv`
- `CBCEvents.csv`
- `FluidBalanceEvents.csv`

## Quick Start

### 1) Run setup and training

From the project root:

```bash
python setup
```

What this does:

- creates `venv/` if needed,
- installs dependencies,
- extracts data zip in `data/` (if present),
- trains both kidney models,
- writes model artifacts to `models/`.

### 2) Launch the kidney risk dashboard

```bash
streamlit run app/streamlit_app.py
```

### 3) Launch the NMP homeostasis dashboard

```bash
streamlit run app/streamlit_app_nmp.py
```

## Modeling Notes

- Binary target per side:
  - positive class: `Transplanted`
- Calibration is selected by lower Brier score between:
  - isotonic
  - sigmoid
- Threshold is selected from ROC curve under a false-positive-rate constraint (`max_fpr=0.10`).

## Demo / Presentation Tips
If you are showing this publicly (GitHub, resume, interview):
- include 1-2 screenshots from each dashboard,
- mention this was built under hackathon time constraints,
- emphasize the combination of **predictive modeling + interpretable clinical UI**.

## Resume-Ready Project Description

Built a dual-dashboard clinical decision support prototype during BioHack OSU 2026 using XGBoost + calibrated probabilities to estimate kidney transplant risk and monitor NMP lab homeostasis, with interactive Streamlit interfaces for cohort comparison, trend-based instability alerts, and transparent target-range logic.

## Known Limitations

- Current training/evaluation uses one dataset split and hackathon-time defaults.
- No formal external validation or prospective deployment.
- Intended as a research/prototyping artifact, not clinical software.

## License
No license file is currently included. Add one before open-source distribution.
