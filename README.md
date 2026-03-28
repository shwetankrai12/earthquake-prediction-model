# Earthquake Damage Prediction
### AI Disaster Early Warning System — XGBoost vs Random Forest

Given seismic sensor data, this model predicts the likely damage level of an earthquake: **Low**, **Medium**, or **High**. It is designed as a component of an early warning pipeline where speed and reliability of classification matter.

---

## Overview

| | |
|---|---|
| **Dataset** | USGS Earthquake Database (via Kaggle) |
| **Task** | Multi-class classification (3 damage levels) |
| **Models** | XGBoost · Random Forest |
| **Primary metric** | F1-score (macro) |
| **Validation** | 5-fold stratified cross-validation |

---

## Models

Two models are trained and compared:

**XGBoost** (primary)
- 300 estimators, max depth 6, learning rate 0.1
- L1 + L2 regularization (`reg_alpha=0.5`, `reg_lambda=1.5`)
- Sample weights for class imbalance

**Random Forest** (comparison)
- 300 estimators, max depth 10
- Built-in `class_weight='balanced'`

Saved models: `earthquake_xgb_model.pkl`, `earthquake_rf_model.pkl`

---

## Features Used

| Feature | Description |
|---|---|
| `magnitude` | Earthquake magnitude |
| `depth_km` | Depth below surface (km) |
| `latitude`, `longitude` | Geographic location |
| `depth_error`, `mag_error` | Measurement uncertainties |
| `azimuthal_gap` | Gap in seismic station coverage |
| `horizontal_distance` | Distance to nearest station |
| `rms` | Root mean square of residuals |
| `abs_latitude` | Distance from equator (engineered) |
| `mag_depth_ratio` | Magnitude / (depth + 1) (engineered) |
| `log_depth` | Log-scaled depth (engineered) |
| `mag_squared` | Non-linear magnitude term (engineered) |
| `mag_type_encoded` | Encoded magnitude type |

---

## Key Fixes Applied

This version corrects several critical ML issues from the original notebook:

**1. Data leakage fixed**
The original label function used simple magnitude/depth thresholds — the same features used for training — making accuracy trivially 99%+. Labels now use a probabilistic damage score with controlled random noise, simulating unmeasured real-world factors (soil type, building quality, population density).

**2. Class imbalance fixed**
Uses stratified train/test splits and sample weights instead of SMOTE, avoiding synthetic data artifacts.

**3. Evaluation fixed**
F1-macro is the primary metric, not raw accuracy. Cohen's Kappa also reported for agreement beyond chance.

**4. Prediction function fixed**
The live prediction function now engineers the exact same features the model was trained on — no feature mismatch at inference time.

---

## Usage

### Install dependencies
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn kagglehub joblib
```

### Run the notebook
Open `earthquake_prediction_fixed.ipynb` in Jupyter or Google Colab and run all cells. The notebook will:
1. Download the USGS dataset via `kagglehub`
2. Clean and engineer features
3. Train both models
4. Evaluate with confusion matrices and cross-validation
5. Save models to `earthquake_xgb_model.pkl` and `earthquake_rf_model.pkl`

### Live prediction
```python
import joblib

xgb_model = joblib.load('earthquake_xgb_model.pkl')

result = predict_earthquake_damage(
    magnitude=6.5,
    depth_km=10.0,
    latitude=35.6,
    longitude=139.7,
    depth_error=5.0,
    mag_error=0.1,
    azimuthal_gap=50.0,
    horizontal_distance=1.0,
    rms=0.8,
    mag_type='MW'
)
# Returns: 'Low', 'Medium', or 'High'
```

---

## Project Structure

```
├── earthquake_prediction_fixed.ipynb   # Main notebook
├── earthquake_xgb_model.pkl            # Trained XGBoost model
├── earthquake_rf_model.pkl             # Trained Random Forest model
└── README.md
```

---

## Dataset

USGS Earthquake Database — sourced from Kaggle via `kagglehub`.

```python
import kagglehub
path = kagglehub.dataset_download("usgs/earthquake-database")
```

A Kaggle account and API token are required to download the dataset.

---

## Limitations

- Damage labels are synthetically generated (no ground-truth structural damage data). The model approximates real-world damage likelihood but is not validated against actual disaster records.
- The noise-injected label scheme is intentional — it prevents overfitting to deterministic rules — but means the task has an inherent accuracy ceiling.
- Not production-ready as a standalone early warning system without integration with real-time seismic feeds and calibration against observed damage data.
