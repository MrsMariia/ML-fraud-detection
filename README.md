# 🔍 Credit Card Fraud Detection

> End-to-end ML pipeline for detecting fraudulent transactions on the Kaggle ULB dataset —
> from baseline models to SHAP explainability, cost-optimised thresholds, and unsupervised anomaly detection.

---

## Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9772 | 0.7212 | — | — | — |
| Random Forest | 0.9773 | 0.8149 | — | — | — |
| Gradient Boosting | 0.8611 | 0.5495 | — | — | — |
| Random Forest (SMOTE) | 0.9851 | 0.8231 | — | — | — |
| LightGBM (SMOTE) | 0.9824 | 0.8546 | — | — | — |
| Stacking Ensemble (SMOTE) | 0.9838 | 0.8425 | — | — | — |
| **XGBoost (SMOTE)** ✅ | **0.9842** | **0.8639** | **0.926** | **0.813** | **0.866** |
| Autoencoder (unsupervised) | 0.9308 | 0.4913 | — | — | — |
| VAE (unsupervised) | 0.9383 | 0.2939 | — | — | — |

**Best model — XGBoost + SMOTE:** catches **100 out of 123 fraud cases** in the test set with only **8 false alarms**.

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB Machine Learning Group.

| | |
|---|---|
| Transactions | 284,807 |
| Fraud cases | 492 (0.172%) |
| Features | V1–V28 (PCA-anonymised) + Time, Amount, Class |
| Period | September 2013, 2 days |

> ⚠️ The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the project root.

---

## Notebook Structure
```
01  Data Loading & Inspection
02  EDA + Feature Engineering       ← hour, dow, Amount_log, hour aggregates
03  Data Preparation                 ← train/test split (stratified)
04  Baseline Models                  ← LR · RF · GB with class_weight="balanced"
05  Evaluation & Visualisation       ← ROC/PR curves, confusion matrices, scatter
06  SMOTE                            ← synthetic oversampling on train only
07  Models on SMOTE                  ← LR · RF · GB · LightGBM · XGBoost + loss curves
08  Stacking Ensemble                ← LR+RF+LGBM+XGB → Logistic meta-learner
09  Autoencoder & VAE                ← unsupervised anomaly detection (train on normal only)
10  Final Comparison                 ← all 11 models, side-by-side plots
11  Temporal Split                   ← train on Day 1, test on Day 2 (honest evaluation)
12  Cost-based Threshold             ← FN=$500 · FP=$5 → cost curve + threshold table
13  Probability Calibration          ← reliability diagram + isotonic regression
14  SHAP Analysis                    ← global importance, waterfall, dependence plot
```

---

## Key Findings

**SMOTE effect is model-dependent.**
Gradient Boosting without `class_weight` gains +22% PR-AUC from SMOTE.
Logistic Regression and Random Forest show minimal improvement — `class_weight="balanced"` already handles imbalance.

**Stacking doesn't beat a single XGBoost** on this task — Logistic Regression as a base model introduces noise that dilutes the ensemble.

**Unsupervised methods (AE, VAE) substantially underperform** supervised ones when labels are available (PR-AUC 0.49 vs 0.86). However, the Autoencoder detects a different *kind* of fraud — anomalies invisible to XGBoost — making it a valuable complementary signal.

**VAE underperforms plain AE** on PCA-transformed features. KL regularisation over-compresses an already-compressed space, reducing reconstruction error discriminability.

**Default threshold 0.5 is suboptimal.** With FN cost 100× higher than FP, the cost-optimal threshold catches ~90% of fraud at significantly lower total loss. See Section 12 for the full threshold lookup table.

**V14 is the single most important feature** (SHAP dependence plot shows a sharp decision boundary near V14 ≈ −5). A missed fraud case analysis reveals that False Negatives exhibit atypical PCA component values across the board — likely a different fraud pattern not well represented in training data.

---

## Visualisations

The notebook produces 20+ plots including:

- Class distribution & Amount histograms
- Fraud rate by hour of day (Plotly interactive)
- ROC and Precision-Recall curves (all models)
- LightGBM & XGBoost training loss curves with early stopping marker
- Autoencoder & VAE reconstruction error distributions
- Confusion matrices (3×4 grid, green = correct / red = error)
- Cost curve: total cost vs threshold
- Reliability diagram (calibration curve)
- SHAP beeswarm, mean |SHAP| bar chart, waterfall plots, dependence plot

---

## Setup
```bash
git clone https://github.com/your-username/fraud-detection
cd fraud-detection
pip install -r requirements.txt
```

Place `creditcard.csv` in the project root, then run the notebook top to bottom.
```
fraud-detection/
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
└── creditcard.csv          ← download from Kaggle (not tracked by git)
```

---

## Requirements
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
lightgbm
xgboost
torch
shap
jupyter
```

Full pinned versions in `requirements.txt`.

---

## Why PR-AUC and not Accuracy?

A classifier that always predicts *"normal"* achieves **99.83% accuracy** on this dataset — a completely useless result. PR-AUC measures performance specifically on the minority (fraud) class and is not inflated by the overwhelming number of normal transactions. All model comparisons in this project use PR-AUC as the primary metric.

---

## License

MIT
