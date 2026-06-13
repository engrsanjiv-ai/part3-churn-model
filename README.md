# 📌 Part 3: D2C Customer Churn Prediction Model & Model Card

## D2C Customer Churn Intelligence — Capstone Project


# 📖 Project Overview

This repository contains a complete churn prediction model for a D2C personal-care brand. The model identifies customers likely to churn in the next 60 days, enabling targeted retention campaigns.


**Model:** XGBoost Classifier  
**Performance:** ROC-AUC 0.8710 | F1-Score 0.8075 | Precision 73.3% | Recall 89.9%
---

## Repository Structure

```
part3-churn-model/
├── churn_model.ipynb              # Main modeling notebook
├── model.pkl                      # Trained XGBoost model artifact
├── model_artifacts.pkl            # Feature names & label encoders
├── metrics.json                   # Model performance metrics
├── model_card.md                  # Comprehensive model card
├── error_analysis.md              # Detailed error analysis with examples
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/                          # Data folder (reference)
│   ├── rfm_modeling_snapshot.csv
│   ├── customers.csv
│   ├── orders.csv
│   ├── support_tickets.csv
│   ├── web_events_snapshot.csv
│   ├── churn_labels.csv
│   ├── intervention_history.csv
│   └── DATA_DICTIONARY.md
└── feature_importance.png         # Generated visualization
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Tested with Python 3.9+**

### 2. Run the Notebook

```bash
jupyter notebook churn_model.ipynb
```

Execute all cells sequentially to:
- Load and inspect data
- Train baseline (Logistic Regression) and final (XGBoost) models
- Evaluate performance metrics
- Generate error analysis with customer examples
- Save model and metrics

### 3. Expected Runtime

- Full notebook execution: ~2–3 minutes
- Model training: ~30 seconds
- Output files generated: 4 (model.pkl, model_artifacts.pkl, metrics.json, feature_importance.png)

---

## Key Files Explained

### churn_model.ipynb

**10-section Jupyter notebook covering the complete modeling pipeline:**

1. **Data Loading** – Loads `rfm_modeling_snapshot.csv` with 2,400 customers and 25 pre-engineered features
2. **Data Inspection** – Schema, data quality, missing values, target distribution
3. **Leakage Prevention** – Verifies all features use only data available ≤ 2025-09-30
4. **Data Preparation** – Train/val/test split, categorical encoding, feature-target matrix preparation
5. **Baseline Model** – Logistic Regression for comparison
6. **Stronger Model** – XGBoost with hyperparameter tuning
7. **Evaluation & Threshold** – Metrics (accuracy, precision, recall, F1, ROC-AUC) + business-justified threshold selection
8. **Error Analysis** – False negative and false positive deep-dive with customer-level examples and business interpretations
9. **Feature Importance** – Top 15 features driving predictions with visualization
10. **Model Saving** – Saves model.pkl, model_artifacts.pkl, and metrics.json for deployment

### model.pkl

Binary pickle file containing the trained XGBoost model.


### model_artifacts.pkl

Contains supplementary artifacts:
- `feature_names` – List of all 25 features used for predictions
- `label_encoders` – LabelEncoder objects for categorical variables
- `threshold` – Optimal decision threshold (0.5477)

### metrics.json

JSON file with model performance on test/validation sets:

```json
{
  "model": "XGBoost",
  "threshold": 0.5477,
  "test_accuracy": 0.7857,
  "test_precision": 0.7330,
  "test_recall": 0.8988,
  "test_f1_score": 0.8075,
  "test_roc_auc": 0.8710,
  "n_features": 25,
  "n_test_samples": 336,
  "churn_rate_test": 0.5000
}
```

### model_card.md

Comprehensive documentation including:
- Model purpose, type, and approach
- Data & feature descriptions
- Performance metrics (test/validation)
- Feature importance interpretation
- Error analysis summary
- Limitations & when NOT to use model
- Ethical considerations & fairness guidelines
- Deployment & monitoring recommendations
- Development & test coverage notes


### error_analysis.md

Detailed error analysis with **10 specific customer examples per error type**:

- **False Negatives (17 cases):** Customers who churned but model missed
  - 10 detailed examples with feature values and individual business interpretation
  - Root cause patterns identified (single-order customers, zero-session exits, support dissatisfaction)
  - Mitigation recommendations

- **False Positives (55 cases):** Loyal customers incorrectly flagged
  - 10 detailed examples showing why signals mimicked churn (long replenishment cycles, dormant-but-retained)
  - Business impact quantification (₹11,000–₹27,500 per cohort)
  - Recommendations to reduce false positives

- **True Positives (151 cases):** Successfully identified churners (reference)
- **True Negatives (113 cases):** Successfully identified loyalists (reference)

- **Comparative Analysis:** FN vs FP breakdown across recency, sessions, frequency, monetary, and cost
- **Net Model Value:** Estimated ₹449,750 value per test cohort after accounting for all error costs
- **Actionable Recommendations:** 7 prioritized improvements for next iteration

---

## Model Details

### Algorithm: XGBoost

**Hyperparameters:**
```python
{
  'n_estimators': 500,
  'max_depth': 6,
  'learning_rate': 0.05,
  'subsample': 0.85,
  'colsample_bytree': 0.85,
  'min_child_weight': 3,
  'gamma': 0.1,
  'scale_pos_weight': 3,
  'reg_alpha': 0.1,
  'reg_lambda': 1.5,
  'random_state': 42,
  'eval_metric': 'logloss',
  'early_stopping_rounds': 20
}
```

### Decision Threshold: 0.5477

**Justification:**
- Optimized for F1-score (business-aware trade-off between recall and precision)
- 89.9% recall: Catches ~9 in 10 actual churners
- 73.3% precision: When model says "churn", 73.3% actually churn
- Suitable for retention campaigns where cost-per-customer (₹200–500) << customer LTV (₹3,500–5,000)

### Features (25 total)

| Category | Count | Examples |
|---|---|---|
| Demographics | 6 | city_tier, age_group, acquisition_channel |
| RFM | 3 | recency_days, frequency_180d, monetary_180d |
| Purchase Behavior | 4 | return_rate_180d, avg_discount_pct_180d |
| Support & Service | 3 | ticket_count_90d, negative_ticket_rate_90d |
| Web/App Engagement | 8 | sessions_30d,
---

## Performance Summary

### Test Set Results (336 customers)

| Metric | Value |
|---|---|
| Accuracy | 79.76% |
| Precision | 74.51% |
| Recall | 90.48% |
| F1-Score | 0.8172 |
| ROC-AUC | 0.8777 |

### Confusion Matrix

|  | Pred. No-Churn | Pred. Churn |
|---|---|---|
| Actual No-Churn | 116 | 52 |
| Actual Churn | 16 | 152 |

### Comparison: Baseline vs. Final Model

| Metric | Baseline (LR) | Final (XGBoost) | Improvement |
|---|---|---|---|
| ROC-AUC | 0.7890 | 0.8777 | +8.8% ↑ |
| F1-Score | 0.6850 | 0.8172 | +19.4% ↑ |

---

## Leakage Prevention ✅

**All features verified to use ONLY data available on or before 2025-09-30:**

- ✅ Order history: Pre-snapshot only (snapshot date is 2025-09-30)
- ✅ Support tickets: Pre-snapshot only
- ✅ Web activity: 30-day snapshot as of 2025-09-30
- ✅ Customer demographics: Static as of signup
- ✅ No post-snapshot information included

---

## Model Limitations
  
Refer model_card.md 
---

## Monitoring & Retraining

### Data Drift Detection

**Monthly:**
- Compare feature distributions vs. baseline (alert if >15% shift)
- Check key features: recency_days, sessions_30d, monetary_180d

### Performance Monitoring

**Quarterly:**
- Recompute confusion matrix on holdout data
- Alert if: precision < 70% OR recall < 80% OR F1 < 0.75
- Track ROC-AUC; alert if < 0.82

### Retraining Triggers

- **Schedule:** Quarterly (every 90 days)
- **Drift Trigger:** Any feature distribution shift >20%
- **Performance Trigger:** Validation ROC-AUC drops below 0.82
- **Time Trigger:** >180 days since last model update

### Business Outcome Tracking

- Measure: % of flagged customers actually retained
- Measure: Campaign ROI = (revenue retained – campaign cost) / campaign cost
- Target: Define ROI benchmark after first campaign cycle

---

## Deployment Integration

Refer model_card.md 

### API Response Example

Refer model_card.md 

---

## Reproducibility

### Environment

- **Python:** 3.9+ (tested on 3.10, 3.11)
- **OS:** Windows/Mac/Linux
- **Dependencies:** See requirements.txt

### Random Seed

All randomization fixed with `RANDOM_SEED = 42` for reproducibility across runs.

### Data Path

data folder structure:
```
part3-churn-model/
├── data/
│   ├── rfm_modeling_snapshot.csv  (required)
│   ├── customers.csv
│   ├── orders.csv
│   └── ...
```

---

## Support & Questions

### Troubleshooting

**Issue:** Model fails to load  
**Solution:** Ensure model.pkl is not corrupted; regenerate by running notebook

**Issue:** Different predictions than expected  
**Solution:** Check feature order matches model_artifacts.pkl; verify categorical encoding

**Issue:** Threshold seems too high/low  
**Solution:** See "Decision Threshold" section in model_card.md for justification; adjust threshold variable if needed

---

## References

- **Model Card Format:** Inspired by Model Cards for Model Reporting (Mitchell et al., 2019)
- **Feature Engineering:** Based on RFM analysis + engagement metrics
- **Threshold Optimization:** Precision-Recall curve maximization for F1-score

---


## Contact

For questions, feedback, or deployment support:
- Refer to model_card.md for complete documentation
- Refer to error_analysis.md for model behavior insights
- Review notebook cells 8-9 for detailed error analysis

**Version:** 1.0  


