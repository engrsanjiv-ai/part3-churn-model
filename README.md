# Part 3: D2C Customer Churn Prediction Model & Model Card

## Overview

This repository contains a complete churn prediction model for a D2C personal-care brand. The model identifies customers likely to churn in the next 60 days, enabling targeted retention campaigns.


**Model:** XGBoost Classifier  
**Performance:** ROC-AUC 0.8777 | F1-Score 0.8172 | Precision 74.5% | Recall 90.5%

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
- Output files generated: 3 (model.pkl, metrics.json, feature_importance.png)

---

## Key Files Explained

### churn_model.ipynb

**10-section Jupyter notebook covering the complete modeling pipeline:**

1. **Data Loading** – Loads `rfm_modeling_snapshot.csv` with 2,400 customers and 25 pre-engineered features
2. **Data Inspection** – Schema, data quality, missing values, target distribution
3. **Leakage Prevention** – Verifies all features use only data available ≤ 2025-09-30
4. **Data Preparation** – Categorical encoding, feature-target matrices, train/val/test split
5. **Baseline Model** – Logistic Regression for comparison
6. **Stronger Model** – XGBoost with hyperparameter tuning
7. **Evaluation & Threshold** – Metrics (accuracy, precision, recall, F1, ROC-AUC) + business-justified threshold selection
8. **Error Analysis** – 10+ customer examples for false negatives/positives with business interpretations
9. **Feature Importance** – Top 15 features driving predictions with visualization
10. **Model Saving** – Saves model, artifacts, and metrics for deployment

### model.pkl

Binary pickle file containing the trained XGBoost model.

**Loading:**
```python
import joblib
model = joblib.load('model.pkl')

# Predict on new data
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.3329).astype(int)
```

### model_artifacts.pkl

Contains supplementary artifacts:
- `feature_names` – List of all 25 features used for predictions
- `label_encoders` – LabelEncoder objects for categorical variables
- `threshold` – Optimal decision threshold (0.3329)

### metrics.json

JSON file with model performance on test/validation sets:

```json
{
  "model": "XGBoost",
  "threshold": 0.3329,
  "test_accuracy": 0.7976,
  "test_precision": 0.7451,
  "test_recall": 0.9048,
  "test_f1_score": 0.8172,
  "test_roc_auc": 0.8777,
  "n_features": 25,
  "n_test_samples": 336,
  "churn_rate_test": 0.50
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

**Sections:** 12 major sections covering all aspects of model governance

### error_analysis.md

Detailed error analysis with **10+ specific customer examples**:

- **False Negatives (16 cases):** Customers who churned but model missed
  - 10 detailed examples with feature values and business interpretation
  - Root cause analysis
  - Mitigation recommendations

- **False Positives (52 cases):** Loyal customers incorrectly flagged
  - 10 detailed examples showing why they were retained
  - Business impact quantification
  - Recommendations to reduce false positives

- **True Positives (152 cases):** Successfully identified churners (reference)
- **True Negatives (116 cases):** Successfully identified loyalists (reference)

- **Aggregate Summary:** Sensitivity, specificity, predictive values
- **ROI Analysis:** Campaign profitability despite errors (9x return)
- **Actionable Recommendations:** Prioritized improvements for next iteration

---

## Model Details

### Algorithm: XGBoost

**Hyperparameters:**
```python
{
  'n_estimators': 100,
  'max_depth': 5,
  'learning_rate': 0.1,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'random_state': 42,
  'eval_metric': 'logloss',
  'early_stopping_rounds': 10
}
```

### Decision Threshold: 0.3329

**Justification:**
- Optimized for F1-score (business-aware trade-off between recall and precision)
- 90.5% recall: Catches ~9 in 10 actual churners
- 74.5% precision: When model says "churn", 74.5% actually churn
- Suitable for retention campaigns where cost-per-customer (₹350) << customer LTV (₹4,200)

### Features (25 total)

| Category | Count | Examples |
|---|---|---|
| Demographics | 6 | city_tier, age_group, acquisition_channel |
| RFM | 3 | recency_days, frequency_180d, monetary_180d |
| Purchase Behavior | 4 | return_rate_180d, avg_discount_pct_180d |
| Support & Service | 3 | ticket_count_90d, negative_ticket_rate_90d |
| Web/App Engagement | 8 | sessions_30d, product_views_30d, cart_adds_30d |
| Lifecycle | 1 | days_since_signup |
| Other | 3 | skin_type, marketing_consent, preferred_category |

### Top 5 Features (by importance)

1. **recency_days** (28.4%) – Days since last purchase
2. **last_visit_days_ago** (19.2%) – Days since last web visit
3. **sessions_30d** (16.8%) – Web sessions in past 30 days
4. **frequency_180d** (15.5%) – Order count in past 180 days
5. **ticket_count_90d** (10.2%) – Support tickets in past 90 days

**Insight:** Engagement and recency dominate. Low-engagement, long-inactive customers are at highest churn risk.

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

**Verification:** See Section 3 in notebook for explicit leakage checks.

---

## Model Limitations

### Known Limitations

1. **Recency Bias** – Heavy dependence on recent purchase patterns; seasonal dips may trigger false alarms
2. **New Customer Blind Spot** – Customers with <30 days account age have sparse history
3. **Web Activity Dependency** – Assumes web/app usage; phone/email-only customers may appear disengaged
4. **Temporal Drift** – May degrade if customer behavior changes significantly
5. **Label Definition** – No purchase in 60 days = churn; doesn't distinguish true churn vs. seasonal low periods

### When NOT to Use

❌ Don't use for customers <30 days old  
❌ Don't use for seasonal products without calendar adjustments  
❌ Don't use after 90+ days without retraining  
❌ Don't rely on scores >0.9 without manual review  
❌ Don't use as sole basis for customer termination  

---

## Ethical Considerations

### Fairness Monitoring

Recommended fairness checks by demographic:
- Churn prediction rate by age_group (target: <10% disparity)
- Churn prediction rate by city_tier (target: <10% disparity)
- Churn prediction rate by acquisition_channel (target: <10% disparity)

### Responsible Use

✅ **DO:**
- Use predictions for retention opportunity identification
- Combine model scores with business context
- Monitor actual outcomes to validate effectiveness
- Retrain quarterly with new data

❌ **DON'T:**
- Use as sole basis for customer termination
- Apply identical offers to all flagged segments
- Rely on scores without human review for VIP customers
- Deploy without fairness audits

---

## Monitoring & Retraining

### Data Drift Detection

**Monthly:**
- Compare feature distributions vs. baseline (alert if >15% shift)
- Check key features: recency, sessions_30d, monetary_180d

### Performance Monitoring

**Quarterly:**
- Recompute confusion matrix on holdout data
- Alert if: precision < 70% OR recall < 70% OR F1 < 0.72
- Track ROC-AUC; alert if < 0.82

### Retraining Triggers

- **Schedule:** Quarterly (every 90 days)
- **Drift Trigger:** Any feature distribution shift >20%
- **Performance Trigger:** Validation ROC-AUC drops below 0.82
- **Time Trigger:** >180 days since last model update

### Business Outcome Tracking

- Measure: % of flagged customers actually retained
- Measure: Campaign ROI = (revenue retained – campaign cost) / campaign cost
- Target: >7x ROI (current: ~9x based on test data)

---

## Deployment Integration

### Loading & Using the Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('model.pkl')
artifacts = joblib.load('model_artifacts.pkl')

# Load new customer data
new_customers = pd.read_csv('customers_to_score.csv')

# Predict
churn_proba = model.predict_proba(new_customers)[:, 1]
churn_pred = (churn_proba >= artifacts['threshold']).astype(int)

# Add to dataframe
new_customers['churn_probability'] = churn_proba
new_customers['predicted_churn'] = churn_pred
```

### API Response Example

```json
{
  "customer_id": "CUST00123",
  "churn_probability": 0.72,
  "predicted_class": 1,
  "risk_level": "high",
  "recommendation": "Immediate retention action recommended"
}
```

---

## Reproducibility

### Environment

- **Python:** 3.9+ (tested on 3.10, 3.11)
- **OS:** Windows/Mac/Linux
- **Dependencies:** See requirements.txt

### Random Seed

All randomization fixed with `RANDOM_SEED = 42` for reproducibility across runs.

### Data Path

Assumes data folder structure:
```
part3-churn-model/
├── data/
│   ├── rfm_modeling_snapshot.csv  (required)
│   ├── customers.csv
│   ├── orders.csv
│   └── ...
```

Modify `data_path = 'data/rfm_modeling_snapshot.csv'` in notebook if data location differs.

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

## License

This model and documentation are provided as-is for the D2C personal-care brand retention campaign.

---

## Contact

For questions, feedback, or deployment support:
- Refer to model_card.md for complete documentation
- Refer to error_analysis.md for model behavior insights
- Review notebook cells 8-9 for detailed error analysis

**Version:** 1.0  


