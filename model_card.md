# Model Card: D2C Customer Churn Prediction Model

**Model Version:** 1.0

---

## 1. Model Overview

### Purpose
Predict which D2C personal-care brand customers are likely to churn in the next 60 days to enable targeted retention campaigns. The model scores each customer with a churn probability, allowing the product, marketing, and customer-support teams to prioritize retention efforts.

### Problem Type
**Binary Classification** (Churn / No-Churn)

### Model Type
**XGBoost Classifier** with business-optimized decision threshold

### Key Model Characteristics
- **Algorithm:** Extreme Gradient Boosting (XGBoost)
- **Number of Trees:** 500 (with early stopping, rounds=20)
- **Max Depth:** 6
- **Learning Rate:** 0.05
- **Subsample:** 0.85
- **Column Subsample:** 0.85
- **Min Child Weight:** 3
- **Gamma:** 0.1
- **Scale Pos Weight:** 3
- **L1 Regularization (reg_alpha):** 0.1
- **L2 Regularization (reg_lambda):** 1.5
- **Decision Threshold:** 0.5477 (F1-score optimized)

---

## 2. Data & Features

### Data Source
- **Primary Data:** `rfm_modeling_snapshot.csv` (pre-engineered features)
- **Snapshot Date:** 2025-09-30
- **Total Samples:** 2,400 customers

### Data Split
| Set | Size | Churn Rate |
|---|---|---|
| Training | 1,728 customers | 46.99% |
| Validation | 336 customers | 43.75% |
| Test | 336 customers | 50.00% |

### Feature Categories (25 features total)

#### Demographics (6 features)
- `city_tier` (Tier 1, 2, 3)
- `age_group` (18-24, 25-34, 35-44, 45+)
- `acquisition_channel` (Google Search, Instagram, Influencer, Referral, Marketplace, Organic)
- `loyalty_tier` (Silver, Gold, Platinum)
- `preferred_category` (6 product categories)
- `marketing_consent` (Yes/No)

#### RFM Features (3 features)
- `recency_days` – Days since last purchase
- `frequency_180d` – Number of orders in last 180 days
- `monetary_180d` – Total spending in last 180 days (INR)

#### Purchase Behavior (4 features)
- `return_rate_180d` – Fraction of orders returned in last 180 days
- `avg_discount_pct_180d` – Average discount percentage received
- `avg_rating_180d` – Average satisfaction rating given
- `category_diversity_180d` – Number of distinct product categories purchased

#### Support & Service (3 features)
- `ticket_count_90d` – Number of support tickets in last 90 days
- `negative_ticket_rate_90d` – Fraction of tickets with negative sentiment
- `avg_resolution_hours_90d` – Average resolution time

#### Web/App Engagement (8 features)
- `sessions_30d` – Number of web/app sessions in last 30 days
- `product_views_30d` – Product detail pages viewed
- `cart_adds_30d` – Items added to cart
- `wishlist_adds_30d` – Items added to wishlist
- `abandoned_carts_30d` – Cart abandonments
- `email_opens_30d` – Marketing emails opened
- `campaign_clicks_30d` – Campaign link clicks
- `last_visit_days_ago` – Days since most recent visit

#### Customer Lifecycle (1 feature)
- `days_since_signup` – Account age in days

### Leakage Prevention
✅ **CONFIRMED:** All features use only data available on or before 2025-09-30
- Order history: Only orders placed on or before snapshot date
- Support tickets: Only tickets created on or before snapshot date
- Web activity: 30-day snapshot as of 2025-09-30
- No post-snapshot information is included in any feature

---

## 3. Model Performance

### Test Set Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.7857 |
| **Precision** | 0.7330 |
| **Recall** | 0.8988 |
| **F1-Score** | 0.8075 |
| **ROC-AUC** | 0.8710 |

### Validation Set Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.7946 |
| Precision | 0.7167 |
| Recall | 0.8776 |
| F1-Score | 0.7890 |
| ROC-AUC | 0.8822 |

### Confusion Matrix (Test Set)

|  | Predicted No-Churn | Predicted Churn |
|---|---|---|
| **Actual No-Churn** | 113 (TN) | 55 (FP) |
| **Actual Churn** | 17 (FN) | 151 (TP) |

### Model Comparison: Baseline vs Final

**Baseline Model (Logistic Regression):**
- Accuracy: 0.8185
- Precision: 0.8209
- Recall: 0.7483
- F1-Score: 0.7829
- ROC-AUC: 0.8846

**Final Model (XGBoost — Tuned + Threshold Optimized):**
- Accuracy: 0.7857
- Precision: 0.7330
- Recall: 0.8988 (**+15.1pp** improvement)
- F1-Score: 0.8075 (**+3.1%** improvement)
- ROC-AUC: 0.8710

> **Why XGBoost over Logistic Regression?** Although LR has higher accuracy and precision, XGBoost catches **89.9% of churners vs 74.8%** — approximately 130 extra customers saved per 1,000 churners. For churn use cases, recall is the priority metric.

---

## 4. Decision Threshold & Business Justification

### Threshold Selection: 0.5477

**Rationale:**
- **Optimized for:** F1-score maximization on validation set using Precision-Recall curve
- **Precision-Recall Trade-off:** 73.3% precision with 89.9% recall
- **Business Logic:**
  - **Recall Priority:** We want to catch most actual churners (89.9%) because missing a churner has high cost (lost customer lifetime value)
  - **Precision Acceptable:** False positive rate is tolerable because retention campaigns have low cost relative to customer acquisition cost

### Alternative Thresholds Considered

| Threshold | Description | Use Case |
|---|---|---|
| **0.5477** | **F1-maximized (SELECTED)** | **Best precision-recall balance** |
| 0.0170 | 85% recall target | Overly aggressive — flags nearly all customers, destroys precision |
| 0.50 | Default | Conservative — misses more churners |

**Selected Threshold Reasoning:**
The 0.5477 threshold provides the best F1-score balance for a retention campaign scenario where:
1. **Cost of False Negative (missing a churner):** ₹3,500–5,000 per lost customer (LTV loss)
2. **Cost of False Positive (unnecessary retention offer):** ₹200–500 per customer
3. The 10:1 cost ratio strongly justifies prioritizing recall over precision

---

## 5. Feature Importance

### Top 10 Features Driving Predictions

| Rank | Feature | Importance | Business Interpretation |
|---|---|---|---|
| 1 | `recency_days` | 0.1861 | Days since last purchase is by far the strongest churn signal |
| 2 | `negative_ticket_rate_90d` | 0.0580 | Support dissatisfaction is a leading churn driver |
| 3 | `return_rate_180d` | 0.0576 | High return rate signals product-fit issues |
| 4 | `last_visit_days_ago` | 0.0574 | Recent platform engagement strongly predicts retention |
| 5 | `monetary_180d` | 0.0464 | Spending level correlates with churn likelihood |
| 6 | `frequency_180d` | 0.0442 | Regular purchasing behaviour indicates loyalty |
| 7 | `category_diversity_180d` | 0.0395 | Broader product exploration reduces churn risk |
| 8 | `avg_resolution_hours_90d` | 0.0371 | Slow support resolution increases churn probability |
| 9 | `sessions_30d` | 0.0334 | Active web/app usage signals continued interest |
| 10 | `cart_adds_30d` | 0.0319 | Cart activity reflects purchase intent |

**Key Insight:** Recency dominates all other features at 0.1861 importance — nearly 3× the next feature. Beyond recency, support experience (negative ticket rate, slow resolution) and purchase behaviour (return rate, monetary value) are the next strongest signals. Customers who haven't purchased recently, have raised dissatisfied support tickets, and show high return rates are at the highest churn risk.

---

## 6. Error Analysis

### False Negatives (17 cases — Actual churners we missed)

**Common Characteristics (from actual error analysis):**
- Moderate recency: avg 45.9 days since last purchase (range: 9–100 days) — these customers weren't obviously lapsed
- Decent session activity: avg 6.8 sessions/30 days — appeared engaged on platform
- Low purchase frequency: avg 2.3 orders in 180 days (median: 2)
- Moderate spending: avg ₹1,685 in 180 days (range: ₹403–₹4,340)

**Why the model missed them:** These customers showed surface-level engagement (sessions, visits) but still churned — likely driven by factors the model underweights (product dissatisfaction, price sensitivity, competitor switching).

**Business Impact:** High — we miss retention opportunity with these 17 customers

**Mitigation:** Combine model score with rule-based triggers on purchase frequency; customers with ≤2 orders in 180 days should be reviewed regardless of predicted probability

### False Positives (55 cases — Loyal customers flagged but didn't churn)

**Common Characteristics (from actual error analysis):**
- High recency: avg 84.9 days since last purchase (range: 16–262 days) — genuinely looked lapsed
- Low session activity: avg 4.9 sessions/30 days (median: 3) — low platform engagement
- Very low purchase frequency: avg 1.5 orders in 180 days (median: 1)
- Lower spending: avg ₹1,079 in 180 days vs ₹1,685 for missed churners

**Why the model flagged them:** Their behavioural signals genuinely resembled churners — long recency, low frequency, low sessions. They likely experienced a seasonal or lifecycle dip but retained nonetheless.

**Business Impact:** Moderate — cost of unnecessary retention offer (₹200–500 per customer × 55 = ₹11,000–₹27,500) is manageable vs. customer LTV

**Mitigation:** Combine model scores with RFM loyalty tier; deprioritise retention offers for high-tenure customers with strong historical spend even if recently inactive

---

## 7. Model Limitations

### Known Limitations

1. **Recency Bias:** Model heavily depends on recent purchase patterns. Seasonal products may show false churn signals during off-season.

2. **New Customer Blind Spot:** Customers with fewer than 30 days since signup have sparse engagement history; model may underpredict churn.

3. **Web Activity Dependency:** Model assumes customers use web/app. Customers preferring phone/email ordering may appear disengaged without being at-risk.

4. **Temporal Drift:** Model trained on 2025-01-01 to 2025-09-30 data may degrade if customer behavior shifts significantly post-training.

5. **Label Definition:** Target assumes no purchase in 60-day window = churn. Doesn't distinguish between true churn vs. natural low-purchase periods.

6. **Class Imbalance:** Despite engineered features, the dataset has ~47.5% churn in training, which may not reflect real-world base rates.

### When NOT to Use This Model

❌ Do not use for customers acquired in the last 30 days (insufficient historical data)  
❌ Do not use for seasonal products without external calendar adjustments  
❌ Do not use after 90+ days without retraining (concept drift risk)  
❌ Do not use for highly personalized medical/healthcare products (different churn drivers)  
❌ Do not rely on probabilities >0.9 without manual review (extreme cases may indicate data errors)

---

## 8. Ethical Considerations & Bias

### Potential Biases

1. **Acquisition Channel Bias:** Customers acquired through paid ads may show different churn patterns than organic/referral customers. Retention strategy should account for channel differences.

2. **City Tier Bias:** Urban (Tier 1) customers may have different retention needs vs. rural (Tier 3) customers.

3. **Age Group Disparities:** Younger cohorts (18-24) may show higher churn due to experimentation; older cohorts (45+) may be more price-sensitive.

4. **Loyalty Tier Exclusion:** Customers not enrolled in the loyalty program (null values) are excluded from that feature, which may bias predictions against certain segments.

### Responsible Use Guidelines

✅ **DO:**
- Use model predictions to identify at-risk segments for deeper analysis
- Combine model scores with business context and customer service input
- Monitor actual churn outcomes to validate model effectiveness
- Retrain quarterly with fresh data
- Document retention campaign decisions and outcomes

❌ **DON'T:**
- Use model as the sole basis for any customer account decision
- Apply identical retention offers to all flagged customers (segment-specific strategies perform better)
- Rely on model without human review for high-value customers
- Use model for non-retention purposes (e.g., dynamic pricing discrimination)
- Deploy without fairness checks across demographic groups

### Fairness Metrics (Recommended for Monitoring)

Monitor disparate impact across groups:
- Churn prediction rate by `age_group`
- Churn prediction rate by `city_tier`
- Churn prediction rate by `acquisition_channel`
- Flag segments with >10% disparity for investigation

---

## 9. Deployment & Monitoring

### Model Artifacts

| File | Description |
|---|---|
| `model.pkl` | Trained XGBoost classifier |
| `model_artifacts.pkl` | Threshold, feature names, label encoders |
| `metrics.json` | Full performance metrics |

### Loading & Prediction

```python
import joblib

model = joblib.load('model.pkl')
artifacts = joblib.load('model_artifacts.pkl')

# Predict churn probability
proba = model.predict_proba(X_new)[:, 1]

# Apply optimized threshold
predictions = (proba >= artifacts['threshold']).astype(int)
# artifacts['threshold'] = 0.5477
```

### API Output Schema

```json
{
  "customer_id": "CUST00123",
  "churn_probability": 0.72,
  "predicted_class": 1,
  "risk_level": "high",
  "recommendation": "Immediate retention action recommended"
}
```

### Monitoring Plan

#### Data Drift Detection
- Monthly check: distribution of key features (`recency_days`, `sessions_30d`, `monetary_180d`)
- Alert if feature mean changes >15% month-over-month

#### Model Performance Monitoring
- Compute monthly confusion matrix on new holdout data
- Alert if precision drops below 70% or recall drops below 80%
- Track F1-score; alert if it falls below 0.75

#### Business Outcome Tracking
- Measure actual churn rate of flagged vs. unflagged customers
- Calculate campaign ROI: (revenue retained – campaign cost) / campaign cost
- Monitor customer satisfaction (NPS, review ratings) for retention campaign recipients

#### Retraining Triggers
- **Schedule:** Retrain quarterly (every 90 days)
- **Drift Trigger:** If any key feature distribution shifts >20%
- **Performance Trigger:** If validation ROC-AUC drops below 0.82
- **Time Trigger:** If >180 days since last model update

---

## 10. Development & Testing

### Development Environment
- **Python Version:** 3.9+
- **Key Libraries:** XGBoost, scikit-learn, pandas, numpy, joblib
- **Random Seed:** 42
- **Hardware:** CPU (no GPU required)

### Test Coverage
✅ Model loading from pickle: Validated  
✅ Prediction consistency: Validated  
✅ Batch prediction: Validated  
✅ Edge cases (missing values, outliers): Validated  
✅ Threshold behavior: Validated  
✅ Label encoder classes preserved in artifacts: Validated

### Code Quality
- Code reviewed for leakage: ✅ No leakage detected
- Features validated as pre-snapshot: ✅ Confirmed (snapshot date: 2025-09-30)
- Reproducibility: ✅ Random seed fixed at 42

---

## 11. Contact & Support

**Model Owner:** Sanjiv


For questions or issues:
1. Check model metrics in `metrics.json`
2. Review error analysis in notebook Section 8
3. Verify data quality and feature engineering in notebook Sections 2–3
4. Contact ML team with specific use cases

---

## 12. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-06-13 | Initial release — XGBoost classifier with F1-optimized threshold (0.5477) |

---

**END OF MODEL CARD**
