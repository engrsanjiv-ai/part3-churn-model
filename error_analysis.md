# Error Analysis: Churn Prediction Model


**Model:** XGBoost Classifier  
**Decision Threshold:** 0.3328796327114105  
**Dataset:** Test Set (336 customers)

---

## Executive Summary

The current churn model evaluation is based on the live test set and actual model outputs.

- **Accuracy:** 0.7976
- **Precision:** 0.7451
- **Recall:** 0.9048
- **F1 Score:** 0.8172
- **ROC AUC:** 0.8777

### Confusion Matrix (Test Set)
- True Positives: 152
- False Positives: 52
- False Negatives: 16
- True Negatives: 116

The model is strong at catching churners, but it still produces a material number of false positives. The following analysis uses the notebook data and actual prediction results together.

---

## 1. False Negatives: Actual churners missed

**Count:** 16 of 336 test customers (4.8%)  
**Business impact:** HIGH

False negatives are the most costly error type for retention: these customers churned, but the model did not flag them.

### Key observations
- The model is conservative for moderate-risk churners.
- Some missed customers still had strong value or engagement signals, so the model under-estimated their risk.
- These false negatives often sit below the threshold despite recent purchase or loyalty indicators.

### Example false negatives

1. **CUST00184**
   - Predicted churn probability: 0.0524
   - Recency: 14 days
   - Frequency (180d): 3
   - Monetary (180d): 2456.91
   - Sessions (30d): 6
   - Loyalty tier: Platinum
   - Channel: Instagram

2. **CUST01655**
   - Predicted churn probability: 0.0845
   - Recency: 13 days
   - Frequency (180d): 2
   - Monetary (180d): 1358.99
   - Sessions (30d): 2
   - Channel: Google Search

3. **CUST00592**
   - Predicted churn probability: 0.1039
   - Recency: 20 days
   - Frequency (180d): 1
   - Monetary (180d): 627.36
   - Sessions (30d): 3
   - Channel: Referral

4. **CUST02060**
   - Predicted churn probability: 0.1238
   - Recency: 23 days
   - Frequency (180d): 2
   - Monetary (180d): 1331.01
   - Sessions (30d): 4
   - Channel: Google Search

5. **CUST01303**
   - Predicted churn probability: 0.1315
   - Recency: 20 days
   - Frequency (180d): 1
   - Monetary (180d): 844.74
   - Sessions (30d): 3
   - Channel: Google Search

### False negative root causes
- **Threshold calibration:** Many missed churners are near the decision boundary and would be recovered with a slightly more sensitive threshold for key segments.
- **High-value/loyalty signals under-weighted:** The model may not fully capture value signals for customers with good historical spend andratings.
- **False reassurance from limited negative signals:** Low ticket counts or good recent sessions can hide an underlying churn risk.

### Recommendations
- Use segment-specific thresholds for high-value or loyalty-tier customers.
- Add explicit loyalty and satisfaction features to reduce missed churn for moderately engaged customers.
- Manually review borderline predictions in the 0.10–0.33 range for high-value customers.

---

## 2. False Positives: Customers flagged but did not churn

**Count:** 52 of 336 test customers (15.5%)  
**Business impact:** MODERATE

False positives generate unnecessary retention effort and cost, so reducing these improves precision without damaging recall.

### Key observations
- Many false positives are long-recency customers with low recent activity.
- Several false positives still have strong historical loyalty or spending.
- The model tends to overpredict churn for customers with sparse recent sessions but stable purchase history.

### Example false positives

1. **CUST01246**
   - Predicted churn probability: 0.9025
   - Recency: 262 days
   - Frequency (180d): 0
   - Monetary (180d): 0.00
   - Sessions (30d): 1
   - Loyalty tier: Silver
   - Channel: Influencer

2. **CUST01370**
   - Predicted churn probability: 0.8960
   - Recency: 161 days
   - Frequency (180d): 2
   - Monetary (180d): 1246.04
   - Sessions (30d): 2
   - Channel: Organic

3. **CUST00437**
   - Predicted churn probability: 0.8901
   - Recency: 151 days
   - Frequency (180d): 1
   - Monetary (180d): 729.22
   - Sessions (30d): 0
   - Channel: Marketplace

4. **CUST01405**
   - Predicted churn probability: 0.8597
   - Recency: 140 days
   - Frequency (180d): 1
   - Monetary (180d): 1013.03
   - Sessions (30d): 2
   - Loyalty tier: Gold

5. **CUST01017**
   - Predicted churn probability: 0.8359
   - Recency: 133 days
   - Frequency (180d): 2
   - Monetary (180d): 1167.28
   - Sessions (30d): 3
   - Channel: Influencer

### False positive root causes
- **Long recency alone:** The model treats long purchase gaps as churn risk even for customers who remain loyal.
- **Natural purchase cycles:** Seasonal or cyclical buying behavior is not well captured.
- **Loyalty signals not strong enough:** Customers with prior value and tier status still get flagged.

### Recommendations
- Apply business rules for loyalty tiers (Gold/Platinum) to lower churn score or require stronger churn signals.
- Add shopping intent and engagement signals such as wishlist adds, cart activity, and campaign response.
- Incorporate seasonality or category-specific purchase rhythms to avoid treating normal gaps as churn.

---

## 3. Summary and next steps

This corrected analysis aligns with actual model outputs and notebook-derived predictions:
- False positives are 52, not 56
- False negatives remain 16
- The decision threshold is 0.3328796327114105

### Immediate action items
- Tune the threshold for high-value / loyalty-tier segments.
- Add business rules to reduce over-treatment for long-recency but loyal customers.
- Monitor the 0.10–0.33 probability band for missed churn risk.

**END OF ERROR ANALYSIS**
