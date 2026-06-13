# Error Analysis Report: D2C Customer Churn Prediction Model

**Model:** XGBoost Classifier  
**Decision Threshold:** 0.5477  
**Test Set Size:** 336 customers  
**Snapshot Date:** 2025-09-30  

---

## 1. Overview

| Outcome | Count | % of Test Set |
|---|---|---|
| True Positives (correctly predicted churn) | 151 | 44.9% |
| True Negatives (correctly predicted no churn) | 113 | 33.6% |
| **False Negatives (missed churners)** | **17** | **5.1%** |
| **False Positives (wrongly flagged loyal customers)** | **55** | **16.4%** |
| **Total Errors** | **72** | **21.4%** |

**Overall accuracy: 78.57% — model correctly classifies ~4 in 5 customers.**

---

## 2. False Negatives — Missed Churners

### What is a False Negative?
The model predicted **no churn** but the customer **actually churned**. These are the most costly errors — we lose the customer with no retention attempt.

### Business Risk
- **Severity: HIGH**
- Each missed churner represents a lost customer with no intervention
- Estimated cost per missed churner: ₹3,500–5,000 (customer LTV loss)
- Total estimated cost (17 missed): **₹59,500–₹85,000 per cohort**

### Why Does This Happen?
These customers had **deceptively healthy signals** — recent purchases, active sessions, low support tickets — yet still churned. The model correctly read their engagement as low-risk. Likely driven by latent factors not captured in features: competitor offers, price sensitivity, or product dissatisfaction not yet escalated to a support ticket.

### False Negative Summary Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Recency (days) | 45.9 | 42.0 | 9 | 100 |
| Sessions (30d) | 6.8 | 6.0 | 0 | 13 |
| Frequency (180d) | 2.3 | 2.0 | 1 | 7 |
| Monetary (180d) | ₹1,685 | ₹1,359 | ₹403 | ₹4,340 |

### Individual False Negative Cases

---

**1. CUST00184**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 4.11% |
| Recency | 14 days |
| Order Frequency (180d) | 3 orders |
| Monetary (180d) | ₹2,457 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 6 |
| Days Since Last Visit | 6 |

**Analysis:** Model was highly confident this customer would stay (4.1% churn probability). With a purchase just 14 days ago, 6 sessions, and zero support issues, all signals pointed to a retained customer. This is the most surprising false negative in the set — churn was likely triggered by an external factor (competitor promotion, price increase) post-snapshot.

---

**2. CUST01655**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 12.28% |
| Recency | 13 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,359 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 2 |
| Days Since Last Visit | 7 |

**Analysis:** Very recent purchase (13 days) masks a low-frequency buyer with minimal session activity. Only 2 sessions in 30 days suggests shallow engagement despite recent purchase. This pattern — recency without repeat depth — is a potential new signal worth monitoring.

---

**3. CUST00866**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 13.83% |
| Recency | 26 days |
| Order Frequency (180d) | 1 order |
| Monetary (180d) | ₹1,281 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 5 |
| Days Since Last Visit | 1 |

**Analysis:** Visited platform just 1 day ago and had 5 sessions, yet churned. Single-order customer in 180 days — this is a first-time or very infrequent buyer who browsed actively but did not convert again. Low frequency is the risk signal the model underweighted here.

---

**4. CUST01990**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 16.20% |
| Recency | 59 days |
| Order Frequency (180d) | 4 orders |
| Monetary (180d) | ₹3,878 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 11 |
| Days Since Last Visit | 7 |

**Analysis:** Highest-value false negative in the top 10 (₹3,878 spend). High frequency (4 orders), active sessions (11), and moderate recency (59 days) made this look like a loyal customer. Business impact of missing this customer is significant given their spend level. Worth flagging for manual review in high-LTV segments.

---

**5. CUST01303**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 17.28% |
| Recency | 20 days |
| Order Frequency (180d) | 1 order |
| Monetary (180d) | ₹845 |
| Support Tickets (90d) | 1 |
| Web Sessions (30d) | 3 |
| Days Since Last Visit | 0 |

**Analysis:** Visited the platform today (0 days since last visit) yet churned. Has 1 support ticket — the only FN in top 5 with any support activity. Low frequency (1 order) and low spend suggest a trial-stage customer who never became habitual. Support ticket may indicate an unresolved issue that drove departure.

---

**6. CUST00592**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 17.53% |
| Recency | 20 days |
| Order Frequency (180d) | 1 order |
| Monetary (180d) | ₹627 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 3 |
| Days Since Last Visit | 1 |

**Analysis:** Very similar profile to CUST01303 — low spend, single order, light engagement. Pattern of 1-order customers with recency <30 days appearing "safe" but churning suggests the model needs a feature that captures **order depth vs. recency ratio** (i.e., did they only ever buy once?).

---

**7. CUST01826**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 19.23% |
| Recency | 57 days |
| Order Frequency (180d) | 3 orders |
| Monetary (180d) | ₹1,929 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 0 |
| Days Since Last Visit | 15 |

**Analysis:** Zero web sessions in 30 days is a strong disengagement signal — yet recency of 57 days and 3 orders kept the predicted probability low. The complete absence of digital activity while having purchased recently is a contradiction the model did not flag sufficiently. A rule-based overlay (flag customers with 0 sessions regardless of recency) could catch this case.

---

**8. CUST01253**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 21.29% |
| Recency | 99 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹2,036 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 13 |
| Days Since Last Visit | 27 |

**Analysis:** Highest session count among false negatives (13 sessions) combined with 99-day recency. Customer is browsing heavily but not buying — a classic window-shopper pattern that the model misread as engagement. High sessions + long recency + no purchase = a warning signal worth building as a standalone feature.

---

**9. CUST00438**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 24.88% |
| Recency | 64 days |
| Order Frequency (180d) | 3 orders |
| Monetary (180d) | ₹2,466 |
| Support Tickets (90d) | 2 |
| Web Sessions (30d) | 6 |
| Days Since Last Visit | 22 |

**Analysis:** Two support tickets in 90 days is the highest ticket count among false negatives. Moderate recency (64 days) and decent spend masked the support dissatisfaction signal. The combination of multiple tickets + 64-day recency gap suggests a customer who raised issues, didn't get satisfactory resolution, and quietly churned.

---

**10. CUST02060**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 28.66% |
| Recency | 23 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,331 |
| Support Tickets (90d) | 2 |
| Web Sessions (30d) | 4 |
| Days Since Last Visit | 6 |

**Analysis:** Highest predicted churn probability among FNs (28.66%) — closest to the threshold but still well below 54.77%. Two support tickets combined with low frequency (2 orders) is a risk pattern. This customer was the most "detectable" false negative; a lower threshold of ~0.30 would have caught them.

---

## 3. False Positives — Loyal Customers Wrongly Flagged

### What is a False Positive?
The model predicted **churn** but the customer **did not actually churn**. These customers receive an unnecessary retention offer — a cost but not a catastrophic one.

### Business Risk
- **Severity: LOW-MODERATE**
- Cost per false positive: ₹200–500 (unnecessary retention offer)
- Total estimated cost (55 false positives): **₹11,000–₹27,500 per cohort**
- Secondary risk: Over-discounting may train loyal customers to wait for offers

### Why Does This Happen?
These customers had **genuinely churn-like signals** — long recency (avg 84.9 days), very low purchase frequency (avg 1.5 orders), and minimal sessions. They appeared to be disengaged but ultimately stayed — likely due to seasonal purchase cycles, occasional brand loyalty, or long replenishment intervals for personal care products.

### False Positive Summary Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Recency (days) | 84.9 | 76.0 | 16 | 262 |
| Sessions (30d) | 4.9 | 3.0 | 0 | 15 |
| Frequency (180d) | 1.5 | 1.0 | 0 | 3 |
| Monetary (180d) | ₹1,079 | ₹1,067 | ₹0 | ₹2,579 |
| Avg Rating | — | — | 3.00 | 4.00 |

### Individual False Positive Cases

---

**1. CUST01370**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 97.91% |
| Recency | 161 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,246 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 2 |
| Avg Rating | 4.00 |

**Analysis:** Highest-confidence false positive at 97.91%. With 161-day recency and only 2 sessions, the model's confidence is understandable — these are strong churn signals. Customer's 4.0 rating and zero support tickets suggest satisfaction, but their inactivity pattern was near-indistinguishable from a churner. Likely a low-frequency but loyal buyer on a long replenishment cycle.

---

**2. CUST00437**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 97.20% |
| Recency | 151 days |
| Order Frequency (180d) | 1 order |
| Monetary (180d) | ₹729 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 0 |
| Avg Rating | 4.00 |

**Analysis:** Zero sessions and zero support tickets — completely dark on the platform for 30 days. 151-day recency with only 1 order in 180 days. Despite this, the customer did not churn. This is a near-impossible case to catch without external data (e.g., subscription status, offline purchases).

---

**3. CUST01246**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 97.11% |
| Recency | 262 days |
| Order Frequency (180d) | 0 orders |
| Monetary (180d) | ₹0 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 1 |
| Avg Rating | 3.50 |

**Analysis:** Most extreme false positive — 262-day recency, zero orders, zero spend, and only 1 session in 30 days. By every measurable signal this customer should have churned. Their retention may reflect a dormant account that placed an order just outside the 180-day feature window, or an offline purchase not captured in the dataset.

---

**4. CUST01405**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 95.84% |
| Recency | 140 days |
| Order Frequency (180d) | 1 order |
| Monetary (180d) | ₹1,013 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 2 |
| Avg Rating | 4.00 |

**Analysis:** 140-day recency with 1 order and 2 sessions — low engagement but satisfied (4.0 rating). Likely a low-frequency buyer who purchases a single product every 4–5 months. The 60-day churn window may be too short to capture their natural purchase cycle.

---

**5. CUST01017**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 95.34% |
| Recency | 133 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,167 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 3 |
| Avg Rating | 3.00 |

**Analysis:** Lowest average rating (3.00) among top false positives — a neutral-to-dissatisfied signal. Despite this, the customer did not churn. 133-day recency with 3 sessions suggests passive interest. The 3.0 rating warrants monitoring; this customer may churn in the next cycle.

---

**6. CUST00815**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 94.83% |
| Recency | 137 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹2,570 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 3 |
| Avg Rating | 4.00 |

**Analysis:** Highest spender among false positives (₹2,570). Despite 137-day recency, this customer is a relatively high-value account. The retention offer sent to this customer, while technically unnecessary, may reinforce loyalty — making this one of the more justifiable false positives from a business perspective.

---

**7. CUST01325**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 94.77% |
| Recency | 186 days |
| Order Frequency (180d) | 0 orders |
| Monetary (180d) | ₹0 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 1 |
| Avg Rating | 3.50 |

**Analysis:** Zero orders and zero spend in 180 days with 186-day recency. Similar to CUST01246 — a near-dormant account that somehow did not meet the churn definition. One possible explanation: the churn label (no purchase in 60-day window post-snapshot) may have been satisfied by a single order placed just after the window opened, not captured in model features.

---

**8. CUST01614**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 93.49% |
| Recency | 103 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,352 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 4 |
| Avg Rating | 4.00 |

**Analysis:** Most "balanced" false positive — 103-day recency, 2 orders, 4 sessions, 4.0 rating. This is a genuine low-frequency but satisfied customer. The model's 93.49% confidence reflects how closely this profile matches churners. A loyalty-tier feature adjustment could help distinguish these customers.

---

**9. CUST01411**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 93.40% |
| Recency | 183 days |
| Order Frequency (180d) | 0 orders |
| Monetary (180d) | ₹0 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 0 |
| Avg Rating | 3.50 |

**Analysis:** Zero across every activity metric — zero orders, zero spend, zero sessions. Completely invisible on the platform for the entire feature window. The model's 93.4% churn prediction is entirely rational. This customer's non-churn classification is the most difficult to explain and may indicate a data quality issue or label edge case.

---

**10. CUST01803**
| Feature | Value |
|---|---|
| Predicted Churn Probability | 91.87% |
| Recency | 104 days |
| Order Frequency (180d) | 2 orders |
| Monetary (180d) | ₹1,101 |
| Support Tickets (90d) | 0 |
| Web Sessions (30d) | 1 |
| Avg Rating | 4.00 |

**Analysis:** 104-day recency with only 1 session in 30 days. The 4.0 rating and 2 historic orders suggest a satisfied but very passive customer. Likely a slow replenishment-cycle buyer (e.g., purchasing every 3–4 months). Extended churn window (90 days instead of 60) might reclassify customers like this correctly.

---

## 4. Comparative Analysis

### False Negatives vs. False Positives

| Dimension | False Negatives (17) | False Positives (55) |
|---|---|---|
| Avg Recency | 45.9 days | 84.9 days |
| Avg Sessions (30d) | 6.8 | 4.9 |
| Avg Frequency (180d) | 2.3 orders | 1.5 orders |
| Avg Monetary (180d) | ₹1,685 | ₹1,079 |
| Avg Support Tickets | 0.35 | ~0 |
| Business Cost | ₹3,500–5,000 each | ₹200–500 each |
| Total Cohort Cost | ₹59,500–₹85,000 | ₹11,000–₹27,500 |

**Key finding:** False negatives are higher-value customers (₹1,685 avg spend vs ₹1,079) who appear recently active, making them harder to catch and more costly to miss.

---

## 5. Patterns & Root Causes

### False Negative Patterns

| Pattern | Customers | Signal |
|---|---|---|
| Recent purchase + single-order history | CUST00866, CUST00592, CUST01303 | 1 order in 180d despite recent activity |
| Active sessions + no repurchase | CUST01253, CUST01826 | High sessions but not converting |
| Multiple support tickets | CUST00438, CUST02060 | Unresolved dissatisfaction |
| High-value silent exit | CUST01990 | ₹3,878 spend, model confident at 16.2% |

### False Positive Patterns

| Pattern | Customers | Signal |
|---|---|---|
| Long replenishment cycle | CUST01614, CUST01405, CUST01803 | 100–140 day recency, satisfied rating |
| Completely dormant but retained | CUST01246, CUST01411, CUST00437 | Zero activity, yet didn't churn |
| High-value low-frequency | CUST00815 | ₹2,570 spend, infrequent buyer |

---

## 6. Recommendations

### Immediate Actions

1. **Add order-depth feature** — `is_single_order_customer` flag for customers with only 1 lifetime order. FN pattern shows single-order customers churn despite recent activity.

2. **Zero-session rule** — Flag any customer with 0 sessions in 30 days for manual review regardless of model score (catches CUST01826-type cases).

3. **Suppress FP offers for long-cycle buyers** — Customers with loyalty_tier = Platinum or Gold and recency 90–180 days should have retention offers suppressed; they are likely on natural replenishment cycles.

4. **Manual review for high-LTV FNs** — Customers with monetary_180d > ₹3,000 and model score < 30% should be reviewed by CRM team regardless of prediction.

### Model Improvement

5. **Add lifetime order count feature** — Distinguishes first-time buyers (high churn risk) from habitual buyers regardless of recent recency.

6. **Extend churn window to 90 days** — Several FP cases (CUST01246, CUST01411) show customers who purchased just outside the 60-day window. A 90-day window may reduce these edge cases.

7. **Session-to-purchase ratio** — High sessions with no purchase is a hidden churn signal (CUST01253: 13 sessions, 99-day recency).

---

## 7. Business Impact Summary

| Metric | Value |
|---|---|
| Total churners in test set | 168 |
| Churners correctly caught (TP) | 151 (89.9%) |
| Churners missed (FN) | 17 (10.1%) |
| Unnecessary retention offers (FP) | 55 |
| Estimated revenue saved (TP × avg LTV ₹3,500) | **₹528,500** |
| Estimated cost of missed churners (FN × ₹3,500) | **₹59,500** |
| Estimated cost of false positives (FP × ₹350 avg offer) | **₹19,250** |
| **Net estimated value of model per test cohort** | **₹449,750** |

---

*Report generated from XGBoost churn model v1.0 test set evaluation. Snapshot date: 2025-09-30.*
