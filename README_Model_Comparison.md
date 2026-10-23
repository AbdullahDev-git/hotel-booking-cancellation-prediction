# Model Comparison (Copy-Paste for Report)

## Filled Comparison Table

| Model | CV F1 | CV Recall | CV Precision | CV ROC-AUC | Test F1 | Test Recall | Test ROC-AUC |
|-------|-------|-----------|--------------|------------|---------|-------------|--------------|
| **Naive Bayes** (GaussianNB, var_smoothing=1e-3) | 0.577 | 0.791 | 0.454 | 0.793 | 0.58 | 0.80 | 0.799 |
| **Decision Tree** (max_depth=12, class_weight="balanced") | **0.666** | **0.849** | **0.548** | **0.869** | **0.67** | **0.87** | **0.875** |

---

## Winner: **Decision Tree (max_depth=12)**

## Why

The Decision Tree outperforms Naive Bayes on all metrics (CV F1: 0.666 vs 0.577, Test F1: 0.67 vs 0.58). Key reasons:

1. **Feature interactions:** Features are correlated (lead_time ↔ lead_time_group, market_segment ↔ distribution_channel). Naive Bayes assumes independence → violated. Decision Trees naturally model interactions through hierarchical splits.

2. **Non-linear boundaries:** Cancellation risk is U-shaped with lead_time (very short AND very long = higher risk). GaussianNB assumes Gaussian likelihoods → can't capture this. Trees split at multiple thresholds.

3. **Robust to one-hot correlations:** One-hot columns are mutually exclusive; NB's Gaussian assumption breaks, trees handle via binary splits.

4. **Interpretability:** DT provides clear rules (root split on lead_time, then parking/country/agent) useful for hotel managers.

5. **Test-set validation:** CV F1 ≈ Test F1 for both → no overfitting. DT maintains 15% F1 advantage on held-out data.

---

## Key Observations

- Decision Tree is the clear winner for this dataset
- Both models achieve high recall (>0.80) via `class_weight="balanced"` — meets business need to catch cancellations
- Precision remains modest (0.46-0.54) — many false alarms, but acceptable cost for overbooking management
- ROC-AUC gap (0.869 vs 0.793) shows DT is better at *ranking* bookings by cancellation risk
- Feature engineering helped: `has_prev_cancel`, `lead_time_group`, `total_nights` all appear in DT's top 15 features
- **Fairness note:** `country_PRT` (Portuguese guests) is a top-5 feature in DT → Part 4 must examine discrimination risk

---

## Next Steps (Part 3)

1. **Random Forest** — ensemble of trees, reduces variance
2. **XGBoost / LightGBM** — gradient boosting, typically best on tabular data
3. **Hyperparameter tuning** — n_estimators, max_depth, learning_rate, subsample
4. **Threshold tuning** — optimize decision threshold for F1/Recall trade-off
5. **Target:** Test F1 > 0.72, ROC-AUC > 0.90