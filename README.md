# Predicting Hotel Booking Cancellations
**CS-13410 Introduction to Machine Learning: Semester Project (Fall 2026)**

**Dataset:** Hotel Booking Demand (Antonio, de Almeida & Nunes, 2019)
**Task:** Binary classification — predict whether a hotel booking is cancelled
**Group:** Muhammad Abdullah Noor (70146205), Mehrab Fiaz (70148905), Muhammad Ahmed (70149644)

## Files
- `Notebook.ipynb` — main notebook (Part 1 + Part 2)
- `hotel_bookings.csv` — dataset
- `Project_Proposal.docx` — one-page proposal
- `requirements.txt` — Python dependencies
- `part1_pipeline.py` — reusable leakage-free preprocessing pipeline
- `abdullah_decision_tree.py` — standalone Decision Tree exploration script
- `tree_best_params.json` — best Decision Tree hyperparameters (`{"max_depth": 12}`)
- `ML Project.docx` — official project brief

## How to Run
```bash
pip install -r requirements.txt
```
Open `Notebook.ipynb` and click **Run All** (keep `hotel_bookings.csv` in the same folder).

## Progress
- [x] Part 1: Data preparation & leakage-free pipeline
- [x] Part 2: Naive Bayes vs Decision Tree
- [ ] Part 3: Ensembles & tuning
- [ ] Part 4: Neural network, clustering, fairness
- [ ] Part 5: Final report & viva

---

## Part 1 Results Summary

**Data cleaning:**
- Raw rows: 119,390 → After removing duplicates: 87,396 → After removing zero-guest bookings: 87,230
- Duplicate rows removed: 31,994 (26.8%)
- Zero-guest bookings removed: 166
- Cancellation rate after cleaning: **27.5%**

**Target leakage removed:**
- `reservation_status` (perfect proxy: Canceled/No-Show = 1, Check-Out = 0)
- `reservation_status_date` (cancellation/check-out date)
- `assigned_room_type` (assigned at check-in)
- `company` (94.3% missing)
- `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month` (temporal leakage)

**Feature engineering (5 new features):**
- `total_nights` = weekend + week nights
- `total_guests` = adults + children + babies
- `has_children` = binary flag (children > 0 or babies > 0)
- `has_prev_cancel` = binary flag (previous_cancellations > 0)
- `lead_time_group` = 5 buckets (0-7, 8-30, 31-90, 91-180, 181+ days)

**Leakage-free pipeline:**
- `Pipeline` + `ColumnTransformer` with per-column treatment
- Numeric (wide-range): median impute → 1%/99% percentile capping → StandardScaler
- Numeric (zero-inflated counts): median impute → log1p → StandardScaler
- Binary: passthrough
- Categorical: "Missing" impute → OneHotEncoder (min_frequency=0.01, drop="if_binary")
- Feature selection: SelectKBest (ANOVA F-test, k=30)
- Verified: test set scaled using training statistics only

**Class imbalance handling:**
- Compared: No handling vs Class weights vs SMOTE (5-fold stratified CV, Logistic Regression baseline)
- **Chosen: `class_weight="balanced"`** (simplest, performs same as SMOTE)
- CV F1: 0.658, Recall: 0.823, ROC-AUC: 0.86

**Final Part 1 model (Logistic Regression + class weights) on test set:**
- F1: 0.66, Recall: 0.83, Precision: 0.55, ROC-AUC: 0.864

---

## Part 2 Results Summary

### Naive Bayes (Ahmed's section)
- **Model:** GaussianNB with Part 1 preprocessing pipeline
- **Hyperparameter tuning:** `var_smoothing` ∈ [1e-9, 1e-4, 1e-3, 1e-2, 1e-1, 1.0] × priors ∈ [default, balanced]
- **Best setting:** `var_smoothing = 1e-3`, default priors (learned from data: 72.5% not cancelled, 27.5% cancelled)

**5-fold CV (train set):**
| Metric | Score |
|--------|-------|
| F1 | **0.577** |
| Recall | **0.791** |
| Precision | **0.454** |
| ROC-AUC | **0.793** |

**Test set (single evaluation):**
| Metric | Score |
|--------|-------|
| F1 | **0.58** |
| Recall | **0.80** |
| Precision | **0.46** |
| ROC-AUC | **0.799** |

**Information-gain hand-worked example:**
- 12-bookings sample (5 cancelled, 7 not cancelled, random_state=55)
- Three candidate splits evaluated by entropy:
  - `long_lead` (lead_time > 90 days): IG = 0.1686
  - `has_request` (special requests > 0): IG = 0.0006
  - `online_ta` (market_segment = Online TA): **IG = 0.3436** ← **WINNER**
- `online_ta` creates one pure branch (4 non-agency bookings all stayed)
- Verified: scikit-learn entropy tree also picks `online_ta` as root split

---

### Decision Tree (Abdullah's section)
- **Model:** DecisionTreeClassifier with Part 1 preprocessing pipeline
- **Hyperparameter tuning:** `max_depth` ∈ [3, 5, 8, 12, None], `class_weight="balanced"`

**5-fold CV (train set):**
| max_depth | Train F1 | CV F1 | CV F1 std | CV ROC-AUC |
|-----------|----------|-------|-----------|------------|
| 3 | 0.541 | 0.541 | 0.002 | 0.712 |
| 5 | 0.616 | 0.617 | 0.002 | 0.832 |
| 8 | 0.659 | 0.653 | 0.002 | 0.870 |
| **12** | **0.701** | **0.666** | **0.001** | **0.869** |
| None | 0.990 | 0.598 | 0.004 | 0.723 |

**Best depth: 12** (highest CV F1 = 0.666, minimal overfitting)

**Root split:** `lead_time <= -0.847` (scaled → short lead times)

**Top 5 feature importances:**
1. `lead_time` (scaled)
2. `market_segment_Online TA` / `distribution_channel_TA/TO`
3. `country_PRT` (Portuguese guests) → **fairness concern for Part 4**
4. `agent_agent_9` (specific travel agent)
5. `required_car_parking_spaces`

**Strong feature correlations (|r| > 0.8):**
- `deposit_type_No Deposit` ↔ `deposit_type_Non Refund`: -0.953
- `stays_in_week_nights` ↔ `total_nights`: 0.941
- `market_segment_Direct` ↔ `distribution_channel_Direct`: 0.918
- `previous_cancellations` ↔ `has_prev_cancel`: 0.905

---

### Model Comparison (Mehrab's section)

**Cross-Validation (5-fold stratified, train set only):**
| Model | CV F1 | CV Recall | CV Precision | CV ROC-AUC |
|-------|-------|-----------|--------------|------------|
| Naive Bayes | 0.577 | 0.791 | 0.454 | 0.793 |
| **Decision Tree** | **0.666** | **0.849** | **0.548** | **0.869** |

**Test Set (held-out, evaluated once):**
| Model | Test F1 | Test Recall | Test Precision | Test ROC-AUC |
|-------|---------|-------------|----------------|--------------|
| Naive Bayes | 0.58 | 0.80 | 0.46 | 0.799 |
| **Decision Tree** | **0.67** | **0.87** | **0.54** | **0.875** |

---

## Key Conclusions (Part 2)

1. **Decision Tree (max_depth=12) decisively outperforms Naive Bayes** on all metrics
   - CV F1 advantage: +0.089 (15% relative)
   - Test F1 advantage: +0.09
   - ROC-AUC advantage: +0.076

2. **Why Decision Tree wins:**
   - Handles feature interactions (correlated features: lead_time/lead_time_group, market_segment/distribution_channel)
   - Captures non-linear boundaries (U-shaped lead_time effect)
   - Robust to one-hot encoding correlations
   - Provides interpretable rules for business stakeholders

3. **Both models achieve high recall (>0.80)** due to `class_weight="balanced"` — meets business requirement to catch most cancellations

4. **No overfitting to CV:** Test F1 ≈ CV F1 for both models (NB: 0.577→0.58, DT: 0.666→0.67)

5. **Precision remains modest (0.46-0.54)** — many false alarms (unnecessary deposit requests), but acceptable cost for high recall

6. **Fairness flag:** `country_PRT` (Portuguese guests) is a top-5 feature → Part 4 must examine discrimination risk

---

## Next Steps (Part 3)
- Random Forest, AdaBoost, XGBoost/LightGBM
- Hyperparameter optimization (grid search / Optuna)
- Ridge/Lasso/ElasticNet for linear baseline
- Target: Test F1 > 0.72, ROC-AUC > 0.90