# Logistics Delivery Time Forecasting and Optimization

**Week 4 Assignment — Predictive Modeling and Optimization in Logistics Systems**

A reproducible Python project that generates a synthetic shipment dataset, trains three regression models (Linear Regression, Random Forest, Gradient Boosting), evaluates them rigorously, and derives actionable operational recommendations through model-based what-if analysis.

---

## Assignment Context

This project was developed for the **Week 4 Assignment** of a data analytics / machine learning course focused on predictive modeling in logistics. The assignment requirements addressed in this project include:

- ✅ Well-documented Python script with inline code comments and docstrings
- ✅ Model selection with clear rationale (Linear Regression selected over Random Forest)
- ✅ Evaluation metrics: MAE, RMSE, R² on held-out test set
- ✅ 5-fold cross-validation for stable model selection
- ✅ Optimization strategies: traffic-aware dispatch, warehouse load management, route planning
- ✅ Permutation feature importance analysis
- ✅ Comprehensive written report (`report.md`)
- ✅ Reproducible pipeline using scikit-learn `Pipeline` and `ColumnTransformer`

---

## Project Structure

```
logistics_forecasting_project/
├── logistics_forecasting.py   # Main Python script — all functions documented
├── report.md                  # Full written assignment report (600+ words)
├── report_template.md         # Original report template (retained for reference)
├── requirements.txt           # Python dependency list
├── README.md                  # This file
└── outputs/                   # Generated artefacts (created on first run)
    ├── simulated_shipments.csv     # 1,200 synthetic shipment records
    ├── test_predictions.csv        # Predicted vs actual for the test set
    ├── feature_importance.csv      # Permutation importance scores
    ├── summary.json                # Full model results and recommendations
    ├── report_data.json            # Structured data for offline report rendering
    ├── results_summary.xlsx        # Excel workbook (if openpyxl installed)
    ├── feature_importance.png      # Bar chart of permutation importance
    └── residual_plot.png           # Predicted vs residual scatter plot
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- (Recommended) A virtual environment

### Installation

```bash
# Clone or navigate to the project directory
cd logistics_forecasting_project

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Standard run (generates all artefacts including plots)
python logistics_forecasting.py

# Skip plot generation (headless / CI environments)
python logistics_forecasting.py --no-plots

# Custom dataset size and seed
python logistics_forecasting.py --rows 2000 --seed 123

# Save artefacts to a custom directory
python logistics_forecasting.py --output-dir results/
```

---

## Dataset

### Overview

| Property | Value |
|---|---|
| Records | 1,200 synthetic shipment entries |
| Train / Test split | 80 % / 20 % (960 / 240 records) |
| Random seed | 42 (fully reproducible) |
| Target variable | `delivery_time_hours` |
| Feature count | 10 (7 numeric + 3 categorical) |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `distance_km` | Numeric | Road distance from origin to destination (5–600 km) |
| `package_weight_kg` | Numeric | Gross package weight (0.2–60 kg, γ-distributed) |
| `order_size_items` | Numeric (int) | Number of items in the order (1–24) |
| `traffic_index` | Numeric | Congestion index at dispatch time (0=clear, 1=gridlock) |
| `weather_severity` | Numeric (ordinal) | Weather severity at dispatch (0=clear … 3=severe) |
| `warehouse_load_pct` | Numeric | Warehouse capacity utilisation at dispatch (25–100 %) |
| `route_complexity` | Numeric | Composite route-difficulty score (0=simple, 1=complex) |
| `service_level` | Categorical | Customer SLA tier: standard / express / economy |
| `region` | Categorical | Geographic zone: urban / suburban / rural |
| `vehicle_type` | Categorical | Assigned dispatch vehicle: van / truck / bike |
| `delivery_time_hours` | **Target** | Total elapsed delivery time in hours (≥ 0.5 h) |

---

## Modeling Approach

### Preprocessing Pipeline

All preprocessing is encapsulated in a scikit-learn `Pipeline` with `ColumnTransformer`, preventing data leakage:

- **Numeric features:** Median imputation → StandardScaler (for linear model) or median imputation only (for tree models)
- **Categorical features:** Most-frequent imputation → OneHotEncoder (`handle_unknown="ignore"`)

### Model Candidates

| Model | Key Hyperparameters | Notes |
|-------|-------------------|-------|
| Linear Regression | — (OLS, no regularisation) | Interpretable baseline; scale-sensitive → StandardScaler applied |
| Random Forest | `n_estimators=250`, `min_samples_leaf=2` | Ensemble of trees; captures non-linearity |
| Gradient Boosting | `n_estimators=200`, `lr=0.1`, `max_depth=4` | Sequential boosting; strong on tabular data |

### Model Selection

Models are ranked by **5-fold cross-validation MAE** (computed on training data only). The test set provides a final, uncontaminated evaluation of the selected model.

---

## Results

### Performance Metrics (Test Set)

| Model | MAE (h) | RMSE (h) | R² | CV MAE (h) | CV Std (h) |
|-------|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** ✅ | **1.59** | **1.96** | **0.827** | **1.50** | **0.035** |
| Random Forest | 1.72 | 2.18 | 0.785 | 1.83 | 0.103 |

> **Selected model: Linear Regression**  
> Lower CV MAE (1.50 h vs 1.83 h), much lower CV variance (0.035 h vs 0.103 h), and strong interpretability make it the clear winner on this dataset.

### What the Metrics Mean

- **MAE = 1.59 h** — Average prediction error is ~1 h 35 min. Acceptable for same-day planning windows.
- **RMSE = 1.96 h** — Higher than MAE due to occasional larger errors from noise spikes.
- **R² = 0.827** — The model explains **82.7 %** of variance in delivery time.

---

## Optimization Recommendations

The selected model was used to evaluate five what-if scenarios against a representative (median) shipment profile:

| Recommendation | Estimated Savings | Action |
|---|:---:|---|
| Dispatch-time traffic avoidance (−0.15 traffic index) | **−0.57 h** | Schedule non-urgent shipments during off-peak hours |
| Warehouse load reduction (−10 percentage points) | **−0.34 h** | Stagger dispatch windows to cap load at 75 % |
| Route complexity reduction (−0.15 via route planning) | **−0.31 h** | Deploy algorithmic route optimisation (OR-Tools, OSRM) |
| Upgrade to express service level | −1.50 h | Priority dispatch for time-sensitive shipments |
| Rural corridor routing (long-distance) | *evaluate* | Bypass urban congestion for suitable routes |

**Combined potential savings (top 3 levers): ~1.22 h per shipment (~17 % reduction)**

---

## Feature Importance

Permutation importance (15 repeats, scored on MAE) ranked features as:

1. **`traffic_index`** — dominant delay driver
2. **`distance_km`** — direct travel-time contributor  
3. **`route_complexity`** — winding routes add time independently of distance
4. **`warehouse_load_pct`** — dispatch queue effect during peak periods
5. **`service_level`** — express reduces time; economy adds buffer

---

## Next Steps

- [ ] Replace synthetic data with real TMS/WMS shipment export
- [ ] Add temporal features: hour of day, day of week, holiday flag
- [ ] Explore regularised regression (Ridge/Lasso) for real data
- [ ] Add XGBoost / LightGBM to the candidate set
- [ ] Wrap the model as a REST API (FastAPI) for production use
- [ ] Design A/B pilots to causally validate dispatch-time recommendations

---

## Dependencies

| Package | Minimum Version | Purpose |
|---------|:--------------:|---------|
| `numpy` | 1.24 | Numerical computation and data generation |
| `pandas` | 2.0 | DataFrame manipulation and CSV I/O |
| `scikit-learn` | 1.3 | Pipelines, models, metrics, cross-validation |
| `matplotlib` | 3.7 | Feature importance and residual plots (optional) |
| `openpyxl` | 3.1 | Excel output via pandas ExcelWriter (optional) |

---

## Reproducibility

All results are fully reproducible with `--seed 42` (the default). The random seed is passed to `numpy.random.default_rng`, `train_test_split`, `KFold`, and all model estimators. Running the script with no arguments will always produce identical outputs.

---

*All data is simulated. Validate all recommendations against real operational data before deployment.*
