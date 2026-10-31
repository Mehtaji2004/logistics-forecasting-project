# Logistics Delivery Time Forecasting and Optimization

An educational Python project for predicting shipment delivery duration and turning model insights into logistics what-if recommendations. The assignment brief asks for a defined forecasting problem, documented Python workflow, model comparison, evaluation, and operational optimization ideas.

> **Data note:** This starter generates a hypothetical dataset. Its model scores and recommendations are examples only; they do not describe a real carrier or prove that an operational change will cause the predicted improvement.

## Problem statement

Predict **delivery time in hours** from shipment and operating conditions. The intended use is to flag shipments that may need attention and to help operations teams explore dispatch, route-planning, and warehouse-capacity scenarios.

## Dataset

`logistics_forecasting.py` creates a reproducible synthetic dataset with these inputs:

| Feature | Description |
| --- | --- |
| `distance_km` | Shipment distance in kilometers |
| `package_weight_kg` | Package weight |
| `order_size_items` | Number of items in the order |
| `traffic_index` | Simulated traffic level from 0 to 1 |
| `weather_severity` | Simulated weather category from 0 (clear) to 3 (severe) |
| `warehouse_load_pct` | Warehouse utilization percentage |
| `route_complexity` | Simulated route difficulty from 0 to 1 |
| `service_level` | Standard, express, or economy |
| `region` | Urban, suburban, or rural |
| `vehicle_type` | Van, truck, or bike |
| `delivery_time_hours` | **Target** to forecast |

The synthetic target is generated from these factors plus random noise. The code documents the assumptions in `generate_shipments()`.

## Project files

- `logistics_forecasting.py` — data generation, preprocessing, training, evaluation, and what-if recommendations. Functions and the module include docstrings.
- `requirements.txt` — Python package dependencies.
- `report_template.md` — a report outline for documenting the completed assignment. Replace placeholders with results from your run.
- `.vscode/launch.json` — run the script from VS Code with the default settings.
- `outputs/` — generated CSV and JSON artifacts (created when you run the program).

## Set up in VS Code

1. Open this project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies: `python -m pip install -r requirements.txt`.
5. In VS Code, select the `.venv` Python interpreter if it is not selected automatically.
6. Open `logistics_forecasting.py` and click **Run Python File**, or use the run command below.

## Run

```bash
python logistics_forecasting.py
```

Optional arguments:

```bash
python logistics_forecasting.py --rows 2000 --seed 7 --output-dir outputs
```

The run saves:

- `outputs/simulated_shipments.csv` — generated sample records
- `outputs/test_predictions.csv` — actual and predicted delivery times for the held-out split
- `outputs/summary.json` — model metrics and representative what-if recommendations

## Modeling approach

The workflow holds out 20% of the generated rows for final evaluation. It compares:

1. **Linear Regression** as a simple, interpretable baseline.
2. **Random Forest Regressor** to capture nonlinear patterns and feature interactions.

Numeric inputs are imputed and scaled for the linear model. Categories are imputed and one-hot encoded. Five-fold cross-validation on the training data reports mean and standard deviation of MAE and selects the candidate. The held-out test set is reserved for the final comparison and reported performance.

Reported measures:

- **MAE (hours):** typical absolute forecast error; lower is better.
- **RMSE (hours):** gives larger errors more weight; lower is better.
- **R²:** variance explained relative to a mean-only baseline; closer to 1 is generally better.

## Optimization discussion

The script compares a representative shipment with scenarios that reduce warehouse load, route complexity, or traffic exposure individually. These are model-based what-if estimates, not causal findings. A real implementation should account for labor, vehicle capacity, service-level commitments, and cost, then validate the change through a pilot before rollout.

Possible next steps with real data include time-based validation, tuning model parameters, checking forecast errors by region/service level, comparing cost per on-time shipment, and solving a constrained vehicle or route allocation problem.

## Limitations

- The generated data is artificial and deliberately simplified.
- The random train/test split assumes records are independent; real shipment data may require a chronological split to prevent future information leaking into evaluation.
- Feature relationships in the sample were chosen by the author, so performance scores only show that the workflow runs on the simulated pattern.
- The what-if scenarios do not optimize a fleet schedule or prove causal impact.
