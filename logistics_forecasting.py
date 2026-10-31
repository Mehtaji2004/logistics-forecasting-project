"""Forecast delivery duration and explore simple logistics improvements.

This module creates a reproducible, hypothetical shipment dataset, compares a
linear regression baseline with a random forest, evaluates both on a held-out
set, and produces practical what-if recommendations. It is intended as an
educational project starter; the generated data is simulated and must not be
treated as evidence about a real logistics operation.

Run from the project directory::

    python logistics_forecasting.py

The script writes the simulated data, test predictions, and a JSON summary to
the ``outputs/`` directory by default. See README.md for setup and details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "delivery_time_hours"
NUMERIC_FEATURES = [
    "distance_km",
    "package_weight_kg",
    "order_size_items",
    "traffic_index",
    "weather_severity",
    "warehouse_load_pct",
    "route_complexity",
]
CATEGORICAL_FEATURES = ["service_level", "region", "vehicle_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def generate_shipments(rows: int = 1_200, seed: int = 42) -> pd.DataFrame:
    """Create a hypothetical shipment dataset for demonstration.

    Args:
        rows: Number of shipment records to generate. Must be at least 30.
        seed: Random seed for repeatable results.

    Returns:
        A dataframe containing operational features and delivery time in hours.

    Raises:
        ValueError: If fewer than 30 rows are requested.
    """
    if rows < 30:
        raise ValueError("rows must be at least 30 to support a useful split")

    rng = np.random.default_rng(seed)
    distance = rng.uniform(5, 600, rows)
    weight = rng.gamma(shape=2.2, scale=4.0, size=rows).clip(0.2, 60)
    order_size = rng.integers(1, 25, rows)
    traffic = rng.uniform(0, 1, rows)
    weather = rng.choice([0, 1, 2, 3], rows, p=[0.55, 0.25, 0.15, 0.05])
    warehouse_load = rng.uniform(25, 100, rows)
    route_complexity = rng.uniform(0, 1, rows)
    service = rng.choice(["standard", "express", "economy"], rows, p=[0.55, 0.25, 0.20])
    region = rng.choice(["urban", "suburban", "rural"], rows, p=[0.40, 0.35, 0.25])
    vehicle = rng.choice(["van", "truck", "bike"], rows, p=[0.55, 0.30, 0.15])

    speed_factor = np.select([region == "urban", region == "rural"], [1.2, 0.9], default=1.0)
    service_adjustment = np.select(
        [service == "express", service == "economy"], [-1.5, 3.0], default=0.0
    )
    vehicle_adjustment = np.select(
        [vehicle == "truck", vehicle == "bike"], [1.0, 0.5], default=0.0
    )
    noise = rng.normal(0, 1.8, rows)

    # The formula gives the model real patterns to learn while retaining noise.
    delivery_time = (
        1.8
        + (distance / 48) * speed_factor
        + weight * 0.035
        + order_size * 0.045
        + traffic * 3.8
        + weather * 1.25
        + (warehouse_load / 100) * 3.2
        + route_complexity * 2.4
        + service_adjustment
        + vehicle_adjustment
        + noise
    ).clip(0.5)

    return pd.DataFrame(
        {
            "shipment_id": [f"SHP-{i:05d}" for i in range(1, rows + 1)],
            "distance_km": distance.round(2),
            "package_weight_kg": weight.round(2),
            "order_size_items": order_size,
            "traffic_index": traffic.round(3),
            "weather_severity": weather,
            "warehouse_load_pct": warehouse_load.round(1),
            "route_complexity": route_complexity.round(3),
            "service_level": service,
            "region": region,
            "vehicle_type": vehicle,
            TARGET: delivery_time.round(2),
        }
    )


def make_pipeline(model: Any, scale_numeric: bool = False) -> Pipeline:
    """Build a preprocessing and regression pipeline for mixed feature types.

    Args:
        model: A scikit-learn estimator.
        scale_numeric: Whether to standardize numeric columns (for linear model).

    Returns:
        A pipeline that imputes missing values, encodes categories, and fits model.
    """
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_models(data: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Fit candidate models and return test metrics, predictions, and winner.

    The held-out set is used for the final comparison. Five-fold cross
    validation is calculated on training data only to provide a stability
    estimate without using the held-out observations for model selection.

    Args:
        data: Shipment records containing ``FEATURES`` and ``TARGET``.
        seed: Random seed for the split and cross-validation folds.

    Returns:
        A tuple of metrics dataframe, prediction dataframe, and fitted best model.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=seed
    )
    candidates = {
        "Linear Regression": make_pipeline(LinearRegression(), scale_numeric=True),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2, random_state=seed, n_jobs=-1
            )
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, pipeline in candidates.items():
        cv_mae = -cross_val_score(
            pipeline, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=None
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        fitted[name] = pipeline
        predictions[name] = predicted
        rows.append(
            {
                "model": name,
                "test_mae_hours": float(mean_absolute_error(y_test, predicted)),
                "test_rmse_hours": float(np.sqrt(mean_squared_error(y_test, predicted))),
                "test_r2": float(r2_score(y_test, predicted)),
                "cv_mae_mean_hours": float(cv_mae.mean()),
                "cv_mae_std_hours": float(cv_mae.std(ddof=1)),
            }
        )

    # Select by cross-validation on training data; reserve the test set for the
    # final, unbiased comparison of the already-selected candidate.
    metrics = pd.DataFrame(rows).sort_values("cv_mae_mean_hours").reset_index(drop=True)
    winner_name = str(metrics.loc[0, "model"])
    winner = fitted[winner_name]
    prediction_frame = pd.DataFrame(
        {
            "actual_delivery_time_hours": y_test.to_numpy(),
            "predicted_delivery_time_hours": predictions[winner_name],
        }
    )
    prediction_frame["absolute_error_hours"] = (
        prediction_frame["actual_delivery_time_hours"]
        - prediction_frame["predicted_delivery_time_hours"]
    ).abs()
    return metrics, prediction_frame, winner


def build_recommendations(data: pd.DataFrame, model: Pipeline) -> list[str]:
    """Create transparent operational what-if recommendations from the model.

    The scenarios lower one controllable input at a time to its dataset median
    or a modest fixed amount. They estimate association under the fitted model;
    they do not establish causal impact or account for implementation cost.

    Args:
        data: Dataset used to define a representative shipment profile.
        model: Fitted forecasting pipeline.

    Returns:
        Human-readable recommendations ordered by estimated hours saved.
    """
    baseline = data[FEATURES].median(numeric_only=True).to_dict()
    profile: dict[str, Any] = {**baseline, "service_level": "standard", "region": "urban", "vehicle_type": "van"}
    base_prediction = float(model.predict(pd.DataFrame([profile]))[0])
    scenarios = [
        ("Reduce warehouse load from the representative profile by 10 percentage points", "warehouse_load_pct", -10.0),
        ("Reduce route complexity by 0.15 through route planning", "route_complexity", -0.15),
        ("Reduce traffic exposure by 0.15 through dispatch-time planning", "traffic_index", -0.15),
    ]
    impacts: list[tuple[float, str]] = []
    for description, feature, adjustment in scenarios:
        scenario = profile.copy()
        scenario[feature] = max(0, float(scenario[feature]) + adjustment)
        scenario_prediction = float(model.predict(pd.DataFrame([scenario]))[0])
        impacts.append((base_prediction - scenario_prediction, description))

    impacts.sort(reverse=True)
    return [
        f"{description}; the model estimates about {saved:.2f} fewer delivery hours for the representative shipment. Validate the change with a controlled pilot and include cost/capacity constraints."
        for saved, description in impacts
    ]


def run_project(rows: int, seed: int, output_dir: Path) -> dict[str, Any]:
    """Run the end-to-end demo and save its data and model summary.

    Args:
        rows: Number of synthetic shipment records.
        seed: Reproducibility seed.
        output_dir: Directory where CSV and JSON artifacts are written.

    Returns:
        The summary object written to ``summary.json``.
    """
    data = generate_shipments(rows=rows, seed=seed)
    metrics, predictions, best_model = evaluate_models(data, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "simulated_shipments.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    summary: dict[str, Any] = {
        "project": "Logistics Delivery Time Forecasting and Optimization",
        "data_note": "All records are simulated for demonstration; no real operational data was used.",
        "rows": rows,
        "random_seed": seed,
        "target": TARGET,
        "features": FEATURES,
        "selected_model": str(metrics.loc[0, "model"]),
        "metrics": metrics.to_dict(orient="records"),
        "recommendations": build_recommendations(data, best_model),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the project demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_200, help="number of synthetic shipments (minimum 30)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible data and split")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="directory for generated CSV and JSON files")
    return parser.parse_args()


def main() -> None:
    """Run the forecasting workflow and print a compact results summary."""
    args = parse_args()
    summary = run_project(args.rows, args.seed, args.output_dir)
    best = summary["metrics"][0]
    print(f"Selected model: {summary['selected_model']}")
    print(
        "Test performance: "
        f"MAE={best['test_mae_hours']:.2f} h, "
        f"RMSE={best['test_rmse_hours']:.2f} h, R²={best['test_r2']:.3f}"
    )
    print(f"Artifacts saved in: {args.output_dir.resolve()}")
    print("Reminder: this project uses simulated data; validate recommendations with real operational data.")


if __name__ == "__main__":
    main()
