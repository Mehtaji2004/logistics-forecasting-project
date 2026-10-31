"""Forecast delivery duration and explore simple logistics improvements.

This module creates a reproducible, hypothetical shipment dataset, compares
three model candidates (Linear Regression, Random Forest, Gradient Boosting),
evaluates all on a held-out test set, and produces practical what-if
recommendations. It is intended as an educational project starter; the
generated data is simulated and must not be treated as evidence about a real
logistics operation.

Run from the project directory::

    python logistics_forecasting.py

    # Skip plot generation (useful in headless environments)
    python logistics_forecasting.py --no-plots

The script writes the simulated data, test predictions, feature-importance
table, and a JSON summary to the ``outputs/`` directory by default. Plots (if
enabled and matplotlib is available) are also saved there as PNG files.
See README.md for setup and details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional visualisation dependencies — the script degrades gracefully without them.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend suitable for headless servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

try:
    import openpyxl  # noqa: F401 – imported for side-effects (xlsx writing via pandas)
    _OPENPYXL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENPYXL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

TARGET = "delivery_time_hours"

# Numeric features fed through median imputation (and optional scaling).
NUMERIC_FEATURES = [
    "distance_km",
    "package_weight_kg",
    "order_size_items",
    "traffic_index",
    "weather_severity",
    "warehouse_load_pct",
    "route_complexity",
]

# Categorical features handled with most-frequent imputation + one-hot encoding.
CATEGORICAL_FEATURES = ["service_level", "region", "vehicle_type"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_shipments(rows: int = 1_200, seed: int = 42) -> pd.DataFrame:
    """Create a hypothetical shipment dataset for demonstration.

    The delivery time is generated from a realistic linear formula with
    controlled noise so that regression models have genuine patterns to
    discover. See the inline comments inside the function for the exact
    formula.

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

    # --- Generate raw feature distributions ---
    distance = rng.uniform(5, 600, rows)              # km, uniform over plausible range
    weight = rng.gamma(shape=2.2, scale=4.0, size=rows).clip(0.2, 60)  # kg, right-skewed
    order_size = rng.integers(1, 25, rows)            # count of items in package
    traffic = rng.uniform(0, 1, rows)                 # 0=clear, 1=gridlock
    weather = rng.choice([0, 1, 2, 3], rows, p=[0.55, 0.25, 0.15, 0.05])  # severity 0-3
    warehouse_load = rng.uniform(25, 100, rows)       # % of warehouse capacity in use
    route_complexity = rng.uniform(0, 1, rows)        # composite route-difficulty index
    service = rng.choice(["standard", "express", "economy"], rows, p=[0.55, 0.25, 0.20])
    region = rng.choice(["urban", "suburban", "rural"], rows, p=[0.40, 0.35, 0.25])
    vehicle = rng.choice(["van", "truck", "bike"], rows, p=[0.55, 0.30, 0.15])

    # --- Derive composite adjustments ---
    # Urban areas are faster (more direct routes) but suffer more congestion;
    # rural areas are slower due to distance and road conditions.
    speed_factor = np.select(
        [region == "urban", region == "rural"], [1.2, 0.9], default=1.0
    )
    # Express shipments are prioritised and dispatched sooner; economy adds buffer.
    service_adjustment = np.select(
        [service == "express", service == "economy"], [-1.5, 3.0], default=0.0
    )
    # Heavy trucks are slower due to loading/unloading; bikes are nimble but
    # limited to short distances in the data-generation formula.
    vehicle_adjustment = np.select(
        [vehicle == "truck", vehicle == "bike"], [1.0, 0.5], default=0.0
    )
    # Gaussian noise simulates unmeasured operational variability.
    noise = rng.normal(0, 1.8, rows)

    # The formula gives the model real patterns to learn while retaining noise.
    delivery_time = (
        1.8                                      # base handling time (hours)
        + (distance / 48) * speed_factor        # travel time modulated by region speed
        + weight * 0.035                         # loading / unloading effort
        + order_size * 0.045                     # pick-and-pack time
        + traffic * 3.8                          # congestion delay (biggest driver)
        + weather * 1.25                         # weather-related slowdown
        + (warehouse_load / 100) * 3.2          # dispatch queue effect
        + route_complexity * 2.4                 # winding/complex routes add time
        + service_adjustment                     # SLA priority
        + vehicle_adjustment                     # vehicle throughput difference
        + noise                                  # irreducible operational variability
    ).clip(0.5)  # floor at 30 minutes to avoid nonsensical values

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


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def make_pipeline(model: Any, scale_numeric: bool = False) -> Pipeline:
    """Build a preprocessing and regression pipeline for mixed feature types.

    The numeric branch applies median imputation (robust to outliers) and
    optionally standardises the features (required for linear models, not
    necessary for tree-based ones). The categorical branch applies most-frequent
    imputation followed by one-hot encoding with unknown-category handling.

    Args:
        model: A scikit-learn estimator.
        scale_numeric: Whether to standardize numeric columns. Set ``True``
            for models sensitive to feature magnitude (e.g., LinearRegression).

    Returns:
        A pipeline that imputes missing values, encodes categories, and fits the
        supplied estimator.
    """
    # Build the numeric sub-pipeline conditionally
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # StandardScaler centres and scales each feature to zero mean, unit variance
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)

    # Categorical sub-pipeline: impute then one-hot encode
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric and categorical transforms via ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_models(
    data: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, str, pd.DataFrame]:
    """Fit candidate models and return test metrics, predictions, and winner.

    Three models are compared:

    1. **Linear Regression** — interpretable baseline; scales well; assumes
       linearity and additive feature effects (hyperparameters: none beyond
       preprocessing).
    2. **Random Forest** — ensemble of decision trees; captures non-linearities;
       hyperparameters: ``n_estimators=250``, ``min_samples_leaf=2``.
    3. **Gradient Boosting** — sequential boosting ensemble; often strong on
       tabular data; hyperparameters: ``n_estimators=200``, ``learning_rate=0.1``,
       ``max_depth=4``, ``min_samples_leaf=5``.

    The held-out test set (20 %) is used for the final reported metrics.
    Five-fold cross-validation on the training set guides model selection
    without contaminating the test set.

    Args:
        data: Shipment records containing ``FEATURES`` and ``TARGET`` columns.
        seed: Random seed for reproducible train/test split and CV folds.

    Returns:
        A 5-tuple of:
        - metrics DataFrame (one row per model, sorted by CV MAE ascending)
        - prediction DataFrame (actual vs predicted for the winner)
        - fitted best-model pipeline
        - winner name (str)
        - (x_test, y_test) held-out split stored as a 2-column DataFrame
    """
    # ------------------------------------------------------------------ split
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=seed
    )

    # ------------------------------------------ define candidate models
    candidates = {
        "Linear Regression": make_pipeline(
            LinearRegression(),
            scale_numeric=True,  # linear models need normalised features
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=250,      # number of trees in the forest
                min_samples_leaf=2,    # minimum leaf size to reduce overfitting
                random_state=seed,
                n_jobs=-1,             # use all available CPU cores
            )
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=200,      # number of boosting stages
                learning_rate=0.1,     # shrinkage applied to each tree's contribution
                max_depth=4,           # maximum depth of individual regression trees
                min_samples_leaf=5,    # minimum leaf size for regularisation
                random_state=seed,
            )
        ),
    }

    # ------------------------------------------------ 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    predictions_dict: dict[str, np.ndarray] = {}

    for name, pipeline in candidates.items():
        # CV score on training data only (negative MAE → convert to positive)
        cv_mae = -cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=None,
        )
        # Fit on the full training split for the final test evaluation
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        fitted[name] = pipeline
        predictions_dict[name] = predicted

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

    # Select by CV MAE (lower = better); test metrics serve as a final sanity check.
    metrics = pd.DataFrame(rows).sort_values("cv_mae_mean_hours").reset_index(drop=True)
    winner_name = str(metrics.loc[0, "model"])
    winner = fitted[winner_name]

    # Build prediction frame for the winner
    prediction_frame = pd.DataFrame(
        {
            "actual_delivery_time_hours": y_test.to_numpy(),
            "predicted_delivery_time_hours": predictions_dict[winner_name],
        }
    )
    prediction_frame["absolute_error_hours"] = (
        prediction_frame["actual_delivery_time_hours"]
        - prediction_frame["predicted_delivery_time_hours"]
    ).abs()

    # Store x_test / y_test for downstream importance calculation
    test_split = pd.DataFrame(x_test).copy()
    test_split["__y__"] = y_test.to_numpy()

    return metrics, prediction_frame, winner, winner_name, test_split


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def compute_feature_importance(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, seed: int = 42
) -> pd.DataFrame:
    """Compute permutation feature importance on the held-out test set.

    Permutation importance measures how much a model's score degrades when a
    single feature's values are randomly shuffled. Unlike tree-based impurity
    importance, it is model-agnostic and evaluated on unseen data.

    Args:
        model: A fitted scikit-learn pipeline.
        x_test: Feature matrix for the test set.
        y_test: True target values for the test set.
        seed: Random seed for the permutation shuffles.

    Returns:
        A DataFrame with columns ``feature``, ``importance_mean``, and
        ``importance_std``, sorted by ``importance_mean`` descending.
    """
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=15,          # number of random shuffles per feature
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": -result.importances_mean,   # convert neg-MAE back
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return importance_df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_results(
    metrics: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate and save diagnostic plots for model performance and features.

    Produces two figures:

    1. **Feature Importance** — horizontal bar chart of permutation importance
       (mean ± 1 std) on the test set for the selected model.
    2. **Residual Plot** — scatter of predicted values vs residuals for the
       selected model, with a zero-error reference line.

    If matplotlib is not installed this function is a no-op and returns an
    empty list.

    Args:
        metrics: Metrics DataFrame (used for title annotations).
        prediction_frame: Predicted vs actual values from the best model.
        importance_df: Output of :func:`compute_feature_importance`.
        output_dir: Directory where PNGs are saved.

    Returns:
        List of Path objects for each saved PNG file.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping plots.")
        return []

    saved: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- Figure 1
    # Feature importance bar chart (sorted, with error bars)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        importance_df["feature"],
        importance_df["importance_mean"],
        xerr=importance_df["importance_std"],
        color="steelblue",
        alpha=0.85,
        capsize=4,
    )
    ax.set_xlabel("Mean decrease in MAE (permutation importance)")
    ax.set_title("Feature Importance — Permutation Method (Test Set)")
    ax.invert_yaxis()   # most important at the top
    fig.tight_layout()
    importance_path = output_dir / "feature_importance.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)
    saved.append(importance_path)

    # -------------------------------------------------------------- Figure 2
    # Residual plot: predicted vs residual (actual − predicted)
    predicted = prediction_frame["predicted_delivery_time_hours"]
    residuals = (
        prediction_frame["actual_delivery_time_hours"] - predicted
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, alpha=0.45, s=20, color="darkorange", edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Predicted delivery time (hours)")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residual Plot — Selected Model (Test Set)")
    fig.tight_layout()
    residual_path = output_dir / "residual_plot.png"
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)
    saved.append(residual_path)

    return saved


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def build_recommendations(data: pd.DataFrame, model: Pipeline) -> list[str]:
    """Create transparent operational what-if recommendations from the model.

    The scenarios lower one controllable input at a time relative to a
    representative (median) shipment profile. They estimate association under
    the fitted model; they do not establish causal impact or account for
    implementation cost.

    Additional scenarios compare:
    * **Express vs standard service level** — quantifies the SLA priority effect.
    * **Urban vs rural region routing** — quantifies geographic routing impact.

    Args:
        data: Dataset used to define a representative shipment profile.
        model: Fitted forecasting pipeline.

    Returns:
        Human-readable recommendations ordered by estimated hours saved.
    """
    # Build a representative shipment profile from dataset medians
    baseline = data[FEATURES].median(numeric_only=True).to_dict()
    profile: dict[str, Any] = {
        **baseline,
        "service_level": "standard",
        "region": "urban",
        "vehicle_type": "van",
    }
    base_prediction = float(model.predict(pd.DataFrame([profile]))[0])

    # ---- Controllable operational scenarios ----
    scenarios: list[tuple[str, str, float | str, str | None]] = [
        (
            "Reduce warehouse load from the representative profile by 10 percentage points",
            "warehouse_load_pct",
            -10.0,
            None,
        ),
        (
            "Reduce route complexity by 0.15 through optimised route planning",
            "route_complexity",
            -0.15,
            None,
        ),
        (
            "Reduce traffic exposure by 0.15 through dispatch-time planning",
            "traffic_index",
            -0.15,
            None,
        ),
        # Service-level comparison: standard → express
        (
            "Upgrade service level from standard to express (SLA priority dispatch)",
            "service_level",
            "express",
            "categorical",
        ),
        # Regional routing comparison: urban → rural for long-distance shipments
        (
            "Route long-distance shipments via rural corridors instead of urban routes",
            "region",
            "rural",
            "categorical",
        ),
    ]

    impacts: list[tuple[float, str]] = []
    for description, feature, adjustment, kind in scenarios:
        scenario = profile.copy()
        if kind == "categorical":
            # Replace the categorical value directly
            scenario[feature] = adjustment
        else:
            # Numeric delta — floor at zero to avoid invalid negative feature values
            scenario[feature] = max(0.0, float(scenario[feature]) + float(adjustment))
        scenario_prediction = float(model.predict(pd.DataFrame([scenario]))[0])
        savings = base_prediction - scenario_prediction
        impacts.append((savings, description))

    # Sort descending so the biggest gain is listed first
    impacts.sort(reverse=True)
    return [
        (
            f"{description}; the model estimates about {saved:.2f} fewer delivery hours "
            "for the representative shipment. Validate the change with a controlled pilot "
            "and include cost/capacity constraints."
            if saved >= 0
            else
            f"{description}; note the model estimates {abs(saved):.2f} additional hours "
            "for the representative shipment — this scenario is a trade-off worth "
            "weighing against other operational benefits."
        )
        for saved, description in impacts
    ]


# ---------------------------------------------------------------------------
# Report data helper
# ---------------------------------------------------------------------------


def generate_report_data(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compile all structured artefacts needed for an offline report.

    Args:
        summary: The summary dict written to ``summary.json``.
        metrics: Full metrics DataFrame (all model candidates).
        importance_df: Permutation importance DataFrame.

    Returns:
        A structured dict suitable for serialisation or template rendering.
    """
    return {
        "project_title": summary["project"],
        "selected_model": summary["selected_model"],
        "selected_model_hyperparams": summary.get("best_model_hyperparams", {}),
        "metrics_table": metrics.to_dict(orient="records"),
        "recommendations": summary["recommendations"],
        "feature_importance": importance_df.to_dict(orient="records"),
        "data_note": summary["data_note"],
        "rows": summary["rows"],
        "features": summary["features"],
        "target": summary["target"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_project(
    rows: int, seed: int, output_dir: Path, no_plots: bool = False
) -> dict[str, Any]:
    """Run the end-to-end demo and save all data and model artefacts.

    Steps performed:
    1. Generate synthetic shipment data.
    2. Evaluate three model candidates with cross-validation.
    3. Compute permutation feature importance on the test set.
    4. Build operational what-if recommendations.
    5. Persist CSV, JSON, importance CSV, and optional PNG artefacts.

    Args:
        rows: Number of synthetic shipment records.
        seed: Reproducibility seed.
        output_dir: Directory where all artefacts are written.
        no_plots: If ``True``, skip plot generation even if matplotlib is available.

    Returns:
        The summary object written to ``summary.json``.
    """
    # Step 1 — Data generation
    print("Generating synthetic shipment data...")
    data = generate_shipments(rows=rows, seed=seed)

    # Step 2 — Model evaluation (now returns 5 values including test split info)
    print("Evaluating model candidates (this may take a moment)...")
    metrics, predictions, best_model, winner_name, test_split = evaluate_models(
        data, seed=seed
    )

    # Recover x_test / y_test from the stored split DataFrame
    x_test_df = test_split.drop(columns=["__y__"])
    y_test_series = test_split["__y__"]

    # Step 3 — Feature importance on the test set
    print("Computing permutation feature importance...")
    importance_df = compute_feature_importance(best_model, x_test_df, y_test_series, seed=seed)

    # Step 4 — Write primary artefacts
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "simulated_shipments.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Optionally write an Excel workbook if openpyxl is available
    if _OPENPYXL_AVAILABLE:
        with pd.ExcelWriter(output_dir / "results_summary.xlsx", engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="Metrics", index=False)
            importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)
            predictions.to_excel(writer, sheet_name="Predictions", index=False)

    # Step 5 — Extract hyperparameters of the winning model for the summary
    best_estimator = best_model.named_steps["model"]
    best_hyperparams: dict[str, Any] = {
        k: v for k, v in best_estimator.get_params().items()
        if not k.startswith("_") and v is not None
    }

    # Step 6 — Compile summary JSON
    summary: dict[str, Any] = {
        "project": "Logistics Delivery Time Forecasting and Optimization",
        "data_note": "All records are simulated for demonstration; no real operational data was used.",
        "rows": rows,
        "random_seed": seed,
        "target": TARGET,
        "features": FEATURES,
        "selected_model": winner_name,
        "best_model_hyperparams": best_hyperparams,
        "metrics": metrics.to_dict(orient="records"),
        "recommendations": build_recommendations(data, best_model),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Step 7 — Generate plots (optional)
    if not no_plots:
        print("Generating diagnostic plots...")
        saved_plots = plot_results(metrics, predictions, importance_df, output_dir)
        if saved_plots:
            print(f"  Saved {len(saved_plots)} plot(s): {[p.name for p in saved_plots]}")

    # Step 8 — Produce report data JSON for offline rendering
    report_data = generate_report_data(summary, metrics, importance_df)
    (output_dir / "report_data.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the project demo."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1_200,
        help="number of synthetic shipments to generate (minimum 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducible data and model training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for generated CSV, JSON, and PNG artefacts",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="skip plot generation (useful in headless or CI environments)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the forecasting workflow and print a compact results summary."""
    args = parse_args()
    summary = run_project(args.rows, args.seed, args.output_dir, no_plots=args.no_plots)
    best = summary["metrics"][0]
    print()
    print("=" * 60)
    print(f"  Selected model : {summary['selected_model']}")
    print(
        f"  Test MAE       : {best['test_mae_hours']:.2f} h  |  "
        f"RMSE: {best['test_rmse_hours']:.2f} h  |  "
        f"R²: {best['test_r2']:.3f}"
    )
    print(
        f"  CV MAE (5-fold): {best['cv_mae_mean_hours']:.2f} ± "
        f"{best['cv_mae_std_hours']:.3f} h"
    )
    print("=" * 60)
    print(f"Artefacts saved in : {args.output_dir.resolve()}")
    print(
        "Reminder: this project uses simulated data. "
        "Validate all recommendations with real operational data before deployment."
    )


if __name__ == "__main__":
    main()
