"""Evaluation metrics, drift diagnostics, and generated visualizations."""
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "registros": int(len(actual)),
    }


def grouped_metrics(
    evaluation: pd.DataFrame, group: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, part in evaluation.groupby(group, observed=True, dropna=False):
        metrics = regression_metrics(part["real"], part["predicha"].to_numpy())
        rows.append({group: str(value), **metrics})
    return rows


def population_stability_index(train: pd.Series, test: pd.Series) -> float:
    """Compute numeric PSI with bins learned only from training data."""
    train = pd.to_numeric(train, errors="coerce").dropna()
    test = pd.to_numeric(test, errors="coerce").dropna()
    if train.empty or test.empty:
        return float("nan")
    edges = np.unique(train.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_share = pd.cut(train, edges, include_lowest=True).value_counts(normalize=True)
    test_share = pd.cut(test, edges, include_lowest=True).value_counts(normalize=True)
    train_share, test_share = train_share.align(test_share, fill_value=0)
    train_share = train_share.clip(lower=1e-6)
    test_share = test_share.clip(lower=1e-6)
    return float(((test_share - train_share) * np.log(test_share / train_share)).sum())


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    numeric = {
        column: population_stability_index(train[column], test[column])
        for column in NUMERIC_FEATURES
    }
    categorical: dict[str, float] = {}
    for column in CATEGORICAL_FEATURES:
        left = train[column].fillna("<NA>").value_counts(normalize=True)
        right = test[column].fillna("<NA>").value_counts(normalize=True)
        left, right = left.align(right, fill_value=0)
        categorical[column] = float(0.5 * (left - right).abs().sum())
    return {"numeric_psi": numeric, "categorical_total_variation": categorical}


def save_plots(evaluation: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    sample = evaluation.sample(min(5000, len(evaluation)), random_state=RANDOM_STATE)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(sample["real"], sample["predicha"], alpha=0.25, s=10)
    limit = float(max(sample["real"].max(), sample["predicha"].max()))
    axis.plot([0, limit], [0, limit], "--", color="black")
    axis.set(xlabel="Producción real (m³)", ylabel="Producción predicha (m³)",
             title="Producción real vs. predicha — holdout temporal")
    figure.tight_layout()
    figure.savefig(report_dir / "model_real_vs_predicted.png", dpi=160)
    plt.close(figure)

    monthly = evaluation.groupby("fecha", observed=True)[["real", "predicha"]].sum()
    figure, axis = plt.subplots(figsize=(9, 5))
    monthly.plot(ax=axis, marker="o")
    axis.set(xlabel="Mes", ylabel="Producción (m³)",
             title="Producción mensual en el holdout")
    figure.tight_layout()
    figure.savefig(report_dir / "model_monthly_holdout.png", dpi=160)
    plt.close(figure)


def save_shap_summary(model: Pipeline, sample: pd.DataFrame, report_dir: Path) -> str:
    if importlib.util.find_spec("shap") is None:
        return "omitido: instalá shap y ejecutá con --with-shap"
    import shap

    transformed = model.named_steps["preprocessor"].transform(sample)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    explainer = shap.TreeExplainer(model.named_steps["regressor"])
    values = explainer.shap_values(transformed)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(report_dir / "shap_importance.csv", index=False)
    return "reports/shap_importance.csv"


