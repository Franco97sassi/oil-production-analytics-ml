"""Offline training and evaluation for the oil-production model.

The API never trains a model.  Run this module after placing the public CSV at
``data/produccion.csv``; it writes the deployable bundle and auditable reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "produccion.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "modelo_produccion_petroleo.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "prod_pet"
ID_COLUMN = "idpozo"
DATE_COLUMN = "fecha"
LAG_COLUMN = "prod_pet_lag1"
NUMERIC_FEATURES = ["mes", "iny_agua", "iny_gas", "tef", LAG_COLUMN]
CATEGORICAL_FEATURES = [
    "tipoextraccion",
    "tipoestado",
    "tipopozo",
    "provincia",
    "cuenca",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [
    "anio", "mes", ID_COLUMN, TARGET, "iny_agua", "iny_gas", "tef",
    *CATEGORICAL_FEATURES,
]
RANDOM_STATE = 42


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load data and construct a strictly consecutive, well-level monthly lag."""
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    frame = frame.copy()
    frame[DATE_COLUMN] = pd.to_datetime(
        {"year": frame["anio"], "month": frame["mes"], "day": 1},
        errors="coerce",
    )
    frame = frame.dropna(subset=[DATE_COLUMN, ID_COLUMN, TARGET])
    frame = frame.sort_values([ID_COLUMN, DATE_COLUMN])

    previous_date = frame.groupby(ID_COLUMN, sort=False)[DATE_COLUMN].shift(1)
    previous_production = frame.groupby(ID_COLUMN, sort=False)[TARGET].shift(1)
    consecutive = frame[DATE_COLUMN].eq(previous_date + pd.offsets.MonthBegin(1))
    frame[LAG_COLUMN] = previous_production.where(consecutive)

    frame = frame[(frame[TARGET] >= 0) & frame[LAG_COLUMN].notna()].copy()
    if frame.empty:
        raise ValueError("No quedaron observaciones con un mes anterior consecutivo.")
    return frame


def temporal_split(
    frame: pd.DataFrame, test_months: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Reserve the final N observed calendar months as a true future holdout."""
    months = pd.Index(frame[DATE_COLUMN].dropna().sort_values().unique())
    if test_months < 1 or len(months) <= test_months:
        raise ValueError("Se necesitan más meses observados que meses de test.")
    cutoff = pd.Timestamp(months[-test_months])
    train = frame[frame[DATE_COLUMN] < cutoff].copy()
    test = frame[frame[DATE_COLUMN] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("El corte temporal produjo un conjunto vacío.")
    return train, test, cutoff


def _one_hot_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_candidates(include_xgboost: bool = False) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "random_forest": Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=15,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", _ordinal_preprocessor()),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_iter=200, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }
    if include_xgboost:
        if importlib.util.find_spec("xgboost") is None:
            raise RuntimeError("Instalá xgboost para usar --include-xgboost.")
        from xgboost import XGBRegressor

        candidates["xgboost"] = Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return candidates


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


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame = load_and_prepare(args.data)
    train_frame, test_frame, cutoff = temporal_split(frame, args.test_months)
    x_train, y_train = train_frame[FEATURES], train_frame[TARGET]
    x_test, y_test = test_frame[FEATURES], test_frame[TARGET]

    predictions: dict[str, np.ndarray] = {
        "persistencia": x_test[LAG_COLUMN].to_numpy(),
        "media": np.full(len(x_test), float(y_train.mean())),
    }
    fitted: dict[str, Any] = {}
    for name, candidate in build_candidates(args.include_xgboost).items():
        candidate.fit(x_train, y_train)
        fitted[name] = candidate
        predictions[name] = candidate.predict(x_test)

    metrics = {
        name: regression_metrics(y_test, prediction)
        for name, prediction in predictions.items()
    }
    selected_name = min(fitted, key=lambda name: metrics[name]["mae"])
    selected_model = fitted[selected_name]
    selected_prediction = predictions[selected_name]

    residuals = y_test.to_numpy() - selected_prediction
    residual_quantiles = {
        "lower": float(np.quantile(residuals, 0.05)),
        "upper": float(np.quantile(residuals, 0.95)),
        "coverage_target": 0.90,
        "method": "holdout_residual_quantiles",
    }
    known_wells = set(train_frame[ID_COLUMN])
    evaluation = test_frame[
        [DATE_COLUMN, ID_COLUMN, "provincia", "cuenca", TARGET]
    ].rename(columns={TARGET: "real"})
    evaluation["predicha"] = selected_prediction
    evaluation["estado_pozo"] = np.where(
        evaluation[ID_COLUMN].isin(known_wells), "conocido", "nuevo"
    )
    evaluation["rango_produccion"] = pd.cut(
        evaluation["real"],
        bins=[-np.inf, 10, 100, 500, np.inf],
        labels=["0–10", "10–100", "100–500", ">500"],
    )

    report = {
        "selected_model": selected_name,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "train_period": [str(train_frame[DATE_COLUMN].min().date()),
                         str(train_frame[DATE_COLUMN].max().date())],
        "test_period": [str(test_frame[DATE_COLUMN].min().date()),
                        str(test_frame[DATE_COLUMN].max().date())],
        "metrics_by_model": metrics,
        "metrics_by_province": grouped_metrics(evaluation, "provincia"),
        "metrics_by_basin": grouped_metrics(evaluation, "cuenca"),
        "metrics_by_production_range": grouped_metrics(evaluation, "rango_produccion"),
        "metrics_new_vs_known_wells": grouped_metrics(evaluation, "estado_pozo"),
        "prediction_interval": residual_quantiles,
        "drift": drift_report(train_frame, test_frame),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    save_plots(evaluation, args.report_dir)
    if args.with_shap:
        report["shap"] = save_shap_summary(
            selected_model, x_test.sample(min(1000, len(x_test)), random_state=RANDOM_STATE),
            args.report_dir,
        )
    (args.report_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation.to_csv(args.report_dir / "holdout_predictions.csv", index=False)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "residual_quantiles": residual_quantiles,
            "features": FEATURES,
            "metadata": {
                "selected_model": selected_name,
                "trained_until": str(train_frame[DATE_COLUMN].max().date()),
                "test_from": str(cutoff.date()),
            },
        },
        args.model,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report["metrics_by_model"], ensure_ascii=False, indent=2))
    print(f"Modelo seleccionado: {report['selected_model']}")


if __name__ == "__main__":
    main()
