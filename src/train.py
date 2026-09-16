"""Offline orchestration for training, evaluation, and artifact publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    DATE_COLUMN,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PUBLISHED_METRICS_PATH,
    DEFAULT_REPORT_DIR,
    FEATURES,
    ID_COLUMN,
    LAG_COLUMN,
    RANDOM_STATE,
    TARGET,
)
from src.data import load_and_prepare, temporal_validation_split
from src.evaluation import (
    drift_report,
    grouped_metrics,
    regression_metrics,
    save_plots,
    save_shap_summary,
)
from src.modeling import (
    build_candidates,
    calibrate_prediction_intervals,
    interval_radii,
)


def _sha256(path: Path) -> str:
    """Return a stable fingerprint without loading a large dataset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame = load_and_prepare(args.data)
    train_frame, validation_frame, calibration_frame, test_frame, cutoffs = (
        temporal_validation_split(
            frame, args.validation_months, args.calibration_months, args.test_months
        )
    )
    x_train, y_train = train_frame[FEATURES], train_frame[TARGET]
    x_validation = validation_frame[FEATURES]
    y_validation = validation_frame[TARGET]
    x_calibration = calibration_frame[FEATURES]
    y_calibration = calibration_frame[TARGET]
    x_test, y_test = test_frame[FEATURES], test_frame[TARGET]

    validation_predictions: dict[str, np.ndarray] = {
        "persistencia": x_validation[LAG_COLUMN].to_numpy(),
        "media": np.full(len(x_validation), float(y_train.mean())),
    }
    test_predictions: dict[str, np.ndarray] = {
        "persistencia": x_test[LAG_COLUMN].to_numpy(),
        "media": np.full(len(x_test), float(y_train.mean())),
    }
    fitted: dict[str, Any] = {}
    for name, candidate in build_candidates(args.include_xgboost).items():
        candidate.fit(x_train, y_train)
        fitted[name] = candidate
        validation_predictions[name] = candidate.predict(x_validation)

    validation_metrics = {
        name: regression_metrics(y_validation, prediction)
        for name, prediction in validation_predictions.items()
    }
    selected_name = min(fitted, key=lambda name: validation_metrics[name]["mae"])
    development_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    x_development = development_frame[FEATURES]
    y_development = development_frame[TARGET]
    for candidate in fitted.values():
        candidate.fit(x_development, y_development)
    selected_model = fitted[selected_name]
    calibration_prediction = selected_model.predict(x_calibration)
    prediction_interval = calibrate_prediction_intervals(
        y_calibration, calibration_prediction, x_calibration[LAG_COLUMN]
    )
    for name in validation_predictions:
        if name == "persistencia":
            test_predictions[name] = x_test[LAG_COLUMN].to_numpy()
        elif name == "media":
            test_predictions[name] = np.full(len(x_test), float(y_development.mean()))
        else:
            test_predictions[name] = fitted[name].predict(x_test)
    test_metrics = {
        name: regression_metrics(y_test, prediction)
        for name, prediction in test_predictions.items()
    }
    selected_prediction = test_predictions[selected_name]
    radii = interval_radii(x_test[LAG_COLUMN], prediction_interval)
    prediction_interval["test_coverage"] = float(
        np.mean(
            (y_test.to_numpy() >= np.maximum(0, selected_prediction - radii))
            & (y_test.to_numpy() <= selected_prediction + radii)
        )
    )
    known_wells = set(development_frame[ID_COLUMN])
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

    report: dict[str, Any] = {
        "report_schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "filename": args.data.name,
            "sha256": _sha256(args.data),
            "prepared_records": int(len(frame)),
        },
        "selected_model": selected_name,
        "selection_metric": "validation_mae",
        "cutoffs": {
            name: value.strftime("%Y-%m-%d") for name, value in cutoffs.items()
        },
        "train_period": [
            str(train_frame[DATE_COLUMN].min().date()),
            str(train_frame[DATE_COLUMN].max().date()),
        ],
        "validation_period": [
            str(validation_frame[DATE_COLUMN].min().date()),
            str(validation_frame[DATE_COLUMN].max().date()),
        ],
        "calibration_period": [
            str(calibration_frame[DATE_COLUMN].min().date()),
            str(calibration_frame[DATE_COLUMN].max().date()),
        ],
        "test_period": [
            str(test_frame[DATE_COLUMN].min().date()),
            str(test_frame[DATE_COLUMN].max().date()),
        ],
        "records_by_split": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "calibration": int(len(calibration_frame)),
            "test": int(len(test_frame)),
        },
        "validation_metrics_by_model": validation_metrics,
        "metrics_by_model": test_metrics,
        "metrics_by_province": grouped_metrics(evaluation, "provincia"),
        "metrics_by_basin": grouped_metrics(evaluation, "cuenca"),
        "metrics_by_production_range": grouped_metrics(evaluation, "rango_produccion"),
        "metrics_new_vs_known_wells": grouped_metrics(evaluation, "estado_pozo"),
        "prediction_interval": prediction_interval,
        "drift": drift_report(development_frame, test_frame),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    save_plots(evaluation, args.report_dir)
    if args.with_shap:
        report["shap"] = save_shap_summary(
            selected_model,
            x_test.sample(min(1000, len(x_test)), random_state=RANDOM_STATE),
            args.report_dir,
        )
    _write_json(report, args.report_dir / "metrics.json")
    published_metrics = getattr(args, "publish_metrics", None)
    if published_metrics is not None:
        _write_json(report, published_metrics)
    evaluation.to_csv(args.report_dir / "holdout_predictions.csv", index=False)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "artifact_schema_version": 1,
            "model": selected_model,
            "prediction_interval": prediction_interval,
            "features": FEATURES,
            "metadata": {
                "selected_model": selected_name,
                "dataset_sha256": report["dataset"]["sha256"],
                "python_contract": "3.12",
                "scikit_learn_version": sklearn.__version__,
                "trained_until": str(validation_frame[DATE_COLUMN].max().date()),
                "validation_from": str(cutoffs["validation"].date()),
                "calibration_from": str(cutoffs["calibration"].date()),
                "test_from": str(cutoffs["test"].date()),
                "prediction_semantics": (
                    "one_step_ahead_monthly_forecast; every input must be known "
                    "before the forecast month starts"
                ),
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
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--calibration-months", type=int, default=3)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    parser.add_argument(
        "--publish-metrics",
        nargs="?",
        const=DEFAULT_PUBLISHED_METRICS_PATH,
        type=Path,
        help=(
            "Publica el reporte estricto versionable; sin ruta usa "
            "docs/metrics/strict_temporal_metrics.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report["metrics_by_model"], ensure_ascii=False, indent=2))
    print(f"Modelo seleccionado: {report['selected_model']}")


if __name__ == "__main__":
    main()
