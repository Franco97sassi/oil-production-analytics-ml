from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Oil Production Prediction API",
    description="API para predecir producción mensual de petróleo por pozo",
    version="1.0.0"
)

# Absolute paths based on the project location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_produccion_petroleo.joblib"


def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model_bundle = load_model()


class PredictionInput(BaseModel):
    """Features known before the forecast month begins.

    All operational measurements refer to the previous completed month.
    """
    mes: int = Field(
        ge=1,
        le=12,
        description="Mes calendario de la observación, entre 1 y 12"
    )

    iny_agua_lag1: float = Field(
        ge=0,
        description="Volumen de agua inyectada en el mes anterior"
    )

    iny_gas_lag1: float = Field(
        ge=0,
        description="Volumen de gas inyectado en el mes anterior"
    )

    tef_lag1: float = Field(
        ge=0,
        le=31,
        description="Días efectivos de funcionamiento en el mes anterior"
    )

    tipoextraccion: str = Field(
        min_length=1,
        description="Método de extracción utilizado por el pozo"
    )

    tipoestado: str = Field(
        min_length=1,
        description="Estado operativo del pozo"
    )

    tipopozo: str = Field(
        min_length=1,
        description="Tipo de pozo"
    )

    provincia: str = Field(
        min_length=1,
        description="Provincia donde se encuentra el pozo"
    )

    cuenca: str = Field(
        min_length=1,
        description="Cuenca hidrocarburífera"
    )

    prod_pet_lag1: float = Field(
        ge=0,
        description="Producción de petróleo registrada en el período anterior"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "mes": 10,
                "iny_agua_lag1": 0,
                "iny_gas_lag1": 0,
                "tef_lag1": 31,
                "tipoextraccion": "Bombeo Mecánico",
                "tipoestado": "Extracción Efectiva",
                "tipopozo": "Petrolífero",
                "provincia": "Santa Cruz",
                "cuenca": "GOLFO SAN JORGE",
                "prod_pet_lag1": 50
            }
        }
    }


@app.get("/")
def root():
    return {
        "message": "Oil Production Prediction API",
        "status": "running",
        "model_loaded": model_bundle is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None
    }


@app.get("/model-info")
def model_info():
    """Expose non-sensitive model lineage for operational diagnostics."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    if not isinstance(model_bundle, dict):
        return {"artifact_format": "legacy", "metadata": {}}
    return {
        "artifact_format": "bundle",
        "features": model_bundle.get("features", []),
        "metadata": model_bundle.get("metadata", {}),
        "prediction_interval": model_bundle.get("prediction_interval", {}).get(
            "method"
        ),
    }


def _interval_radius(interval: dict, lag: float) -> float:
    radius = float(interval["global_radius"])
    for group in interval.get("groups", []):
        lower = float("-inf") if group["lower_lag"] is None else group["lower_lag"]
        upper = float("inf") if group["upper_lag"] is None else group["upper_lag"]
        if lower < lag <= upper:
            return float(group["radius"])
    return radius


def _predict_one(data: PredictionInput) -> dict:
    """Run one prediction using either the current or a legacy artifact."""
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. "
                "Ejecutá `python -m src.train` para generarlo."
            ),
        )
    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        prediction_interval = model_bundle.get("prediction_interval")
        residual_quantiles = model_bundle.get("residual_quantiles")
    else:
        model = model_bundle
        prediction_interval = residual_quantiles = None
    prediction = float(model.predict(pd.DataFrame([data.model_dump()]))[0])
    response = {"produccion_predicha": round(max(0.0, prediction), 2)}
    if prediction_interval:
        radius = _interval_radius(prediction_interval, data.prod_pet_lag1)
        response["intervalo_prediccion_90"] = {
            "inferior": round(max(0.0, prediction - radius), 2),
            "superior": round(max(0.0, prediction + radius), 2),
        }
    elif residual_quantiles:
        response["intervalo_prediccion_90"] = {
            "inferior": round(max(0.0, prediction + residual_quantiles["lower"]), 2),
            "superior": round(max(0.0, prediction + residual_quantiles["upper"]), 2),
        }
    return response


@app.post("/predict")
def predict(data: PredictionInput):
    return _predict_one(data)


@app.post("/predict/batch")
def predict_batch(rows: list[PredictionInput]):
    if not rows:
        raise HTTPException(status_code=422, detail="El lote no puede estar vacío.")
    if len(rows) > 1000:
        raise HTTPException(status_code=413, detail="Máximo 1000 observaciones por lote.")
    return {"predicciones": [_predict_one(row) for row in rows]}
