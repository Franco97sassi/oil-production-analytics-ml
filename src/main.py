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


modelo = load_model()


class PredictionInput(BaseModel):
    mes: int = Field(
        ge=1,
        le=12,
        description="Mes calendario de la observación, entre 1 y 12"
    )

    iny_agua: float = Field(
        ge=0,
        description="Volumen mensual de agua inyectada"
    )

    iny_gas: float = Field(
        ge=0,
        description="Volumen mensual de gas inyectado"
    )

    tef: float = Field(
        ge=0,
        le=31,
        description="Tiempo efectivo de funcionamiento del pozo, expresado en días"
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
                "iny_agua": 0,
                "iny_gas": 0,
                "tef": 31,
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
        "model_loaded": modelo is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": modelo is not None
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. "
                "Ejecutá `python -m src.train` para generarlo."
            )
        )

    entrada = pd.DataFrame([data.model_dump()])

    prediccion = modelo.predict(entrada)[0]

    return {
        "produccion_predicha": round(float(prediccion), 2)
    }