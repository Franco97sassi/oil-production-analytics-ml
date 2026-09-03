from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Oil Production Prediction API",
    description="API para predecir producción mensual de petróleo por pozo",
    version="1.0.0"
)

MODEL_PATH = Path("models/modelo_produccion_petroleo.joblib")
modelo = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    mes: int
    iny_agua: float
    iny_gas: float
    tef: float
    tipoextraccion: str
    tipoestado: str
    tipopozo: str
    provincia: str
    cuenca: str
    prod_pet_lag1: float


@app.get("/")
def root():
    return {
        "message": "Oil Production Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    entrada = pd.DataFrame([data.model_dump()])

    prediccion = modelo.predict(entrada)[0]

    return {
        "produccion_predicha": round(float(prediccion), 2)
    }