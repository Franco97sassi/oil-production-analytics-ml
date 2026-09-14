from fastapi.testclient import TestClient

import src.main as main


client = TestClient(main.app)


VALID_PAYLOAD = {
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


def test_missing_required_field():
    payload = VALID_PAYLOAD.copy()
    payload.pop("provincia")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_month_type():
    payload = VALID_PAYLOAD.copy()
    payload["mes"] = "octubre"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_empty_province():
    payload = VALID_PAYLOAD.copy()
    payload["provincia"] = ""

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
