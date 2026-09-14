from fastapi.testclient import TestClient

import src.main as main


client = TestClient(main.app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Oil Production Prediction API"
    assert data["status"] == "running"
    assert "model_loaded" in data


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_without_model(monkeypatch):
    monkeypatch.setattr(main, "model_bundle", None)

    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert "modelo no está disponible" in response.json()["detail"].lower()


def test_predict_success(monkeypatch):
    class FakeModel:
        def predict(self, data):
            assert list(data.columns) == list(payload)
            return [54.916]

    payload = {
        "mes": 10,
        "iny_agua_lag1": 0,
        "iny_gas_lag1": 0,
        "tef_lag1": 31,
        "tipoextraccion": "Bombeo Mecánico",
        "tipoestado": "Extracción Efectiva",
        "tipopozo": "Petrolífero",
        "provincia": "Santa Cruz",
        "cuenca": "GOLFO SAN JORGE",
        "prod_pet_lag1": 50,
    }
    monkeypatch.setattr(
        main,
        "model_bundle",
        {
            "model": FakeModel(),
            "residual_quantiles": {"lower": -4.0, "upper": 6.0},
        },
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "produccion_predicha": 54.92,
        "intervalo_prediccion_90": {"inferior": 50.92, "superior": 60.92},
    }


def test_predict_uses_conditional_conformal_interval(monkeypatch):
    class FakeModel:
        def predict(self, data):
            return [50]

    monkeypatch.setattr(
        main,
        "model_bundle",
        {
            "model": FakeModel(),
            "prediction_interval": {
                "global_radius": 10,
                "groups": [
                    {"lower_lag": None, "upper_lag": 100, "radius": 3}
                ],
            },
        },
    )
    payload = {
        "mes": 10, "iny_agua_lag1": 0, "iny_gas_lag1": 0, "tef_lag1": 31,
        "tipoextraccion": "BM", "tipoestado": "activo",
        "tipopozo": "petrolifero", "provincia": "Santa Cruz",
        "cuenca": "Golfo San Jorge", "prod_pet_lag1": 50,
    }

    response = client.post("/predict", json=payload)

    assert response.json()["intervalo_prediccion_90"] == {
        "inferior": 47.0, "superior": 53.0
    }


def test_batch_prediction_and_model_info(monkeypatch):
    class FakeModel:
        def predict(self, data):
            return [12]

    monkeypatch.setattr(
        main,
        "model_bundle",
        {
            "model": FakeModel(),
            "features": ["mes"],
            "metadata": {"selected_model": "test-model"},
            "prediction_interval": {"method": "split-conformal", "global_radius": 2},
        },
    )
    payload = {
        "mes": 10, "iny_agua_lag1": 0, "iny_gas_lag1": 0, "tef_lag1": 31,
        "tipoextraccion": "BM", "tipoestado": "activo",
        "tipopozo": "petrolifero", "provincia": "Santa Cruz",
        "cuenca": "Golfo San Jorge", "prod_pet_lag1": 50,
    }

    batch = client.post("/predict/batch", json=[payload, payload])
    info = client.get("/model-info")

    assert batch.status_code == 200
    assert len(batch.json()["predicciones"]) == 2
    assert info.json()["metadata"]["selected_model"] == "test-model"


def test_batch_rejects_an_empty_request():
    response = client.post("/predict/batch", json=[])

    assert response.status_code == 422
