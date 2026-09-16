# 🛢️ Oil Production Analytics & Machine Learning

[![CI](https://github.com/francosassi/oil-production-analytics-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/francosassi/oil-production-analytics-ml/actions/workflows/ci.yml)

End-to-end **Data Analytics and Machine Learning project** focused on analyzing and predicting monthly oil production using public hydrocarbon production data from Argentina.

The project covers the complete workflow from exploratory data analysis and feature engineering to temporal validation, model evaluation, serialization, and deployment through a REST API.

> **Portfolio status:** the code, automated tests and historical notebook
> benchmark are reproducible. The definitive strict-temporal metrics and visual
> demo still require the complete official dataset and Power BI screenshots.
> Follow the owner checklist in [`docs/PORTFOLIO_GUIDE.md`](docs/PORTFOLIO_GUIDE.md)
> before sending the project to recruiters.

## ✅ Verified and auditable results

- The historical notebook benchmark is automatically checked against its saved
  outputs in CI; it is not presented as the definitive strict-temporal result.
- The production training command records the dataset SHA-256, exact temporal
  periods, split sizes, baselines, segmented errors, drift and conformal
  coverage in one machine-readable report.
- The strict result will be displayed here after training on the complete
  official dataset with `python -m src.train --publish-metrics`.

## 🎬 Demo

- **API:** train the model, run `docker compose up --build`, and open
  [Swagger UI](http://localhost:8000/docs).
- **Power BI:** screenshots and the short portfolio walkthrough are pending the
  owner steps documented in [`docs/PORTFOLIO_GUIDE.md`](docs/PORTFOLIO_GUIDE.md).
- **Health:** liveness is available at `/health/live`; readiness at
  `/health/ready` returns HTTP 503 until a valid model artifact is mounted.

---

## 📌 Project Overview

The objective of this project is to analyze historical oil well production data and build a Machine Learning model capable of estimating monthly oil production.

The original dataset contains approximately **1 million production records** with information about wells, production, injection, extraction methods, geographic location, and operational characteristics.

The project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and profiling
- Data visualization
- Feature selection
- Categorical encoding
- Feature engineering
- Lag features
- Machine Learning regression
- Baseline comparison
- Temporal validation
- Error analysis
- Feature importance analysis
- Model serialization
- REST API with FastAPI

---

## 📊 Data Analytics

The exploratory analysis was performed using **Pandas, NumPy and Matplotlib**.

Some of the analyses include:

- Oil production by province
- Monthly production evolution
- Production distribution
- Missing value analysis
- Duplicate detection
- Well-level analysis
- Production statistics
- Operational and geographical characteristics

The original dataset contains:

- **~991,000 records**
- **38 variables**
- **83,000+ well identifiers**
- Multiple producing provinces and hydrocarbon basins

The analysis showed a strong concentration of oil production in provinces such as **Neuquén, Chubut and Santa Cruz**.

---

## 🤖 Machine Learning

The target variable is:

```text
prod_pet
```

which represents monthly oil production.

The initial model used operational, geographical and well characteristics such as:

```text
mes
iny_agua_lag1
iny_gas_lag1
tef_lag1
tipoextraccion
tipoestado
tipopozo
provincia
cuenca
```

Categorical variables are transformed using:

```text
OneHotEncoder
```

The preprocessing and model are combined using a **scikit-learn Pipeline** and `ColumnTransformer`.

---

## 🧠 Feature Engineering

Exploratory modeling showed that static and operational characteristics alone were not sufficient to accurately represent the temporal behavior of individual wells.

A historical feature was therefore introduced:

```text
prod_pet_lag1
```

This represents the oil production of the same well during the **previous month**.

Example:

| Month | Production | Previous Month Production |
|---:|---:|---:|
| 1 | 100 | N/A |
| 2 | 95 | 100 |
| 3 | 91 | 95 |
| 4 | 87 | 91 |

This feature significantly improved predictive performance.

`idpozo` is used to construct the historical feature but is **not used directly as a numerical model feature**.

---

## ⏱️ Temporal Validation

Instead of a random split, the reproducible training script builds a complete
date from `anio` and `mes`, orders observations by well and date, and creates
four disjoint chronological periods. The final three observed calendar months
are a **future temporal holdout** that is never used to choose the model or
calibrate its interval.

```text
Historical period          → Model fitting
Next 3 calendar months     → Model selection by validation MAE
Next 3 calendar months     → Conformal interval calibration
Final 3 calendar months    → Untouched final test
```

This prevents records from a later year entering training merely because their
calendar month is between January and September. A lag is retained only when
the previous observation for that well is exactly one calendar month earlier.

### Prediction contract

The model represents a **one-step-ahead forecast made before the target month
starts**. Production, injection volumes and effective operating time all refer
to the previous completed month (`*_lag1`); the target month's operational
measurements are never model inputs. This prevents availability leakage. The API
descriptions and model metadata make this contract explicit.

### Prediction intervals

The selected model is calibrated on a period that is separate from validation
and final testing. The 90% intervals use finite-sample split conformal absolute
residuals. They are conditioned on quartiles of `prod_pet_lag1` when each group
has enough calibration observations, with a global conformal radius as a safe
fallback. `reports/metrics.json` records calibration size and final-test
coverage so the nominal 90% target can be audited rather than assumed.

---

## 🌲 Model

The original notebook model is a:

**Random Forest Regressor**

Main configuration:

```python
RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

---

## 📈 Model Performance

### Historical notebook benchmark: model without production history

| Metric | Result |
|---|---:|
| MAE | 142.71 |
| RMSE | 470.91 |
| R² | 0.4549 |

### Historical notebook benchmark: model with `prod_pet_lag1`

| Metric | Result |
|---|---:|
| **MAE** | **44.46** |
| **RMSE** | **241.10** |
| **R²** | **0.8562** |

Adding historical production substantially improved model performance.

### Performance by test month

| Month | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 10 | 43.76 | 250.50 | 0.85 |
| 11 | 43.69 | 224.61 | 0.86 |
| 12 | 45.93 | 247.29 | 0.86 |

Performance remained relatively stable across all three future test months.

> **Metric provenance:** these figures are historical exploratory results stored
> in `notebooks/02_modelo.ipynb`; that notebook split on the calendar-month
> values 10, 11 and 12. They must not be presented as the result of the stricter
> complete-date pipeline. Their machine-readable transcription, record counts,
> source cells and dataset provenance live in
> [`docs/metrics/notebook_metrics.json`](docs/metrics/notebook_metrics.json).
> CI verifies that the JSON and saved notebook outputs remain identical with
> `python scripts/audit_published_metrics.py`.

Running `python -m src.train` performs the current, stricter evaluation and
writes its source-of-truth periods, cutoff, baselines, segmented metrics, drift
diagnostics and prediction interval to `reports/metrics.json`. To publish the
same strict report in the repository, run `python -m src.train --publish-metrics`.
That command writes `docs/metrics/strict_temporal_metrics.json`, including the
dataset SHA-256, generation timestamp, split sizes and exact periods, so results
can be traced to a specific input file. Never publish that file from a partial or
modified dataset without documenting it.

---

## 📉 Baseline Comparison

Two explicit baselines are evaluated: persistence (the previous month's
production) and the training-set mean, equivalent to a mean-strategy
`DummyRegressor`. They are implemented directly so their behavior is transparent.

The Random Forest substantially outperformed the baseline across MAE, RMSE and R².

This comparison helps ensure that the predictive model provides value beyond simply predicting the average production.

---

## 🔍 Model Interpretation

Feature importance analysis was performed to understand which variables contributed most to the predictions.

Important features included:

- Previous month's oil production
- Effective production time (`tef`)
- Well type
- Extraction method
- Province
- Basin
- Month

The analysis also showed that models without historical production tended to underestimate wells with exceptionally high production.

---

## 🏗️ Project Architecture

![Arquitectura del proyecto](docs/architecture.svg)

The deployment-oriented component and data-flow diagram is documented in
[`docs/production-architecture.md`](docs/production-architecture.md).

El entrenamiento y la evaluación son procesos **offline**. La API es un
proceso **online** separado que solamente carga el artefacto y sirve
predicciones; nunca reentrena durante una petición.

```text
oil-production-analytics-ml/
│
├── data/                         # local; datos grandes no versionados
├── docs/                         # arquitectura en formato SVG de texto
├── models/                       # artefactos locales de entrenamiento
├── notebooks/
│   ├── 01_exploracion.ipynb
│   └── 02_modelo.ipynb
├── powerbi/
│   └── ypf_production_dashboard.pbix
├── reports/                      # métricas y gráficos generados
├── sql/
│   ├── analytics_queries.sql
│   ├── schema.sql
│   └── views.sql
├── src/
│   ├── config.py                  # contrato de variables y configuración
│   ├── data.py                    # validación, features y cortes temporales
│   ├── evaluation.py              # métricas, drift y visualizaciones
│   ├── load_sqlite.py
│   ├── main.py                   # inferencia online
│   ├── modeling.py                # modelos e intervalos conformales
│   └── train.py                  # orquestación y publicación offline
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

> The original dataset and serialized model are excluded from Git because of their size.

---

## 🛠️ Tech Stack

### Data Analytics

- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

### Machine Learning

- scikit-learn
- Random Forest
- OneHotEncoder
- ColumnTransformer
- Pipeline
- Joblib

### Backend / Model Serving

- FastAPI
- Pydantic
- Uvicorn

### Development

- VS Code
- Git
- GitHub
- Python Virtual Environments

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd oil-production-analytics-ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

Choose the smallest dependency group that matches the task:

```bash
pip install -r requirements-api.txt    # prediction service only
pip install -r requirements-train.txt  # API + training and notebooks
pip install -r requirements-dev.txt    # all of the above + quality tooling
```

The supported interpreter is Python 3.12, declared in `.python-version` and
used by CI. All direct runtime, analysis and test dependencies are pinned in the
UTF-8 encoded `requirements-ci.txt`, so local and CI runs resolve the same declared
versions.

### Run with Docker

After training has created `models/modelo_produccion_petroleo.joblib`, start the
containerized API with:

```bash
docker compose up --build
```

The image runs as a non-root user and mounts `models/` read-only. The
`MODEL_PATH` environment variable can point the API to a different artifact.

### 5. Download and train

Download the official CSV linked in [Data Source](#-data-source), save it
exactly as `data/produccion.csv`, and run:

```bash
python -m src.train
```

The command creates the deployable bundle in `models/` and writes auditable
validation and final-test metrics, holdout predictions, conformal coverage and
plots in `reports/`. The default 3/3/3-month windows can be changed with
`--validation-months`, `--calibration-months` and `--test-months`. Optional XGBoost and SHAP
support is installed and activated explicitly:

```bash
pip install "xgboost>=3,<4" "shap>=0.46,<1"
python -m src.train --include-xgboost --with-shap
```

After reviewing the generated report, publish the strict metrics with:

```bash
python -m src.train --publish-metrics
git add docs/metrics/strict_temporal_metrics.json
```

The published JSON is intentionally generated by the same training execution,
rather than copied by hand from terminal output. Do not confuse it with
`docs/metrics/notebook_metrics.json`, which preserves the historical notebook
benchmark.

To build the analytical SQLite database and its Power BI views:

```bash
python -m src.load_sqlite
```

---

## 🌐 REST API

The trained Machine Learning pipeline is exposed through a **FastAPI REST API**.

Start the server:

```bash
uvicorn src.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Oil Production Prediction

```http
POST /predict
```

The request represents a forecast made before the target month. Production,
injection and effective-time fields all describe the previous completed month.

For operational inspection, `GET /model-info` exposes the selected model,
training cutoffs, expected features and interval method. Batch clients can use
`POST /predict/batch` with between 1 and 1,000 observations.

Example request:

```json
{
  "mes": 10,
  "iny_agua_lag1": 0,
  "iny_gas_lag1": 0,
  "tef_lag1": 31,
  "tipoextraccion": "Bombeo Mecánico",
  "tipoestado": "Extracción Efectiva",
  "tipopozo": "Petrolífero",
  "provincia": "Santa Cruz",
  "cuenca": "GOLFO SAN JORGE",
  "prod_pet_lag1": 52.86
}
```

Example response:

```json
{
  "produccion_predicha": 54.91,
  "intervalo_prediccion_90": {
    "inferior": 48.21,
    "superior": 61.84
  }
}
```

The optional interval uses the matching previous-production regime when that
regime has enough calibration data; otherwise it falls back to the global
split-conformal radius. Coverage is marginal within each calibrated regime, not
a guarantee for every individual observation.

---

## 🔄 Machine Learning Workflow

```text
Public Hydrocarbon Data
          ↓
     Data Loading
          ↓
        EDA
          ↓
    Data Cleaning
          ↓
  Feature Engineering
          ↓
     Lag Features
          ↓
Train / Validation / Calibration / Test
          ↓
    Preprocessing
          ↓
 Candidate Models
          ↓
 Model Evaluation
          ↓
 Model Serialization
          ↓
      FastAPI
          ↓
    POST /predict
```

---

## 🎯 Key Results

- Analyzed approximately **1 million hydrocarbon production records**
- Built an end-to-end Data Analytics and Machine Learning workflow
- Implemented temporal validation instead of relying only on random splitting
- Engineered historical well-production features
- Improved R² from approximately **0.45 to 0.86**
- Reduced MAE from approximately **142.7 to 44.5**
- Evaluated performance independently across future months
- Serialized the complete ML pipeline
- Exposed predictions through a REST API
- Documented limitations and prediction assumptions

---

## 🧭 Technical Decisions

- **Offline training vs. online inference:** `src/train.py` orchestrates focused
  data, modeling and evaluation modules and serializes their result.
  `src/main.py` only validates a request and invokes the serialized pipeline.
- **Complete temporal key:** `anio` and `mes` become a first-of-month timestamp.
  Sorting or splitting on `mes` alone is explicitly avoided.
- **Consecutive lag:** production, injection and effective-time lags are created
  per `idpozo` only when the
  preceding record is exactly one month earlier; gaps are not silently treated
  as the previous month.
- **Model comparison:** validation evaluates persistence
  (`prediction = prod_pet_lag1`), mean, Random Forest and
  HistGradientBoosting in the same period. XGBoost is an optional comparison
  via `--include-xgboost` to avoid making it a mandatory runtime dependency.
- **Selection:** the trainable candidate with the lowest validation MAE is
  refitted on train plus validation and serialized. Calibration and final test
  remain later, disjoint periods. Persistence remains a business baseline.
- **Segmented error:** the JSON report includes metrics by province, basin,
  production range, and new versus previously observed wells.
- **Uncertainty:** a separate period calibrates 90% split-conformal radii by
  previous-production regime, with a global fallback. Final-test coverage is
  recorded in the report.
- **Explainability:** `--with-shap` exports mean absolute SHAP importance when
  the optional package is installed.
- **Drift:** numeric PSI and categorical total-variation distance compare the
  training and holdout populations. These are monitoring signals, not automatic
  proof of model degradation.
- **Operational assumption:** all production and operational measurements are
  lagged; no target-month measurement is accepted by the forecast API.

## 📊 Generated Visuals

To keep the Git history and pull request fully text-compatible, generated PNG
plots are not committed. The offline script creates the real-vs-predicted and
monthly-holdout charts locally in `reports/` on every training run; that folder
is ignored by Git.

---

## ⚠️ Limitations

The final model uses the previous month's actual production (`prod_pet_lag1`) to estimate the following month's production.

Therefore, the reported performance corresponds to a **one-step-ahead forecasting scenario** rather than a recursive multi-month forecast where future lag values would also need to be predicted.

Extremely high-production wells may also present greater prediction errors due to their lower frequency and higher variance.

---

## 🔮 Future Improvements

Possible extensions include:

- LightGBM comparison
- Hyperparameter optimization
- TimeSeriesSplit
- Additional lag features (`lag2`, `lag3`)
- Rolling production averages
- MLflow experiment tracking
- Docker
- Calibrated conformal prediction intervals
- Automated production drift alerts
- Cloud deployment
- Automated retraining pipeline

---
## 📊 Data Source

This project uses official public hydrocarbon production data published by the
Argentine Government.

- **Dataset:** Producción de petróleo y gas por pozo (Capítulo IV)
- **Publisher:** Secretaría de Energía de la República Argentina
- **Source:** Datos Argentina
- **Granularity:** Monthly production by well, field, concession and province
- **Update frequency:** Monthly
- **Oil production (`prod_pet`):** m³
- **Gas production (`prod_gas`):** thousands of m³
- **Water production (`prod_agua`):** m³

Official dataset:

https://datos.gob.ar/ar/dataset/energia-produccion-petroleo-gas-por-pozo-capitulo-iv

The raw dataset is not included in this repository due to its size.

### Main variables used by the ML model

| Variable | Description |
|---|---|
| `mes` | Calendar month (1–12) |
| `iny_agua` | Water injection |
| `iny_gas` | Gas injection |
| `tef` | Effective operating time of the well |
| `tipoextraccion` | Extraction method |
| `tipoestado` | Operational status |
| `tipopozo` | Well type |
| `provincia` | Argentine province |
| `cuenca` | Hydrocarbon basin |
| `prod_pet_lag1` | Oil production from the previous monthly period |
| `prod_pet` | Target variable: monthly oil production (m³) |

### Reproducibility

1. Download the dataset from the official source above.
2. Save the CSV as:

   `data/produccion.csv`

3. Install the project dependencies.
4. Run the preprocessing/training pipeline described in this repository.

### Dataset License

The source code of this repository is licensed under the MIT License.

The dataset is provided by the Argentine Government and is not covered by
this repository's MIT License. Usage of the dataset is subject to the terms
of its original publisher.

## 👤 Author

**Franco Martín Sassi**

Software Engineer focused on Backend, Data, Machine Learning and AI Engineering.
