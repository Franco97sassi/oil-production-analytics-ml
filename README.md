# 🛢️ Oil Production Analytics & Machine Learning

End-to-end **Data Analytics and Machine Learning project** focused on analyzing and predicting monthly oil production using public hydrocarbon production data from Argentina.

The project covers the complete workflow from exploratory data analysis and feature engineering to temporal validation, model evaluation, serialization, and deployment through a REST API.

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
iny_agua
iny_gas
tef
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

Instead of relying only on a random train/test split, the final model uses a **temporal validation strategy**.

```text
Months 1–9   → Training
Months 10–12 → Testing
```

This provides a more realistic evaluation by training the model using historical observations and testing it on later periods.

The final model represents a **one-step-ahead prediction scenario**, meaning that the previous month's actual production is available when predicting the following month.

---

## 🌲 Model

The final model is a:

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

### Model without production history

| Metric | Result |
|---|---:|
| MAE | 142.71 |
| RMSE | 470.91 |
| R² | 0.4549 |

### Final model with `prod_pet_lag1`

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

---

## 📉 Baseline Comparison

A `DummyRegressor` was used as a baseline to verify that the Machine Learning model was learning meaningful patterns.

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

```text
ypf-data-ml/
│
├── data/
│   └── produccion.csv
│
├── models/
│   └── modelo_produccion_petroleo.joblib
│
├── notebooks/
│   ├── 01_exploracion.ipynb
│   └── 02_modelo.ipynb
│
├── src/
│   └── main.py
│
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

```bash
pip install -r requirements.txt
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
  "status": "ok"
}
```

### Oil Production Prediction

```http
POST /predict
```

Example request:

```json
{
  "mes": 10,
  "iny_agua": 0,
  "iny_gas": 0,
  "tef": 31,
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
  "produccion_predicha": 54.91
}
```

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
 Temporal Train/Test Split
          ↓
    Preprocessing
          ↓
   Random Forest
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

## ⚠️ Limitations

The final model uses the previous month's actual production (`prod_pet_lag1`) to estimate the following month's production.

Therefore, the reported performance corresponds to a **one-step-ahead forecasting scenario** rather than a recursive multi-month forecast where future lag values would also need to be predicted.

Extremely high-production wells may also present greater prediction errors due to their lower frequency and higher variance.

---

## 🔮 Future Improvements

Possible extensions include:

- XGBoost / LightGBM
- Hyperparameter optimization
- TimeSeriesSplit
- Additional lag features (`lag2`, `lag3`)
- Rolling production averages
- SHAP explainability
- MLflow experiment tracking
- Docker
- Automated testing
- Model monitoring and drift detection
- Cloud deployment
- Automated retraining pipeline

---

## 👤 Author

**Franco Martín Sassi**

Software Engineer focused on Backend, Data, Machine Learning and AI Engineering.