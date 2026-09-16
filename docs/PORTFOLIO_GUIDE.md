# Guía del propietario para publicar el portfolio

Esta guía separa las tareas que necesitan acceso humano a datos, Power BI o
cuentas externas de las mejoras que ya están automatizadas en el repositorio.

## 1. Generar el resultado estricto con el dataset completo

1. Descargá **Producción de petróleo y gas por pozo (Capítulo IV)** desde la
   fuente oficial enlazada en el README.
2. Guardá el archivo completo, sin muestrearlo ni editarlo, como:

   ```text
   data/produccion.csv
   ```

3. Creá el entorno e instalá las dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install -r requirements-train.txt
   ```

4. Ejecutá y publicá el reporte estricto:

   ```bash
   python -m src.train --publish-metrics
   ```

5. Conservá estos archivos para la revisión:

   ```text
   docs/metrics/strict_temporal_metrics.json
   reports/model_real_vs_predicted.png
   reports/model_monthly_holdout.png
   reports/metrics.json
   reports/holdout_predictions.csv
   models/modelo_produccion_petroleo.joblib
   ```

6. Antes de publicar, verificá que el CSV sea la versión oficial completa y
   revisá si las unidades y los resultados son plausibles. No copies métricas a
   mano: el JSON publicado debe salir de la misma ejecución de entrenamiento.

> El CSV, las predicciones fila a fila y el modelo no deben subirse al
> repositorio sin revisar primero tamaño, licencia y privacidad. El JSON de
> métricas sí está diseñado para versionarse.

## 2. Capturar el dashboard de Power BI

Abrí `powerbi/ypf_production_dashboard.pbix` en Power BI Desktop y exportá entre
tres y cuatro capturas PNG:

1. Resumen ejecutivo con KPIs.
2. Producción por provincia o cuenca.
3. Vista operativa por estado, pozo o método de extracción.
4. Evolución temporal o página de detalle.

Requisitos recomendados:

- Resolución mínima de 1600 × 900.
- Sin rutas locales, datos personales ni ventanas del editor alrededor.
- Filtros en un estado comprensible y consistente.
- Un nombre descriptivo, por ejemplo `dashboard_overview.png`.

Guardá las imágenes en `docs/images/powerbi/`. Para documentar las medidas,
prepará por cada página:

```text
Nombre de la página:
Objetivo:
KPIs:
Filtros:
Visuales:
Medidas DAX importantes:
Decisión de negocio que permite tomar:
```

## 3. Preparar la demo de Swagger

Con el modelo ya entrenado:

```bash
docker compose up --build
```

Abrí `http://localhost:8000/docs` y grabá esta secuencia:

1. `GET /health/live` devuelve 200.
2. `GET /health/ready` confirma que el modelo está listo.
3. `GET /model-info` muestra versión, features y lineage.
4. `POST /predict` devuelve la producción y su intervalo.
5. Opcionalmente, `POST /predict/batch` muestra inferencia por lote.

La grabación debería durar entre 30 y 45 segundos. Exportala como MP4 o GIF sin
mostrar tokens, rutas personales ni otras ventanas. Una demo desplegada es
preferible, pero no es obligatoria para la primera versión del portfolio.

## 4. Información personal que falta decidir

Antes de enviar el repositorio a recruiters, confirmá:

- Idioma principal: se recomienda README en inglés y una versión española.
- Rol objetivo: Data Scientist, ML Engineer o Data Analyst.
- URL de LinkedIn y portfolio.
- Email profesional, sólo si querés hacerlo público.
- Proveedor cloud y presupuesto, si se hará una demo online.

## 5. Validación final

Ejecutá:

```bash
ruff format --check .
ruff check .
mypy
python scripts/audit_published_metrics.py
python -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
docker build -t oil-production-api:portfolio .
```

Después verificá manualmente que:

- El resultado estricto supera o explica su diferencia frente a persistencia.
- Las fechas de train, validation, calibration y test no se solapan.
- Las imágenes se ven correctamente desde GitHub.
- Swagger funciona desde una clonación limpia.
- El README no afirma que la demo o el deployment existen antes de publicarlos.

## 6. Qué ya quedó automatizado

- Dependencias separadas para API, entrenamiento y desarrollo.
- Formato, lint, type checking, tests y cobertura en CI.
- Contenedor no-root con healthcheck.
- Liveness y readiness separados.
- Contrato versionado y validado del artefacto.
- Hash del dataset y versiones de runtime guardadas en metadata.
