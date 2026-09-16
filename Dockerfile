FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/modelo_produccion_petroleo.joblib

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements-api.txt

COPY src ./src
RUN mkdir -p models && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
