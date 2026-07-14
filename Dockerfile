FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[trader,dashboard]"
RUN mkdir -p ./data
COPY data/company_event_sample.json data/evaluation_seed.json data/sample_documents.json data/strategy_registry.json ./data/
RUN addgroup --system trader \
    && adduser --system --ingroup trader --home /app --no-create-home trader \
    && chown -R trader:trader /app

USER trader

EXPOSE 8000 8501
CMD ["uvicorn", "crypto_event_trader.api:app", "--host", "0.0.0.0", "--port", "8000"]
