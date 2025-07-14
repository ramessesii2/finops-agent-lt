# Forecasting Agent Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app/src:/app"

COPY pyproject.toml /app/

# Install PDM and dependencies
RUN pip install --no-cache-dir pdm && \
    pdm install --prod --no-editable --no-lock

COPY src/ /app/src/
COPY toto/ /app/toto/
COPY config.yaml /app/

# Expose forecast API/metrics port
EXPOSE 8081

# Entrypoint
CMD ["pdm", "run", "python", "-m", "main", "--config", "/app/config.yaml"]
