# Forecasting Agent Dockerfile
FROM python:3.10-slim

# Install build tools & git (for darts dependency that compiles lightgbm etc.)
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Ensure Python can locate local src package
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Copy project files
COPY . /app

# Install PDM to resolve project deps defined in pyproject.toml
RUN pip install --no-cache-dir pdm && \
    pdm install --prod --no-editable --no-lock

# Expose forecast API/metrics port
EXPOSE 8081

# Entrypoint
CMD ["pdm", "run", "python", "-m", "forecasting_agent.main", "--config", "/app/config.yaml"]
