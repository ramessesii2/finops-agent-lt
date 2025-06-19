# forecasting-agent
A modular FinOps agent that ingests cost-and-utilization metrics from any Prometheus-compatible endpoint, applies pluggable short-term time-series forecasting models (Kats-Prophet, Datadog Toto, NVIDIA Tesseract), estimates idle-capacity savings, and exports forecasts and optimization hints as Prometheus metrics and JSON for easy dashboarding in Grafana or similar tools

## Architecture

![Architecture](./docs/architecture.png)

## Project Structure

```
forecasting-agent/
├── src/
│   ├── forecasting_agent/
│   │   ├── __init__.py
│   │   ├── main.py                 # Application entry point
│   │   ├── config.py              # Configuration management
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py         # Prometheus metrics endpoints
│   │   │   ├── recommendations.py # JSON recommendations endpoint
│   │   │   └── health.py          # Health check endpoints
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Base collector interface
│   │   │   └── prometheus.py     # Prometheus metrics collector
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Base model interface
│   │   │   ├── prophet.py       # Prophet implementation
│   │   │   ├── toto.py         # Datadog Toto implementation
│   │   │   └── tesseract.py    # NVIDIA Tesseract implementation
│   │   ├── optimizers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Base optimizer interface
│   │   │   └── idle_capacity.py # Idle capacity optimization
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── metrics.py       # Metric processing utilities
│   │       └── validation.py    # Data validation utilities
├── tests/
│   ├── __init__.py
│   ├── test_collectors/
│   ├── test_models/
│   └── test_optimizers/
├── docs/
│   ├── architecture.png
│   └── development.md
├── deployments/
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── servicemonitor.yaml
│   └── docker/
│       └── Dockerfile
├── pyproject.toml
└── README.md
```

## Features

- Ingest cost-and-utilization metrics from any Prometheus-compatible endpoint
- Apply pluggable short-term time-series forecasting models (Kats-Prophet, Datadog Toto, NVIDIA Tesseract)
- Estimate idle-capacity savings
- Export forecasts and optimization hints as Prometheus metrics and JSON for easy dashboarding in Grafana or similar tools

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

The agent can be configured using environment variables or a configuration file:

```yaml
# config.yaml
collector:
  type: prometheus
  url: http://prometheus:9090
  scrape_interval: 300  # 5 minutes

models:
  default: prophet
  prophet:
    changepoint_prior_scale: 0.05
  toto:
    confidence_interval: 0.95
  tesseract:
    gpu_enabled: true

optimizer:
  idle_threshold: 0.5  # 50% utilization threshold
  min_savings_threshold: 100  # Minimum daily savings to report

metrics:
  port: 8000
  path: /metrics
```

## Development

### Adding a New Forecasting Model

1. Create a new model class in `src/forecasting_agent/models/`
2. Implement the `BaseModel` interface
3. Register the model in the model factory

Example:
```python
from forecasting_agent.models.base import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
    def forecast(self, data):
        # Implement forecasting logic
        pass
```

### Adding a New Optimizer

1. Create a new optimizer class in `src/forecasting_agent/optimizers/`
2. Implement the `BaseOptimizer` interface
3. Register the optimizer in the optimizer factory

## Deployment

### Docker

```bash
docker build -t forecasting-agent .
docker run -p 8000:8000 forecasting-agent
```

### Kubernetes

```bash
kubectl apply -f deployments/kubernetes/
```

## Metrics

The agent exposes the following Prometheus metrics:

- `forecast_cluster_monthly_cost{cluster="prod"}`: Forecasted monthly cost
- `forecast_confidence_interval{cluster="prod",bound="upper"}`: Upper bound of forecast
- `forecast_confidence_interval{cluster="prod",bound="lower"}`: Lower bound of forecast
- `optimization_potential_savings{cluster="prod",type="node"}`: Potential savings from optimizations

## License

Apache 2.0
