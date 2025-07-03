# Forecasting Agent

FinOps forecasting agent that continuously ingests cost and utilization metrics from Prometheus, applies advanced time-series forecasting models (NBEATS, Datadog TOTO), and provides accurate predictions with built-in validation and monitoring capabilities.

## ✨ Key Features

- **🔄 Continuous Operation**: Runs as a service with configurable collection intervals
- **🤖 Multiple Models**: Support for NBEATS and Datadog's TOTO zero-shot forecasting
- **📊 Built-in Validation**: 70/30 train/test split with MAPE, MAE, RMSE accuracy metrics
- **🔍 Health Monitoring**: Prometheus connectivity checks and graceful error handling
- **🌐 HTTP API**: RESTful endpoints for forecast data and cluster metrics
- **⚙️ Configurable**: YAML-based configuration for all components
- **🚀 Cloud-Native**: Kubernetes-ready with Docker deployment support

## Architecture

![Architecture](./docs/architecture.png)

## 🏗️ Project Structure

```
forecasting-agent/
├── src/
│   └── forecasting_agent/
│       ├── main.py                    # Main application entry point
│       ├── collectors/
│       │   └── prometheus.py          # Prometheus metrics collector
│       ├── adapters/
│       │   ├── forecasting/
│       │   │   ├── nbeats_adapter.py  # NBEATS forecasting model
│       │   │   ├── toto_adapter.py    # Datadog TOTO zero-shot model
│       │   │   └── prophet_adapter.py # Prophet forecasting model
│       │   └── prometheus_timeseries_adapter.py # Data conversion utilities
│       ├── validation/
│       │   └── forecast_validator.py  # Model accuracy validation
│       └── optimizers/
│           └── idle_capacity.py       # Cost optimization logic
├── toto/                              # Datadog TOTO model (git submodule)
├── tests/
│   └── test_toto_adapter.py          # Unit tests
├── config.yaml                        # Main configuration file
├── pyproject.toml                     # Python dependencies
└── deployments/
    └── kubernetes/
        └── deployment.yaml            # Kubernetes deployment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Prometheus server with cost/utilization metrics
- (Optional) CUDA-compatible GPU for TOTO model

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd forecasting-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pdm install
# OR using pip:
pip install -e .

# Install TOTO model (optional)
git submodule update --init --recursive
```

### Basic Usage

```bash
# Configure your Prometheus endpoint in config.yaml
# Then run the agent
python -m forecasting_agent.main config.yaml
```

The agent will:
1. 🔄 Continuously collect metrics from Prometheus
2. 🤖 Generate forecasts using your chosen model
3. 📊 Validate accuracy using train/test splits
4. 🌐 Serve results via HTTP API at `http://localhost:8081`

## ⚙️ Configuration

The agent is configured via `config.yaml`. Here's the current configuration structure:

```yaml
# Prometheus data source
collector:
  type: prometheus
  url: http://localhost:8082
  lookback_days: 2
  step: "1h"
  disable_ssl: true

# Model selection and parameters
models:
  type: toto  # or 'nbeats'
  freq: "h"
  forecast_horizon: 7  # days
  quantiles: [0.1, 0.5, 0.9]
  
  # NBEATS-specific settings
  nbeats:
    input_chunk_length: 24
    output_chunk_length: 24
    n_epochs: 50
    
  # TOTO-specific settings
  toto:
    checkpoint: Datadog/Toto-Open-Base-1.0
    device: cpu  # or cuda
    context_length: 4096
    num_samples: 256

# Validation settings
validation:
  enabled: true
  interval_cycles: 2  # Run validation every 2 forecast cycles
  train_ratio: 0.7    # 70% training, 30% testing

# Agent runtime settings
agent:
  interval: 300  # seconds between forecast updates

# HTTP API settings
metrics:
  host: 0.0.0.0
  port: 8081
```

## 🔌 API Endpoints

The agent exposes the following HTTP endpoints:

### Forecast Data
- **`GET /metrics/{cluster_name}`** - Get forecast JSON for a specific cluster
  ```json
  {
    "cost_usd": {
      "q0.10": [{"x": "2024-01-01T00:00:00Z", "y": 1.23}],
      "q0.50": [{"x": "2024-01-01T00:00:00Z", "y": 1.45}],
      "q0.90": [{"x": "2024-01-01T00:00:00Z", "y": 1.67}]
    },
    "cpu_pct": { /* similar structure */ },
    "mem_pct": { /* similar structure */ }
  }
  ```

### Cluster List
- **`GET /metrics`** - List available clusters
  ```json
  {"clusters": ["production", "staging"]}
  ```

## 🤖 Supported Models

### NBEATS (Neural Basis Expansion Analysis)
- **Type**: Deep learning model for time series forecasting
- **Best for**: Regular patterns, seasonal data
- **Configuration**: `models.type: nbeats`
- **GPU**: Optional (CPU fallback available)

### TOTO (Datadog's Zero-Shot Forecasting)
- **Type**: Transformer-based zero-shot forecasting
- **Best for**: Limited historical data, diverse time series
- **Configuration**: `models.type: toto`
- **GPU**: Recommended for performance
- **Checkpoint**: Uses Hugging Face model `Datadog/Toto-Open-Base-1.0`

## 📊 Validation & Accuracy

The agent includes built-in forecast validation:

- **Train/Test Split**: Configurable ratio (default 70/30)
- **Metrics**: MAPE, MAE, RMSE calculated automatically
- **Frequency**: Runs every N forecast cycles (configurable)
- **Logging**: Detailed accuracy reports in application logs

```
INFO: Component cost: MAPE=12.34%, MAE=0.0456, RMSE=0.0789
INFO: Component cpu_usage: MAPE=8.91%, MAE=2.1234, RMSE=3.4567
INFO: Validation completed for 6 components across 2 clusters
```

## 🚀 Deployment

### Kubernetes (Recommended)

```bash
# Apply the deployment
kubectl apply -f deployments/kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=forecasting-agent
kubectl logs -l app=forecasting-agent
```

### Local Development

```bash
# Install dependencies
pdm install

# Run directly
PYTHONPATH=src pdm run python -m forecasting_agent.main config.yaml

# Or with debugging
PYTHONPATH=src pdm run python -m debugpy --listen 5678 --wait-for-client -m forecasting_agent.main config.yaml
```

## 🔧 Troubleshooting

**Prometheus Connection Failed**
- Verify Prometheus URL in `config.yaml`
- Check network connectivity: `curl http://your-prometheus:9090/api/v1/query?query=up`
- Review agent logs for detailed error messages

**High MAPE Values**
- Increase `lookback_days` for more training data
- Try different model types (`nbeats` vs `toto`)
- Adjust `forecast_horizon` to shorter periods
- Check data quality and missing values

## 📈 Monitoring

The agent provides comprehensive logging:

```bash
# View real-time logs
tail -f forecasting-agent.log

# Filter for validation results
grep "MAPE" forecasting-agent.log

# Monitor health checks
grep "health check" forecasting-agent.log
```

## 📄 License

Apache 2.0 - See LICENSE file for details
