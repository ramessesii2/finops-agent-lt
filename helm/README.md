# FinOps Agent Helm Chart

A Helm chart for deploying FinOps Agent - a Financial Operations Forecasting Agent that provides cost optimization and forecasting capabilities for Kubernetes clusters.

## Overview

FinOps Agent is a forecasting application that:
- Collects metrics from Prometheus
- Uses TOTO models for forecasting
- Provides cost optimization recommendations
- Exposes metrics for monitoring

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Access to a Prometheus instance

## Installation

### Add the repository
```bash
helm repo add finops-agent <repository-url>
helm repo update
```

### Install the chart
```bash
# Install with default values
helm install finops-agent ./helm

# Install with custom values
helm install finops-agent ./helm -f custom-values.yaml

# Install in a specific namespace
helm install finops-agent ./helm --namespace finops --create-namespace
```

## Configuration

The following table lists the configurable parameters and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `ramessesii/finops-agent` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `Always` |
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `8081` |
| `config.collector.url` | Prometheus URL | `http://kof-mothership-promxy.kof.svc.cluster.local:8082` |
| `config.collector.lookback_days` | Data lookback period | `7` |
| `config.models.type` | Model type | `toto` |
| `config.models.forecast_horizon` | Forecast horizon in days | `7` |
| `resources` | Resource limits and requests | `{}` |
| `nodeSelector` | Node selector | `{}` |
| `tolerations` | Tolerations | `[]` |
| `affinity` | Affinity rules | `{}` |

## Examples

### Basic Installation
```bash
helm install my-finops-agent ./helm
```

### Production Installation with Custom Configuration
```bash
helm install finops-agent ./helm \
  --set replicaCount=2 \
  --set config.collector.url=http://my-prometheus:9090 \
  --set resources.requests.cpu=200m \
  --set resources.requests.memory=512Mi \
  --set resources.limits.cpu=2000m \
  --set resources.limits.memory=4Gi
```

## Monitoring

The application exposes metrics on `/metrics` endpoint (port 8081 by default). You can configure Prometheus to scrape these metrics:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: finops-agent
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: finops-agent
  endpoints:
  - port: metrics
    path: /metrics
```

## Upgrading

```bash
helm upgrade finops-agent ./helm
```

## Uninstalling

```bash
helm uninstall finops-agent
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -l app.kubernetes.io/name=finops-agent
```

### View logs
```bash
kubectl logs -f deployment/finops-agent
```

### Check configuration
```bash
kubectl get configmap finops-agent-config -o yaml
```

## Support

For issues and questions, please refer to the project repository or create an issue in the GitHub repository.
