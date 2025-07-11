# TOTO Forecast Tensor Shape Guide

## Overview

This document explains the structure and meaning of TOTO forecast sample tensors, providing concrete examples to help understand how node-metric combinations map to tensor dimensions.

## Tensor Shape Format

TOTO forecast samples have the shape: `[batch_size, n_variates, horizon_timesteps, n_samples]`

### Real Example Analysis

**Observed Shape**: `torch.Size([1, 21, 168, 256])`

Let's break down each dimension:

## Dimension Breakdown

### Dimension 0: Batch Size (1)
- **Meaning**: Number of forecast batches processed simultaneously
- **Value**: `1` - Single batch of data
- **Usage**: Allows for batch processing of multiple forecast requests

### Dimension 1: Number of Variates (21)
- **Meaning**: Total number of node-metric combinations (variates)
- **Value**: `21` - Indicates 21 separate time series being forecasted
- **Calculation**: `nodes × metrics = total_variates`

#### Example Variate Calculation:
```
Real System Architecture: 5 cluster-level + 16 node-level = 21 variates

Cluster-Level Metrics (5 variates):
- cost_usd (node="cluster-aggregate")
- cpu_pct (node="cluster-aggregate")
- mem_pct (node="cluster-aggregate")
- node_count (node="cluster-aggregate")
- mem_total_gb (node="cluster-aggregate")

Node-Level Metrics (4 metrics × 4 nodes = 16 variates):
- cost_usd_per_node
- cpu_pct_per_node
- mem_pct_per_node
- cpu_total_cores_per_node

Nodes:
- aws-ramesses-regional-0-cp-0 (control plane)
- aws-ramesses-regional-0-worker-1 (worker)
- aws-ramesses-regional-0-worker-2 (worker)
- aws-ramesses-regional-0-worker-3 (worker)

Total Variates: 5 cluster-level + (4 metrics × 4 nodes) = 5 + 16 = 21 variates
```

#### Variate Mapping Example:
```
Variate Index | Node                              | Metric
0            | cluster-aggregate                 | cost_usd
1            | cluster-aggregate                 | cpu_pct
2            | cluster-aggregate                 | mem_pct
3            | cluster-aggregate                 | node_count
4            | cluster-aggregate                 | mem_total_gb
5            | aws-ramesses-regional-0-cp-0      | cost_usd_per_node
6            | aws-ramesses-regional-0-cp-0      | cpu_pct_per_node
7            | aws-ramesses-regional-0-cp-0      | mem_pct_per_node
8            | aws-ramesses-regional-0-cp-0      | cpu_total_cores_per_node
9            | aws-ramesses-regional-0-worker-1  | cost_usd_per_node
10           | aws-ramesses-regional-0-worker-1  | cpu_pct_per_node
11           | aws-ramesses-regional-0-worker-1  | mem_pct_per_node
12           | aws-ramesses-regional-0-worker-1  | cpu_total_cores_per_node
13           | aws-ramesses-regional-0-worker-2  | cost_usd_per_node
14           | aws-ramesses-regional-0-worker-2  | cpu_pct_per_node
15           | aws-ramesses-regional-0-worker-2  | mem_pct_per_node
16           | aws-ramesses-regional-0-worker-2  | cpu_total_cores_per_node
17           | aws-ramesses-regional-0-worker-3  | cost_usd_per_node
18           | aws-ramesses-regional-0-worker-3  | cpu_pct_per_node
19           | aws-ramesses-regional-0-worker-3  | mem_pct_per_node
20           | aws-ramesses-regional-0-worker-3  | cpu_total_cores_per_node
```

### Dimension 2: Horizon Timesteps (168)
- **Meaning**: Number of future time points to forecast
- **Value**: `168` - Represents 168 timesteps into the future
- **Calculation**: `forecast_days × timesteps_per_day`

#### Example Horizon Calculation:
```
Scenario: 7-day forecast with hourly granularity

Forecast Period: 7 days
Time Granularity: 1 hour
Calculation: 7 days × 24 hours/day = 168 timesteps

Timeline:
- Timestep 0: +1 hour from now
- Timestep 1: +2 hours from now
- Timestep 23: +24 hours from now (1 day)
- Timestep 47: +48 hours from now (2 days)
- ...
- Timestep 167: +168 hours from now (7 days)
```

### Dimension 3: Monte Carlo Samples (256)
- **Meaning**: Number of probabilistic forecast samples for uncertainty quantification
- **Value**: `256` - 256 different forecast scenarios
- **Purpose**: Enables calculation of confidence intervals and quantiles

#### Sample Distribution Example:
```
For each variate at each timestep, we have 256 possible values:

Variate 0 (prod-cluster-cp-0, cost_usd_per_node), Timestep 0:
Sample 0: $0.087/hour
Sample 1: $0.091/hour
Sample 2: $0.089/hour
...
Sample 255: $0.092/hour

From these 256 samples, we calculate:
- 10th percentile (P10): $0.085/hour
- 50th percentile (P50/median): $0.089/hour  
- 90th percentile (P90): $0.093/hour
```

## Tensor Access Patterns

### Accessing Specific Forecasts

```python
# Tensor shape: [1, 21, 168, 256]
samples = forecast_result.samples

# Get all samples for a specific variate and timestep
variate_idx = 0  # First node-metric combination
timestep_idx = 23  # 24 hours from now
variate_timestep_samples = samples[0, variate_idx, timestep_idx, :]
# Shape: [256] - all Monte Carlo samples for this variate at this time

# Get all timesteps for a specific variate
variate_all_times = samples[0, variate_idx, :, :]
# Shape: [168, 256] - all timesteps and samples for this variate

# Calculate quantiles for a specific variate and timestep
import torch
p10 = torch.quantile(variate_timestep_samples, 0.1)
p50 = torch.quantile(variate_timestep_samples, 0.5)  # median
p90 = torch.quantile(variate_timestep_samples, 0.9)
```

### Memory Considerations

```python
# Memory usage calculation
batch_size = 1
n_variates = 21
horizon = 168
n_samples = 256
dtype_size = 4  # float32 = 4 bytes

memory_bytes = batch_size * n_variates * horizon * n_samples * dtype_size
memory_mb = memory_bytes / (1024 * 1024)

print(f"Tensor memory usage: {memory_mb:.1f} MB")
# Output: Tensor memory usage: 36.0 MB
```

## Performance Implications

### Tensor Size Scaling
```
Small cluster (2 nodes, 5 metrics, 1 day, 100 samples):
[1, 10, 24, 100] = 24,000 values = 96 KB

Medium cluster (5 nodes, 8 metrics, 7 days, 256 samples):
[1, 40, 168, 256] = 1,720,320 values = 6.9 MB

Large cluster (20 nodes, 15 metrics, 30 days, 512 samples):
[1, 300, 720, 512] = 110,592,000 values = 442 MB
```

### Optimization Tips
1. **Reduce samples** for faster computation (256 → 128 or 64)
2. **Shorter horizons** for less memory usage
3. **Batch processing** for multiple clusters
4. **Selective metrics** - only forecast needed metrics per node


## Integration with Forecasting Pipeline

The tensor flows through the system as follows:

```
1. Raw Prometheus Data
   ↓
2. PrometheusToTotoAdapter
   → Creates MaskedTimeseries input tensor
   ↓
3. TOTO Model
   → Generates forecast samples tensor [1, 21, 168, 256]
   ↓
4. TOTOAdapter.forecast()
   → Calculates quantiles from samples
   ↓
5. ForecastFormatConverter
   → Converts to final output format
```