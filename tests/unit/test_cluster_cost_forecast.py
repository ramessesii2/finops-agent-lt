import pytest
import pandas as pd
from src.forecasting_agent.main import ForecastingAgent

@pytest.fixture
def test_config():
    # Minimal config for the agent
    return {
        'collector': {'prometheus_url': 'http://localhost:8082', 'lookback_days': 7},
        'models': {'forecast_horizon': 7, 'prophet': {}},
        'optimizer': {},
        'metrics': {'port': 8001, 'host': '0.0.0.0'},
        'agent': {'interval': 3600}
    }

def test_generate_forecast_with_dummy_cost_metrics(test_config):
    # Create dummy cost metrics for a single cluster with two nodes
    dates = pd.date_range('2024-01-01', periods=24, freq='H')
    cluster_name = 'test-cluster'
    # Node 1
    node1 = pd.DataFrame({
        'timestamp': dates,
        'metric_name': 'node_total_hourly_cost',
        'value': [0.10] * len(dates),
        'labels': [{'clusterName': cluster_name, 'node': 'node-1'}] * len(dates)
    })
    # Node 2
    node2 = pd.DataFrame({
        'timestamp': dates,
        'metric_name': 'node_total_hourly_cost',
        'value': [0.20] * len(dates),
        'labels': [{'clusterName': cluster_name, 'node': 'node-2'}] * len(dates)
    })
    # Combine
    df = pd.concat([node1, node2], ignore_index=True)
    df.set_index('timestamp', inplace=True)

    agent = ForecastingAgent.__new__(ForecastingAgent)
    agent.config = test_config

    # Run forecast
    result = agent.generate_forecast(df)
    forecasts = result['forecasts']

    # There should be a forecast for the cluster
    assert any(f['metric_name'] == cluster_name for f in forecasts)

    # Check that forecasted values are reasonable (should be close to 0.30 per hour for the cluster)
    for f in forecasts:
        if f['metric_name'] == cluster_name:
            forecast_df = f['forecast']['forecast']
            # All forecasted values should be positive and close to 0.30
            assert (forecast_df['value'] > 0).all()
            print(forecast_df['value'])
            # Optionally, check that the mean is close to 0.30
            mean_forecast = forecast_df['value'].mean()
            assert abs(mean_forecast - 0.30) < 0.05, f"Mean forecast {mean_forecast} not close to expected 0.30" 