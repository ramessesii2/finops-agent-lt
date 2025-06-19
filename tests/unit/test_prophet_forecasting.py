#!/usr/bin/env python3
"""
Test script for Prophet forecasting with individual model fitting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_metrics_data():
    """Create test metrics data for forecasting."""
    # Generate sample time series data with trend and seasonality
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='1h')
    
    metrics_data = []
    
    # Create different metrics
    metrics = [
        'node_cpu_seconds_total',
        'node_memory_MemTotal_bytes', 
        'node_total_hourly_cost'
    ]
    
    for metric in metrics:
        # Generate realistic time series with trend and seasonality
        n_points = len(dates)
        
        # Base trend (increasing for cost, stable for others)
        if 'cost' in metric:
            trend = np.linspace(1.0, 1.5, n_points)  # Increasing cost
        else:
            trend = np.ones(n_points)  # Stable trend
        
        # Add daily seasonality (24-hour cycle)
        hour_of_day = np.array([d.hour for d in dates])
        seasonality = 0.2 * np.sin(2 * np.pi * hour_of_day / 24)
        
        # Add weekly seasonality (7-day cycle)
        day_of_week = np.array([d.dayofweek for d in dates])
        weekly_seasonality = 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Add noise
        noise = np.random.normal(0, 0.05, n_points)
        
        # Combine all components
        values = trend + seasonality + weekly_seasonality + noise
        
        # Ensure positive values
        values = np.maximum(values, 0.1)
        
        # Scale based on metric type
        if 'cpu' in metric:
            values *= 0.8  # CPU utilization 0-1
        elif 'memory' in metric:
            values *= 16 * 1024 * 1024 * 1024  # 16GB in bytes
        elif 'cost' in metric:
            values *= 2.0  # Cost in dollars per hour
        
        for i, (date, value) in enumerate(zip(dates, values)):
            metrics_data.append({
                'timestamp': date,
                'metric_name': metric,
                'value': value,
                'labels': {
                    'instance': 'test-node',
                    'clusterName': 'test-cluster'
                }
            })
    
    df = pd.DataFrame(metrics_data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.mark.unit
def test_prophet_forecasting(test_config):
    """Test Prophet forecasting with individual model fitting."""
    # Import the Prophet model
    from src.forecasting_agent.models.prophet import ProphetModel
    
    # Create test data
    test_metrics = create_test_metrics_data()
    logger.info(f"Created test data with {len(test_metrics)} records")
    logger.info(f"Metrics: {test_metrics['metric_name'].unique()}")
    
    # Test forecasting for each metric
    forecasts = []
    
    # Group by metric_name
    grouped = test_metrics.groupby('metric_name')
    
    for metric_name, group in grouped:
        logger.info(f"\nProcessing metric: {metric_name}")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': group.index.tz_localize(None),  # Remove timezone
            'y': group['value'].values
        })
        
        logger.info(f"Training Prophet model for {metric_name} with {len(df)} data points")
        
        # Create and fit model
        model = ProphetModel(test_config['models']['prophet'])
        model.fit(df)
        
        # Generate forecast
        horizon = test_config['models']['forecast_horizon']
        forecast = model.forecast(horizon=horizon, frequency='1D')
        
        logger.info(f"Generated forecast for {metric_name} with {len(forecast['forecast'])} future points")
        
        # Validate forecast structure
        assert 'forecast' in forecast, "Forecast should contain 'forecast' key"
        assert 'lower_bound' in forecast, "Forecast should contain 'lower_bound' key"
        assert 'upper_bound' in forecast, "Forecast should contain 'upper_bound' key"
        assert 'prophet_forecast' in forecast, "Forecast should contain 'prophet_forecast' key"
        
        # Check forecast values
        forecast_df = forecast['forecast']
        assert len(forecast_df) == horizon, f"Forecast should have {horizon} points"
        
        # Check that forecast values are reasonable
        assert all(forecast_df['value'] >= 0), "Forecast values should be non-negative"
        
        # Check confidence intervals
        lower_bounds = forecast['lower_bound']['value']
        upper_bounds = forecast['upper_bound']['value']
        forecast_values = forecast_df['value']
        
        assert all(lower_bounds <= forecast_values), "Lower bounds should be <= forecast values"
        assert all(upper_bounds >= forecast_values), "Upper bounds should be >= forecast values"
        
        logger.info(f"✅ Forecast validation passed for {metric_name}")
        
        forecasts.append({
            "metric_name": metric_name,
            "forecast": forecast,
            "model": model
        })
    
    logger.info(f"\n✅ Successfully generated forecasts for {len(forecasts)} metrics")
    
    # Test that we got forecasts for all metrics
    assert len(forecasts) == len(test_metrics['metric_name'].unique()), "Should have forecasts for all metrics" 