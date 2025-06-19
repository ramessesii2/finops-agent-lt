#!/usr/bin/env python3
"""
Test script to verify Prometheus metrics server functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import requests
import logging
import pytest
from prometheus_client import start_http_server, Gauge, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the same metrics as in main.py
FORECAST_COST = Gauge(
    'forecast_cluster_monthly_cost',
    'Forecasted monthly cost for cluster',
    ['cluster']
)
FORECAST_CONFIDENCE = Gauge(
    'forecast_confidence_interval',
    'Forecast confidence interval bounds',
    ['cluster', 'bound']
)
OPTIMIZATION_SAVINGS = Gauge(
    'optimization_potential_savings',
    'Potential savings from optimizations',
    ['cluster', 'type']
)
COLLECTION_ERRORS = Counter(
    'collection_errors_total',
    'Total number of metric collection errors',
    ['collector']
)

@pytest.mark.unit
def test_metrics_server():
    """Test that the metrics server starts and metrics are accessible."""
    try:
        # Start the metrics server on port 8001
        port = 8001
        start_http_server(port, addr='0.0.0.0')
        logger.info(f"Started Prometheus metrics server on port {port}")
        
        # Set some test metrics
        FORECAST_COST.labels(cluster='test-cluster').set(1000.0)
        FORECAST_CONFIDENCE.labels(cluster='test-cluster', bound='upper').set(1200.0)
        FORECAST_CONFIDENCE.labels(cluster='test-cluster', bound='lower').set(800.0)
        OPTIMIZATION_SAVINGS.labels(cluster='test-cluster', type='node_scaling').set(50.0)
        COLLECTION_ERRORS.labels(collector='prometheus').inc()
        
        logger.info("Set test metrics")
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Try to access the metrics endpoint
        url = f'http://localhost:{port}/metrics'
        logger.info(f"Attempting to access metrics at: {url}")
        
        response = requests.get(url, timeout=10)
        
        assert response.status_code == 200, f"Failed to access metrics endpoint. Status code: {response.status_code}"
        
        logger.info("✅ Successfully accessed metrics endpoint")
        logger.info(f"Response length: {len(response.text)} characters")
        
        # Check if our metrics are in the response
        metrics_text = response.text
        assert 'forecast_cluster_monthly_cost' in metrics_text, "forecast_cluster_monthly_cost metric not found"
        assert 'optimization_potential_savings' in metrics_text, "optimization_potential_savings metric not found"
        assert 'collection_errors_total' in metrics_text, "collection_errors_total metric not found"
        
        logger.info("✅ All expected metrics found")
        
        # Print a sample of the metrics
        logger.info("Sample metrics output:")
        lines = metrics_text.split('\n')[:20]  # First 20 lines
        for line in lines:
            if line.strip():
                logger.info(f"  {line}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing metrics server: {str(e)}")
        raise

if __name__ == '__main__':
    success = test_metrics_server()
    if success:
        print("✅ Metrics server test passed!")
        print("You can now access metrics at: http://localhost:8001/metrics")
    else:
        print("❌ Metrics server test failed!") 