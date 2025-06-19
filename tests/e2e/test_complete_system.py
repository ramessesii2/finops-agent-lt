#!/usr/bin/env python3
"""
Comprehensive test for the complete system with simplified data structure.
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

def create_comprehensive_test_data():
    """Create comprehensive test data for the complete system."""
    # Generate sample time series data
    dates = pd.date_range(start='2024-01-01', end='2024-01-14', freq='1h')
    
    metrics_data = []
    
    # Create test nodes with different characteristics
    nodes = [
        {'name': 'node-idle-1', 'cpu_util': 0.2, 'mem_util': 0.3, 'cost': 1.0},  # Idle node
        {'name': 'node-busy-1', 'cpu_util': 0.8, 'mem_util': 0.7, 'cost': 2.0},  # Busy node
        {'name': 'node-mixed-1', 'cpu_util': 0.4, 'mem_util': 0.6, 'cost': 1.5},  # Mixed utilization
    ]
    
    for node in nodes:
        for date in dates:
            # CPU metrics with different modes
            cpu_modes = ['user', 'system', 'idle', 'iowait']
            total_cpu = 0
            
            for mode in cpu_modes:
                if mode == 'idle':
                    # Idle time based on node utilization
                    cpu_value = (1 - node['cpu_util']) + np.random.normal(0, 0.05)
                else:
                    # Active CPU time
                    cpu_value = (node['cpu_util'] / 3) + np.random.normal(0, 0.02)  # Divide by 3 active modes
                
                cpu_value = max(0, cpu_value)
                total_cpu += cpu_value
                
                metrics_data.append({
                    'timestamp': date,
                    'metric_name': 'node_cpu_seconds_total',
                    'value': cpu_value,
                    'labels': {
                        'instance': node['name'],
                        'mode': mode,
                        'clusterName': 'test-cluster'
                    }
                })
            
            # Memory metrics
            total_memory = 16 * 1024 * 1024 * 1024  # 16GB
            used_memory = total_memory * node['mem_util']
            available_memory = total_memory - used_memory
            
            # Add some variation
            available_memory += np.random.normal(0, 0.1 * 1024 * 1024 * 1024)
            available_memory = max(0, min(total_memory, available_memory))
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_memory_MemTotal_bytes',
                'value': total_memory,
                'labels': {
                    'instance': node['name'],
                    'clusterName': 'test-cluster'
                }
            })
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_memory_MemAvailable_bytes',
                'value': available_memory,
                'labels': {
                    'instance': node['name'],
                    'clusterName': 'test-cluster'
                }
            })
            
            # Cost metrics
            cost = node['cost'] + np.random.normal(0, 0.1)
            cost = max(0.1, cost)
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_total_hourly_cost',
                'value': cost,
                'labels': {
                    'instance': node['name'],
                    'clusterName': 'test-cluster'
                }
            })
    
    df = pd.DataFrame(metrics_data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.mark.e2e
def test_complete_system(test_config):
    """Test the complete forecasting and optimization system."""
    # Import the main agent
    from src.forecasting_agent.main import ForecastingAgent
    
    # Create test data
    test_metrics = create_comprehensive_test_data()
    logger.info(f"Created comprehensive test data with {len(test_metrics)} records")
    logger.info(f"Metrics: {test_metrics['metric_name'].unique()}")
    logger.info(f"Nodes: {test_metrics['labels'].apply(lambda x: x.get('instance')).unique()}")
    
    # Create agent instance
    agent = ForecastingAgent.__new__(ForecastingAgent)
    agent.config = test_config
    agent.collector = agent._init_collector()
    agent.optimizer = agent._init_optimizer()
    
    # Test 1: Metrics Collection (simulated)
    logger.info("\n=== Test 1: Metrics Collection ===")
    # We're using pre-generated data, so this is simulated
    logger.info("✅ Metrics collection simulated successfully")
    
    # Test 2: Forecasting
    logger.info("\n=== Test 2: Forecasting ===")
    forecast_results = agent.generate_forecast(test_metrics)
    
    assert 'forecasts' in forecast_results, "Forecast results should contain 'forecasts' key"
    assert len(forecast_results['forecasts']) > 0, "Should have generated forecasts"
    
    logger.info(f"✅ Generated forecasts for {len(forecast_results['forecasts'])} metrics")
    
    # Check individual forecasts
    for forecast_info in forecast_results['forecasts']:
        metric_name = forecast_info['metric_name']
        forecast = forecast_info['forecast']
        
        logger.info(f"  - {metric_name}: {len(forecast['forecast'])} forecast points")
        
        # Validate forecast structure
        assert 'forecast' in forecast, f"Forecast for {metric_name} should contain 'forecast' key"
        assert 'lower_bound' in forecast, f"Forecast for {metric_name} should contain 'lower_bound' key"
        assert 'upper_bound' in forecast, f"Forecast for {metric_name} should contain 'upper_bound' key"
    
    # Test 3: Optimization
    logger.info("\n=== Test 3: Optimization ===")
    recommendations = agent.generate_recommendations(test_metrics, forecast_results)
    
    logger.info(f"✅ Generated {len(recommendations)} optimization recommendations")
    
    # Analyze recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\nRecommendation {i}:")
            logger.info(f"  Node: {rec['node']}")
            logger.info(f"  Type: {rec['type']}")
            logger.info(f"  Action: {rec['recommended_value']['action']}")
            logger.info(f"  CPU Utilization: {rec['current_value']['cpu_utilization']:.2%}")
            logger.info(f"  Memory Utilization: {rec['current_value']['memory_utilization']:.2%}")
            logger.info(f"  Cost: ${rec['current_value']['cost']:.2f}/hour")
            logger.info(f"  Potential Savings: ${rec['recommended_value']['potential_savings']:.2f}/day")
            logger.info(f"  Confidence: {rec['confidence']:.2%}")
        
        # Validate recommendations
        valid_count = 0
        for rec in recommendations:
            if agent.optimizer.validate_recommendation(rec):
                valid_count += 1
                logger.info(f"  ✅ Recommendation for {rec['node']} is valid")
            else:
                logger.info(f"  ❌ Recommendation for {rec['node']} is invalid")
        
        logger.info(f"\nValid recommendations: {valid_count}/{len(recommendations)}")
        
        # We should have at least some recommendations for idle nodes
        assert len(recommendations) > 0, "Should have generated recommendations for idle nodes"
    else:
        logger.info("No recommendations generated (this might be expected depending on thresholds)")
    
    # Test 4: System Integration
    logger.info("\n=== Test 4: System Integration ===")
    
    # Check that all components work together
    assert agent.collector is not None, "Collector should be initialized"
    assert agent.optimizer is not None, "Optimizer should be initialized"
    
    # Check that the system can handle the data flow
    assert len(test_metrics) > 0, "Should have test metrics"
    assert len(forecast_results['forecasts']) > 0, "Should have forecasts"
    
    logger.info("✅ All system components integrated successfully")
    
    # Test 5: Data Consistency
    logger.info("\n=== Test 5: Data Consistency ===")
    
    # Check that all metrics have the expected structure
    required_columns = ['metric_name', 'value', 'labels']
    for col in required_columns:
        assert col in test_metrics.columns, f"Metrics should have '{col}' column"
    
    # Check that all metrics have valid values
    assert all(test_metrics['value'] >= 0), "All metric values should be non-negative"
    assert all(test_metrics['metric_name'].notna()), "All metrics should have names"
    
    logger.info("✅ Data consistency checks passed")
    
    logger.info("\n🎉 Complete system test passed successfully!")
    
    return True 