#!/usr/bin/env python3
"""
Test script for idle capacity optimizer with proper metrics handling.
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
    """Create test metrics data that matches the actual Prometheus format."""
    # Generate sample time series data
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='1h')
    
    metrics_data = []
    
    # Create test nodes
    nodes = ['node-1', 'node-2', 'node-3']
    
    for node in nodes:
        for date in dates:
            # CPU metrics with different modes
            cpu_modes = ['user', 'system', 'idle', 'iowait']
            for mode in cpu_modes:
                # Generate realistic CPU values
                if mode == 'idle':
                    # Idle time should be high for idle nodes
                    base_value = 0.7 if node == 'node-1' else 0.3
                    cpu_value = base_value + np.random.normal(0, 0.1)
                else:
                    # Active CPU time
                    base_value = 0.1 if node == 'node-1' else 0.4
                    cpu_value = base_value + np.random.normal(0, 0.05)
                
                cpu_value = max(0, cpu_value)
                
                metrics_data.append({
                    'timestamp': date,
                    'metric_name': 'node_cpu_seconds_total',
                    'value': cpu_value,
                    'labels': {
                        'instance': node,
                        'mode': mode,
                        'clusterName': 'test-cluster'
                    }
                })
            
            # Memory metrics
            if node == 'node-1':
                # Node 1: High available memory (low utilization)
                total_memory = 16 * 1024 * 1024 * 1024  # 16GB
                available_memory = 12 * 1024 * 1024 * 1024  # 12GB available
            else:
                # Other nodes: Lower available memory (higher utilization)
                total_memory = 16 * 1024 * 1024 * 1024  # 16GB
                available_memory = 4 * 1024 * 1024 * 1024  # 4GB available
            
            # Add some variation
            available_memory += np.random.normal(0, 0.1 * 1024 * 1024 * 1024)
            available_memory = max(0, min(total_memory, available_memory))
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_memory_MemTotal_bytes',
                'value': total_memory,
                'labels': {
                    'instance': node,
                    'clusterName': 'test-cluster'
                }
            })
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_memory_MemAvailable_bytes',
                'value': available_memory,
                'labels': {
                    'instance': node,
                    'clusterName': 'test-cluster'
                }
            })
            
            # Cost metrics
            if node == 'node-1':
                cost = 0.5  # Low cost node
            else:
                cost = 2.0  # Higher cost nodes
            
            metrics_data.append({
                'timestamp': date,
                'metric_name': 'node_total_hourly_cost',
                'value': cost,
                'labels': {
                    'instance': node,
                    'clusterName': 'test-cluster'
                }
            })
    
    df = pd.DataFrame(metrics_data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.mark.integration
def test_idle_capacity_optimizer(test_config):
    """Test the idle capacity optimizer functionality."""
    # Import the optimizer
    from src.forecasting_agent.optimizers.idle_capacity import IdleCapacityOptimizer
    
    # Create test data
    test_metrics = create_test_metrics_data()
    logger.info(f"Created test data with {len(test_metrics)} records")
    logger.info(f"Metrics: {test_metrics['metric_name'].unique()}")
    logger.info(f"Nodes: {test_metrics['labels'].apply(lambda x: x.get('instance')).unique()}")
    
    # Create optimizer instance
    optimizer = IdleCapacityOptimizer(test_config['optimizer'])
    
    # Test analysis
    recommendations = optimizer.analyze(test_metrics)
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    
    # Assert that we got at least one recommendation
    assert len(recommendations) > 0, "Should generate at least one recommendation"
    
    # Print recommendations
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
        logger.info(f"  Data Points: {rec['details']['data_points']}")
    
    # Test validation
    valid_recommendations = []
    for rec in recommendations:
        if optimizer.validate_recommendation(rec):
            valid_recommendations.append(rec)
            logger.info(f"✅ Recommendation for {rec['node']} is valid")
        else:
            logger.info(f"❌ Recommendation for {rec['node']} is invalid")
    
    logger.info(f"\nValid recommendations: {len(valid_recommendations)}/{len(recommendations)}")
    
    # Assert that we have some recommendations (even if not all are valid)
    assert len(recommendations) > 0, "Should have generated recommendations" 