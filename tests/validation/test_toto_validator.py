"""
Test-driven development for TOTO validation functionality.

Following coding-conventions.md: Tests must be written first before any production code.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestTotoValidator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with realistic data."""
        # Create realistic TOTO data structure with actual tensor operations
        self.real_toto_data = self._create_realistic_toto_data()
        
        # Create mock for comparison where needed
        self.mock_toto_data = Mock()
        self.mock_toto_data.series = torch.randn(1, 24, 3)  # [batch, time, features]
        self.mock_toto_data.padding_mask = torch.ones(1, 24, dtype=torch.bool)
        self.mock_toto_data.id_mask = torch.ones(1, dtype=torch.bool)
        self.mock_toto_data.timestamp_seconds = torch.arange(24) * 3600
        self.mock_toto_data.time_interval_seconds = 3600
        
        self.model_config = {
            'horizon': 12,
            'quantiles': [0.1, 0.5, 0.9]
        }
    
    def _create_realistic_toto_data(self):
        """Create realistic TOTO data with actual patterns for testing core behaviors."""
        # Import the actual MaskedTimeseries or use fallback
        try:
            from toto.data.util.dataset import MaskedTimeseries
        except ImportError:
            class MaskedTimeseries:
                def __init__(self, series, padding_mask, id_mask, timestamp_seconds, time_interval_seconds):
                    self.series = series
                    self.padding_mask = padding_mask
                    self.id_mask = id_mask
                    self.timestamp_seconds = timestamp_seconds
                    self.time_interval_seconds = time_interval_seconds
        
        # Create realistic time series data with patterns
        time_points = 48  # 48 hours of data
        features = 2  # cpu_usage, memory_usage
        
        # Set random seed for reproducible test data
        np.random.seed(42)
        
        # Generate realistic CPU usage pattern (daily cycle) with controlled bounds
        cpu_base = np.array([0.3 + 0.2 * np.sin(2 * np.pi * i / 24) for i in range(time_points)])
        cpu_noise = 0.05 * np.random.randn(time_points)  # Smaller noise factor
        cpu_pattern = np.clip(cpu_base + cpu_noise, 0.0, 1.0)  # Ensure [0,1] bounds
        
        # Generate realistic memory usage pattern (gradual increase with noise)
        memory_base = np.array([0.4 + 0.001 * i for i in range(time_points)])
        memory_noise = 0.03 * np.random.randn(time_points)  # Controlled noise
        memory_pattern = np.clip(memory_base + memory_noise, 0.0, 1.0)  # Ensure [0,1] bounds
        
        # Stack patterns into tensor format [batch, time, features]
        series_data = np.stack([cpu_pattern, memory_pattern], axis=1)
        series_tensor = torch.tensor(series_data, dtype=torch.float32).unsqueeze(0)
        
        # Create proper masks and timestamps
        padding_mask = torch.ones(1, time_points, dtype=torch.bool)
        id_mask = torch.ones(1, dtype=torch.bool)
        timestamps = torch.arange(time_points) * 3600  # Hourly data
        
        return MaskedTimeseries(
            series=series_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=3600
        )
    
    def test_split_toto_data_should_split_70_30_with_real_data(self):
        """Test TOTO data splitting with realistic data and verify data integrity."""
        from validation.toto_validator import split_toto_data
        
        # Test with real data
        train_data, test_data = split_toto_data(self.real_toto_data, train_ratio=0.7)
        
        # Verify split ratios with real data
        total_length = self.real_toto_data.series.shape[1]  # 48 time points
        expected_train_length = int(total_length * 0.7)  # 33 points
        expected_test_length = total_length - expected_train_length  # 15 points
        
        self.assertEqual(train_data.series.shape[1], expected_train_length)
        self.assertEqual(test_data.series.shape[1], expected_test_length)
        
        # Verify data continuity (test starts where train ends)
        original_data = self.real_toto_data.series[0, :, 0].numpy()  # First feature
        train_end = train_data.series[0, -1, 0].item()
        test_start = test_data.series[0, 0, 0].item()
        expected_test_start = original_data[expected_train_length]
        
        self.assertAlmostEqual(test_start, expected_test_start, places=5)
        
        # Verify no data loss
        reconstructed = torch.cat([train_data.series, test_data.series], dim=1)
        torch.testing.assert_close(reconstructed, self.real_toto_data.series)
    
    def test_split_toto_data_should_preserve_feature_patterns(self):
        """Test that splitting preserves the underlying data patterns."""
        from validation.toto_validator import split_toto_data
        
        train_data, test_data = split_toto_data(self.real_toto_data, train_ratio=0.7)
        
        # Extract CPU usage patterns (feature 0)
        original_cpu = self.real_toto_data.series[0, :, 0].numpy()
        train_cpu = train_data.series[0, :, 0].numpy()
        test_cpu = test_data.series[0, :, 0].numpy()
        
        # Verify train data maintains original pattern
        np.testing.assert_array_almost_equal(train_cpu, original_cpu[:len(train_cpu)])
        
        # Verify test data maintains original pattern
        np.testing.assert_array_almost_equal(test_cpu, original_cpu[len(train_cpu):])
        
        # Verify patterns make sense (CPU usage should be between 0 and 1)
        self.assertTrue(np.all(train_cpu >= 0) and np.all(train_cpu <= 1))
        self.assertTrue(np.all(test_cpu >= 0) and np.all(test_cpu <= 1))
    
    def test_extract_actual_values_should_return_dict_of_arrays(self):
        """Test extracting actual values from TOTO data returns metric dict."""
        from validation.toto_validator import extract_actual_values
        
        actual_values = extract_actual_values(self.mock_toto_data)
        
        # Should return dict with metric names as keys, numpy arrays as values
        self.assertIsInstance(actual_values, dict)
        self.assertGreater(len(actual_values), 0)
        for metric_name, values in actual_values.items():
            self.assertIsInstance(values, np.ndarray)
    
    def test_extract_forecast_values_should_handle_toto_result(self):
        """Test extracting forecast values from TOTO adapter result."""
        from validation.toto_validator import extract_forecast_values
        
        # Mock TOTO forecast result structure
        mock_forecast_result = {
            'cpu_usage': {
                'q0.50': [{'y': 0.5}, {'y': 0.6}, {'y': 0.7}]
            },
            'memory_usage': {
                'q0.50': [{'y': 0.8}, {'y': 0.9}, {'y': 1.0}]
            }
        }
        
        forecast_values = extract_forecast_values(mock_forecast_result)
        
        self.assertIsInstance(forecast_values, dict)
        self.assertIn('cpu_usage', forecast_values)
        self.assertIn('memory_usage', forecast_values)
        self.assertEqual(len(forecast_values['cpu_usage']), 3)
        self.assertEqual(len(forecast_values['memory_usage']), 3)
    
    def test_calculate_validation_metrics_should_return_mape_mae_rmse(self):
        """Test calculation of validation metrics."""
        from validation.toto_validator import calculate_validation_metrics
        
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 2.1, 2.9, 3.8])
        
        metrics = calculate_validation_metrics(actual, predicted)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mape', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIsInstance(metrics['mape'], float)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['rmse'], float)
    
    def test_validate_cluster_toto_should_return_metrics_dict(self):
        """Test cluster validation returns metrics for each metric."""
        from validation.toto_validator import validate_cluster_toto
        from adapters.forecasting.toto_adapter import TOTOAdapter
        
        with patch.object(TOTOAdapter, 'forecast') as mock_forecast:
            mock_forecast.return_value = {
                'cpu_usage': {
                    'q0.50': [{'y': 0.5}, {'y': 0.6}]
                }
            }
            
            adapter = TOTOAdapter(self.model_config)
            result = validate_cluster_toto(adapter, self.mock_toto_data)
            
            self.assertIsInstance(result, dict)
    
    def test_validate_clusters_toto_should_handle_multiple_clusters(self):
        """Test validation of multiple clusters."""
        from validation.toto_validator import validate_clusters_toto
        
        cluster_data = {
            'cluster1': self.mock_toto_data,
            'cluster2': self.mock_toto_data
        }
        
        with patch('validation.toto_validator.validate_cluster_toto') as mock_validate:
            mock_validate.return_value = {'cpu_usage': {'mape': 0.1}}
            
            results = validate_clusters_toto(cluster_data, self.model_config)
            
            self.assertIsInstance(results, dict)
            self.assertIn('cluster1', results)
            self.assertIn('cluster2', results)
    
    def test_validate_cluster_toto_should_handle_adapter_errors(self):
        """Test error handling in cluster validation."""
        from validation.toto_validator import validate_cluster_toto
        from adapters.forecasting.toto_adapter import TOTOAdapter
        
        with patch.object(TOTOAdapter, 'forecast', side_effect=Exception("Forecast failed")):
            adapter = TOTOAdapter(self.model_config)
            result = validate_cluster_toto(adapter, self.mock_toto_data)
            
            # Should return error information instead of crashing
            self.assertIn('error', result)
    
    def test_calculate_mape_should_handle_zero_division(self):
        """Test MAPE calculation handles division by zero."""
        from validation.toto_validator import calculate_mape
        
        actual = np.array([0.0, 1.0, 2.0])
        predicted = np.array([1.0, 1.1, 2.1])
        
        mape = calculate_mape(actual, predicted)
        
        # Should handle zero division gracefully
        self.assertIsInstance(mape, float)
        self.assertFalse(np.isnan(mape))
    
    def test_calculate_mae_should_return_mean_absolute_error(self):
        """Test MAE calculation."""
        from validation.toto_validator import calculate_mae
        
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.1, 2.9])
        
        expected_mae = np.mean(np.abs(actual - predicted))
        mae = calculate_mae(actual, predicted)
        
        self.assertAlmostEqual(mae, expected_mae, places=5)
    
    def test_calculate_rmse_should_return_root_mean_squared_error(self):
        """Test RMSE calculation."""
        from validation.toto_validator import calculate_rmse
        
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.1, 2.9])
        
        expected_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        rmse = calculate_rmse(actual, predicted)
        
        self.assertAlmostEqual(rmse, expected_rmse, places=5)


class TestTotoValidatorIntegration(unittest.TestCase):
    """Integration tests for TOTO validation workflow."""
    
    def test_full_validation_workflow_should_work_end_to_end(self):
        """Test complete validation workflow from TOTO data to results."""
        from validation.toto_validator import TotoValidator
        
        # Mock cluster TOTO data
        cluster_data = {
            'test-cluster': Mock()
        }
        cluster_data['test-cluster'].series = torch.randn(1, 20, 2)
        
        model_config = {'horizon': 5, 'quantiles': [0.5]}
        
        validator = TotoValidator()
        
        with patch('adapters.forecasting.toto_adapter.TOTOAdapter'):
            results = validator.validate_clusters(cluster_data, model_config)
            
            # Should return structured results
            self.assertIsInstance(results, dict)
            self.assertIn('test-cluster', results)


if __name__ == '__main__':
    unittest.main()