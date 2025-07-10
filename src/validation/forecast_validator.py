"""
Forecast validation utilities for evaluating model accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from darts import TimeSeries
import logging

logger = logging.getLogger(__name__)


class ForecastValidator:
    """Validates forecast accuracy using train/test splits."""
    
    def __init__(self, train_ratio: float = 0.7):
        """
        Initialize validator.
        
        Args:
            train_ratio: Fraction of data to use for training (default 0.7 for 70/30 split)
        """
        self.train_ratio = train_ratio
        
    def split_timeseries(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Split time series into train/test sets.
        
        Args:
            series: Input time series
            
        Returns:
            Tuple of (train_series, test_series)
        """
        total_length = len(series)
        train_length = int(total_length * self.train_ratio)
        
        train_series = series[:train_length]
        test_series = series[train_length:]
        
        logger.info(f"Split series: {train_length} train points, {len(test_series)} test points")
        return train_series, test_series
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MAPE as percentage
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            logger.warning("All actual values are zero, cannot calculate MAPE")
            return float('inf')
            
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return float(mape)
    
    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(actual - predicted)))
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    
    def extract_forecast_values(self, forecast_dict: Dict[str, Any], quantile: str = "q0.50") -> Dict[str, np.ndarray]:
        """
        Extract forecast values from forecast dictionary.
        
        Args:
            forecast_dict: Forecast results from adapter
            quantile: Which quantile to extract (default median)
            
        Returns:
            Dictionary mapping component names to forecast arrays
        """
        forecast_values = {}
        
        for component, comp_data in forecast_dict.items():
            if quantile in comp_data:
                values = [point["y"] for point in comp_data[quantile]]
                forecast_values[component] = np.array(values)
            else:
                logger.warning(f"Quantile {quantile} not found for component {component}")
                
        return forecast_values
    
    def validate_forecast(self, 
                         adapter,
                         series: TimeSeries, 
                         horizon: Optional[int] = None,
                         quantiles: Optional[list] = None) -> Dict[str, Dict[str, float]]:
        """
        Validate forecast accuracy using train/test split.
        
        Args:
            adapter: Forecasting adapter (NBEATS or Toto)
            series: Full time series data
            horizon: Forecast horizon (if None, uses test set length)
            quantiles: Quantiles for forecast
            
        Returns:
            Dictionary with validation metrics per component
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]
        
        # Split data
        train_series, test_series = self.split_timeseries(series)
        
        # Use test series length as horizon if not specified
        if horizon is None:
            horizon = len(test_series)
        else:
            # Truncate test series to match horizon
            test_series = test_series[:horizon]
        
        logger.info(f"Validating with horizon={horizon}, test_length={len(test_series)}")
        
        # Fit adapter on training data and generate forecast
        if hasattr(adapter, 'fit'):
            adapter.fit(train_series)
            
        forecast_dict = adapter.forecast(train_series, horizon, quantiles)
        
        # Extract forecast values (using median quantile)
        forecast_values = self.extract_forecast_values(forecast_dict, "q0.50")
        
        # Calculate metrics for each component
        validation_results = {}
        
        for component in series.components:
            if component in forecast_values:
                # Get actual test values for this component
                actual_values = test_series[component].values(copy=False).flatten()
                predicted_values = forecast_values[component]
                
                # Ensure same length
                min_length = min(len(actual_values), len(predicted_values))
                actual_values = actual_values[:min_length]
                predicted_values = predicted_values[:min_length]
                
                # Calculate metrics
                mape = self.calculate_mape(actual_values, predicted_values)
                mae = self.calculate_mae(actual_values, predicted_values)
                rmse = self.calculate_rmse(actual_values, predicted_values)
                
                validation_results[component] = {
                    'MAPE': mape,
                    'MAE': mae,
                    'RMSE': rmse,
                    'test_points': len(actual_values)
                }
                
                logger.info(f"Component {component}: MAPE={mape:.2f}%, MAE={mae:.4f}, RMSE={rmse:.4f}")
            else:
                logger.warning(f"Component {component} not found in forecast results")
        
        return validation_results
    
    def validate_all_clusters(self, 
                             adapter_class,
                             cluster_timeseries: Dict[str, TimeSeries],
                             config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Validate forecasts for all clusters.
        
        Args:
            adapter_class: Adapter class to instantiate
            cluster_timeseries: Dictionary of cluster name to TimeSeries
            config: Model configuration
            
        Returns:
            Nested dictionary: {cluster: {component: {metric: value}}}
        """
        all_results = {}
        
        for cluster_name, series in cluster_timeseries.items():
            logger.info(f"Validating cluster: {cluster_name}")
            
            try:
                # Create fresh adapter instance for each cluster
                adapter = adapter_class(config)
                
                # Validate this cluster
                cluster_results = self.validate_forecast(
                    adapter, 
                    series,
                    quantiles=config.get('quantiles', [0.1, 0.5, 0.9])
                )
                
                all_results[cluster_name] = cluster_results
                
            except Exception as e:
                logger.error(f"Validation failed for cluster {cluster_name}: {str(e)}")
                all_results[cluster_name] = {"error": str(e)}
        
        return all_results
    
    def summarize_validation_results(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """
        Create summary DataFrame of validation results.
        
        Args:
            results: Validation results from validate_all_clusters
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for cluster, cluster_results in results.items():
            if "error" in cluster_results:
                continue
                
            for component, metrics in cluster_results.items():
                summary_data.append({
                    'cluster': cluster,
                    'component': component,
                    'MAPE': metrics.get('MAPE', np.nan),
                    'MAE': metrics.get('MAE', np.nan),
                    'RMSE': metrics.get('RMSE', np.nan),
                    'test_points': metrics.get('test_points', 0)
                })
        
        df = pd.DataFrame(summary_data)
        
        if not df.empty:
            # Add overall statistics
            logger.info("Validation Summary:")
            logger.info(f"Average MAPE: {df['MAPE'].mean():.2f}%")
            logger.info(f"Average MAE: {df['MAE'].mean():.4f}")
            logger.info(f"Average RMSE: {df['RMSE'].mean():.4f}")
        
        return df
