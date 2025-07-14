import logging
import numpy as np
from typing import Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)

try:
    from toto.data.util.dataset import MaskedTimeseries
except ImportError:
    # Mock for testing
    class MaskedTimeseries:
        def __init__(self, series, padding_mask, id_mask, timestamp_seconds, time_interval_seconds):
            self.series = series
            self.padding_mask = padding_mask
            self.id_mask = id_mask
            self.timestamp_seconds = timestamp_seconds
            self.time_interval_seconds = time_interval_seconds


def split_toto_data(toto_data: MaskedTimeseries, train_ratio: float = 0.7) -> Tuple[MaskedTimeseries, MaskedTimeseries]:
    """Split TOTO data into train/test sets.
    
    Args:
        toto_data: TOTO MaskedTimeseries object
        train_ratio: Training data fraction (default 0.7)
        
    Returns:
        (train_data, test_data) tuple
    """
    if toto_data.series is None:
        raise ValueError("TOTO data series is None")
    
    series_length = toto_data.series.shape[1]  # Shape: [batch, time, features]
    train_length = int(series_length * train_ratio)
    
    if train_length == 0 or train_length >= series_length:
        raise ValueError(f"Invalid split: train_length={train_length}, total_length={series_length}")
    
    train_series = toto_data.series[:, :train_length, :]
    test_series = toto_data.series[:, train_length:, :]
    
    train_padding_mask = toto_data.padding_mask[:, :train_length] if toto_data.padding_mask is not None else None
    test_padding_mask = toto_data.padding_mask[:, train_length:] if toto_data.padding_mask is not None else None
    
    train_timestamps = toto_data.timestamp_seconds[:train_length] if toto_data.timestamp_seconds is not None else None
    test_timestamps = toto_data.timestamp_seconds[train_length:] if toto_data.timestamp_seconds is not None else None
    
    # Create new MaskedTimeseries objects
    train_data = MaskedTimeseries(
        series=train_series,
        padding_mask=train_padding_mask,
        id_mask=toto_data.id_mask,
        timestamp_seconds=train_timestamps,
        time_interval_seconds=toto_data.time_interval_seconds
    )
    
    test_data = MaskedTimeseries(
        series=test_series,
        padding_mask=test_padding_mask,
        id_mask=toto_data.id_mask,
        timestamp_seconds=test_timestamps,
        time_interval_seconds=toto_data.time_interval_seconds
    )
    
    return train_data, test_data


def extract_actual_values(toto_data: MaskedTimeseries) -> Dict[str, np.ndarray]:
    """Extract actual values from TOTO data.
    
    Args:
        toto_data: TOTO MaskedTimeseries object
        
    Returns:
        Dict mapping metric names to value arrays
    """
    if toto_data.series is None:
        return {}
    
    values = {}
    series_array = toto_data.series.cpu().numpy() if hasattr(toto_data.series, 'cpu') else toto_data.series
    
    if len(series_array.shape) != 3:
        logger.warning(f"Unexpected series shape: {series_array.shape}, expected [batch, time, features]")
        return {}
    
    for feature_idx in range(series_array.shape[2]):
        metric_name = f"metric_{feature_idx}"
        values[metric_name] = series_array[0, :, feature_idx]
    
    return values


def extract_forecast_values(forecast_result: Dict[str, Any], quantile: str = "q0.50") -> Dict[str, np.ndarray]:
    """Extract forecast values from TOTO adapter result.
    
    Args:
        forecast_result: Result from TOTOAdapter.forecast()
        quantile: Quantile to extract (default "q0.50")
        
    Returns:
        Dict mapping metric names to forecast arrays
    """
    if not isinstance(forecast_result, dict):
        logger.warning(f"Expected dict forecast result, got {type(forecast_result)}")
        return {}
    
    forecast_values = {}
    
    for metric_name, metric_data in forecast_result.items():
        if isinstance(metric_data, dict) and quantile in metric_data:
            values = [point.get('y', 0) for point in metric_data[quantile]]
            forecast_values[metric_name] = np.array(values)
        elif isinstance(metric_data, (list, np.ndarray)):
            forecast_values[metric_name] = np.array(metric_data)
        else:
            logger.warning(f"Unsupported forecast format for metric {metric_name}: {type(metric_data)}")
    
    return forecast_values


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE as percentage
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Arrays must have same length: actual={len(actual)}, predicted={len(predicted)}")
    
    mask = actual != 0
    if not np.any(mask):
        logger.warning("All actual values are zero, returning infinite MAPE")
        return float('inf')
    
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return float(mape)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Mean Absolute Error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Arrays must have same length: actual={len(actual)}, predicted={len(predicted)}")
    
    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Root Mean Squared Error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Arrays must have same length: actual={len(actual)}, predicted={len(predicted)}")
    
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_validation_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate all validation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dict with mape, mae, and rmse values
    """
    if len(actual) == 0 or len(predicted) == 0:
        logger.warning("Empty arrays provided for metric calculation")
        return {'mape': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
    
    return {
        'mape': calculate_mape(actual, predicted),
        'mae': calculate_mae(actual, predicted),
        'rmse': calculate_rmse(actual, predicted)
    }


def validate_cluster_toto(adapter, toto_data: MaskedTimeseries, train_ratio: float = 0.7) -> Dict[str, Dict[str, float]]:
    """Validate forecasts for a single cluster.
    
    Args:
        adapter: TOTOAdapter instance
        toto_data: TOTO data for the cluster
        train_ratio: Train/test split ratio
        
    Returns:
        Dict with validation metrics per component
    """
    try:
        train_data, test_data = split_toto_data(toto_data, train_ratio)
        
        test_length = test_data.series.shape[1] if test_data.series is not None else 0
        if test_length == 0:
            return {"error": "No test data available"}
        
        forecast_result = adapter.forecast(train_data, horizon=test_length)
        
        actual_values = extract_actual_values(test_data)
        forecast_values = extract_forecast_values(forecast_result)
        cluster_metrics = {}
        for metric_name, actual_vals in actual_values.items():
            if metric_name in forecast_values:
                predicted_vals = forecast_values[metric_name]
                
                min_length = min(len(actual_vals), len(predicted_vals))
                if min_length > 0:
                    actual_vals = actual_vals[:min_length]
                    predicted_vals = predicted_vals[:min_length]
                    
                    cluster_metrics[metric_name] = calculate_validation_metrics(actual_vals, predicted_vals)
                else:
                    logger.warning(f"No valid data for metric {metric_name}")
            else:
                logger.warning(f"Metric {metric_name} not found in forecast results")
        
        return cluster_metrics
        
    except Exception as e:
        logger.error(f"Cluster validation failed: {str(e)}")
        return {"error": str(e)}


def validate_clusters_toto(cluster_toto_data: Dict[str, MaskedTimeseries], model_config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Validate forecasts for multiple clusters using TOTO format.
    
    Args:
        cluster_toto_data: Dictionary mapping cluster names to TOTO data
        model_config: TOTO model configuration
        
    Returns:
        Nested dictionary: {cluster: {component: {metric: value}}}
    """
    from adapters.forecasting.toto_adapter import TOTOAdapter
    
    validation_results = {}
    train_ratio = 0.7  # Following existing convention
    
    for cluster_name, toto_data in cluster_toto_data.items():
        logger.info(f"Validating cluster: {cluster_name} using TOTO format")
        
        try:
            # Create fresh adapter instance for each cluster
            adapter = TOTOAdapter(model_config)
            
            # Validate this cluster
            cluster_results = validate_cluster_toto(adapter, toto_data, train_ratio)
            validation_results[cluster_name] = cluster_results
            
            if "error" not in cluster_results:
                logger.info(f"Validation completed for cluster {cluster_name} with {len(cluster_results)} metrics")
            
        except Exception as e:
            logger.error(f"Validation failed for cluster {cluster_name}: {str(e)}")
            validation_results[cluster_name] = {"error": str(e)}
    
    if not validation_results:
        raise ValueError("No validation results produced")
    
    return validation_results


class TotoValidator:
    """TOTO-specific validator class following existing patterns."""
    
    def __init__(self, train_ratio: float = 0.7):
        """Initialize TOTO validator.
        
        Args:
            train_ratio: Fraction of data to use for training
        """
        self.train_ratio = train_ratio
    
    def validate_clusters(self, cluster_toto_data: Dict[str, MaskedTimeseries], model_config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Validate forecasts for all clusters.
        
        Args:
            cluster_toto_data: Dictionary mapping cluster names to TOTO data
            model_config: TOTO model configuration
            
        Returns:
            Validation results for all clusters
        """
        return validate_clusters_toto(cluster_toto_data, model_config)