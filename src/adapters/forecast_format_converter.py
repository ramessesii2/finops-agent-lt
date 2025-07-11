"""
Forecast Format Converter: Pure function to convert TOTO forecast output to cluster-grouped Prometheus format.

This module follows clean architecture principles with clear separation of concerns:
- Input: Pure TOTO forecast output
- Output: Cluster-grouped Prometheus-style format
- Pure function with no side effects
"""
import logging
from typing import Dict, Any, List, Optional
import math
from core.metric_types import MetricTypeClassifier


class ForecastFormatConverter:
    """Pure converter for transforming TOTO forecast output to cluster-grouped Prometheus format."""
    
    def __init__(self):
        """Initialize the converter."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Import MetricTypeClassifier for dynamic naming standards
        from core.metric_types import MetricTypeClassifier, MetricAggregationLevel
        self.metric_classifier = MetricTypeClassifier()
        self.MetricAggregationLevel = MetricAggregationLevel
    
    def _get_forecast_name(self, metric_name: str) -> str:
        """
        Generate forecast metric name by simply appending _forecast suffix.
        
        Keep it simple - just append _forecast to the input metric name.
        This follows clean architecture principles with minimal transformation.
        
        Args:
            metric_name: Original metric name from input series
            
        Returns:
            Forecast metric name: {metric_name}_forecast
        """
        # Keep it simple - just append _forecast suffix
        return f"{metric_name}_forecast"
    
    def convert_to_cluster_grouped_format(
        self,
        toto_forecast_output: Dict[str, Any],
        cluster_name: str,
        node_name: Optional[str] = None,
        extra_labels: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert pure TOTO forecast output to cluster-grouped Prometheus format.
        
        This is a pure function that transforms forecast data without side effects.
        
        Args:
            toto_forecast_output: Pure TOTO forecast output from TOTOAdapter.forecast()
            cluster_name: Name of the cluster for labeling
            node_name: Optional node name (defaults to "aggregated")
            extra_labels: Optional additional labels to include
            
        Returns:
            List of Prometheus-style forecast entries in the format:
            [
                {
                    "metric": {
                        "__name__": "cpu_usage_forecast",
                        "clusterName": "prod-eu1",
                        "node": "ip-10-0-3-42",
                        "quantile": "0.50",
                        "horizon": "7d"
                    },
                    "values": [0.27, 0.29, 0.31],
                    "timestamps": [1719955200000, 1719958800000, 1719962400000]
                },
                ...
            ]
        """
        if not toto_forecast_output:
            raise ValueError("Empty TOTO forecast output provided")
        
        if not cluster_name:
            raise ValueError("Cluster name is required")
        
        series_results = toto_forecast_output.get('series', {})
        timestamps = toto_forecast_output.get('timestamps', [])
        variate_metadata = toto_forecast_output.get('variate_metadata', [])
        horizon = toto_forecast_output.get('horizon', 0)
        
        metric_names = toto_forecast_output.get('metric_names', [])
        
        # Debug logging to understand actual format
        self.logger.debug(f"TOTO output keys: {list(toto_forecast_output.keys())}")
        self.logger.debug(f"Series results type: {type(series_results)}, length: {len(series_results) if series_results else 0}")
        self.logger.debug(f"Timestamps type: {type(timestamps)}, length: {len(timestamps) if timestamps else 0}")
        self.logger.debug(f"Variate metadata type: {type(variate_metadata)}, length: {len(variate_metadata) if variate_metadata else 0}")
        
        if series_results:
            self.logger.debug(f"First few series result keys: {list(series_results.keys())[:3]}")
            if series_results:
                first_key = next(iter(series_results.keys()))
                first_value = series_results[first_key]
                self.logger.debug(f"Sample series result structure for {first_key}: {list(first_value.keys()) if isinstance(first_value, dict) else type(first_value)}")
        
        if not series_results or not timestamps:
            self.logger.error(f"VALIDATION FAILED - series_results: {bool(series_results)}, timestamps: {bool(timestamps)}")
            raise ValueError("Invalid TOTO forecast output format")
        
        if variate_metadata:
            unique_metrics = set()
            for metadata in variate_metadata:
                unique_metrics.add(metadata['metric_name'])
            metric_names = list(unique_metrics)
        elif not metric_names:
            raise ValueError("No variate_metadata or metric_names found in TOTO forecast output")
        
        # Initialize result list
        cluster_forecasts = []
        node_name = node_name or "aggregated"
        extra_labels = extra_labels or {}
        
        # Convert each metric and quantile combination
        if variate_metadata:
            # Process variate_id-based series structure (renamed from quantiles for clarity)
            for variate_id, variate_data in series_results.items():
                if 'metadata' not in variate_data or 'quantiles' not in variate_data:
                    self.logger.warning(f"Invalid variate data for {variate_id}")
                    continue
                
                metadata = variate_data['metadata']
                metric_name = metadata['metric_name']
                variate_quantiles = variate_data['quantiles']
                
                # Simply append _forecast suffix to metric name - keep it simple
                forecast_name = self._get_forecast_name(metric_name)
                
                # Create forecast entry for each quantile
                for quantile_key, quantile_values in variate_quantiles.items():
                    if not isinstance(quantile_values, list) or len(quantile_values) != len(timestamps):
                        self.logger.warning(f"Invalid quantile values for {variate_id} {quantile_key}")
                        continue
                    
                    # Extract quantile value from key (e.g., "q0.50" -> "0.50")
                    quantile_str = quantile_key.replace('q', '')
                    
                    # Build metric labels with node name from metadata
                    metric_labels = {
                        "__name__": forecast_name,
                        "clusterName": cluster_name,
                        "node": metadata.get('node_name', node_name),
                        "quantile": quantile_str,
                        "horizon": f"{horizon//24}d" if horizon >= 24 else f"{horizon}h"
                    }
                    
                    # Add extra labels
                    metric_labels.update(extra_labels)
                    
                    # Process values for metric type (intelligent rounding for discrete metrics)
                    processed_values = self._process_values_for_metric_type(metric_name, quantile_values)
                    
                    # Create forecast entry
                    forecast_entry = {
                        "metric": metric_labels,
                        "values": processed_values,
                        "timestamps": timestamps
                    }
                    
                    cluster_forecasts.append(forecast_entry)

        if not cluster_forecasts:
            raise ValueError(f"No valid forecast entries generated for cluster: {cluster_name}")
        
        self.logger.debug(f"Generated {len(cluster_forecasts)} forecast entries for cluster: {cluster_name}")
        return cluster_forecasts
    
    def _format_horizon(self, horizon_hours: int) -> str:
        """
        Format horizon in human-readable format.
        
        Pure function to convert hours to appropriate unit string.
        
        Args:
            horizon_hours: Horizon in hours
            
        Returns:
            Formatted string (e.g., "7d", "168h", "2w")
        """
        if horizon_hours <= 0:
            return "0h"
        
        if horizon_hours % (24 * 7) == 0:  # Weeks
            weeks = horizon_hours // (24 * 7)
            return f"{weeks}w"
        elif horizon_hours % 24 == 0:  # Days
            days = horizon_hours // 24
            return f"{days}d"
        else:  # Hours
            return f"{horizon_hours}h"
    
    def _process_values_for_metric_type(self, metric_name: str, values: List[float]) -> List[float]:
        """
        Discrete metrics (node_count, cpu_total_cores) are rounded to integers.
        Continuous metrics (percentages, costs) maintain float precision.
        All positive-only metrics are capped at >= 0 to prevent meaningless negative values.
        
        Args:
            metric_name: Name of the metric to determine processing type
            values: Raw forecast values from TOTO model
            
        Returns:
            Processed values with appropriate data types and non-negative constraints
        """
        # Define discrete metrics that should be integers
        discrete_metrics = {
            'node_count_per_cluster',
            'cpu_total_cores_per_node',
            # Future discrete metrics can be added here
            # 'pod_count',
            # 'container_count',
        }
        
        # Define positive-only metrics that should never be negative
        positive_metrics = {
            # Cost metrics - negative costs are meaningless
            'cost_usd_per_cluster',
            'cost_usd_per_node',
            
            # Percentage metrics - negative percentages are invalid
            'cpu_pct_per_cluster',
            'cpu_pct_per_node',
            'mem_pct_per_cluster', 
            'mem_pct_per_node',
            
            # Count metrics - negative counts are impossible
            'node_count_per_cluster',
            'cpu_total_cores_per_node',
            
            # Memory/storage metrics - negative capacity is meaningless
            'mem_total_gb_per_cluster',
            
            # Future positive-only metrics can be added here
            # 'disk_usage_gb',
            # 'network_bandwidth_mbps',
        }
        
        if metric_name in discrete_metrics:
            # Round discrete metrics to integers and ensure non-negative
            processed_values = [max(0, round(value)) for value in values]
            self.logger.debug(f"Applied integer rounding and non-negative capping to discrete metric: {metric_name}")
            return processed_values
        elif metric_name in positive_metrics:
            # Cap positive-only metrics to ensure non-negative values
            processed_values = [max(0.0, round(value, 6)) for value in values]
            self.logger.debug(f"Applied non-negative capping to positive metric: {metric_name}")
            return processed_values
        else:
            processed_values = [round(value, 6) for value in values]
            return processed_values
