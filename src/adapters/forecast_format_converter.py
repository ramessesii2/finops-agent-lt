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
        
        # Extract data from TOTO forecast output
        quantile_results = toto_forecast_output.get('quantiles', {})
        timestamps = toto_forecast_output.get('timestamps', [])
        metric_names = toto_forecast_output.get('metric_names', [])
        horizon = toto_forecast_output.get('horizon', 0)
        
        if not quantile_results or not timestamps or not metric_names:
            raise ValueError("Invalid TOTO forecast output format")
        
        # Initialize result list
        cluster_forecasts = []
        node_name = node_name or "aggregated"
        extra_labels = extra_labels or {}
        
        # Convert each metric and quantile combination
        for metric_name in metric_names:
            if metric_name not in quantile_results:
                self.logger.warning(f"Missing quantile data for metric: {metric_name}")
                continue
                
            metric_quantiles = quantile_results[metric_name]
            # Simply append _forecast suffix to metric name - keep it simple
            forecast_name = self._get_forecast_name(metric_name)
            
            # Create forecast entry for each quantile
            for quantile_key, quantile_values in metric_quantiles.items():
                if not isinstance(quantile_values, list) or len(quantile_values) != len(timestamps):
                    self.logger.warning(f"Invalid quantile values for {metric_name} {quantile_key}")
                    continue
                
                # Extract quantile value from key (e.g., "q0.50" -> "0.50")
                quantile_str = quantile_key.replace('q', '')
                
                # Build metric labels
                metric_labels = {
                    "__name__": forecast_name,
                    "clusterName": cluster_name,
                    "node": node_name,
                    "quantile": quantile_str,
                    "horizon": self._format_horizon(horizon)
                }
                
                # Add extra labels if provided
                metric_labels.update(extra_labels)
                
                # Apply intelligent rounding for discrete metrics
                processed_values = self._process_values_for_metric_type(metric_name, quantile_values)
                
                # Create Prometheus-style forecast entry
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
    
    def convert_multi_cluster_forecasts(
        self,
        multi_cluster_toto_output: Dict[str, Dict[str, Any]],
        cluster_node_mapping: Optional[Dict[str, str]] = None,
        global_extra_labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert multiple cluster TOTO forecasts to cluster-grouped format.
        
        Args:
            multi_cluster_toto_output: Dict mapping cluster names to TOTO forecast outputs
            cluster_node_mapping: Optional mapping of cluster names to node names
            global_extra_labels: Optional labels to apply to all clusters
            
        Returns:
            Dict with cluster-grouped format: {cluster_name: [forecast_entries]}
        """
        if not multi_cluster_toto_output:
            raise ValueError("Empty multi-cluster TOTO output provided")
        
        cluster_node_mapping = cluster_node_mapping or {}
        global_extra_labels = global_extra_labels or {}
        
        cluster_grouped_results = {}
        
        for cluster_name, toto_output in multi_cluster_toto_output.items():
            try:
                node_name = cluster_node_mapping.get(cluster_name)
                
                cluster_forecasts = self.convert_to_cluster_grouped_format(
                    toto_forecast_output=toto_output,
                    cluster_name=cluster_name,
                    node_name=node_name,
                    extra_labels=global_extra_labels
                )
                
                cluster_grouped_results[cluster_name] = cluster_forecasts
                
            except Exception as e:
                self.logger.error(f"Failed to convert forecast for cluster {cluster_name}: {str(e)}")
                continue
        
        if not cluster_grouped_results:
            raise ValueError("No clusters could be converted successfully")
        
        self.logger.info(f"Successfully converted forecasts for {len(cluster_grouped_results)} clusters")
        return cluster_grouped_results
    
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


# Pure utility functions following functional programming principles

def convert_single_cluster_forecast(
    toto_forecast_output: Dict[str, Any],
    cluster_name: str,
    node_name: Optional[str] = None,
    extra_labels: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Pure function to convert a single cluster's TOTO forecast to Prometheus format.
    
    This is a functional interface to the converter for simple use cases.
    
    Args:
        toto_forecast_output: Pure TOTO forecast output
        cluster_name: Name of the cluster
        node_name: Optional node name
        extra_labels: Optional additional labels
        
    Returns:
        List of Prometheus-style forecast entries
    """
    converter = ForecastFormatConverter()
    return converter.convert_to_cluster_grouped_format(
        toto_forecast_output=toto_forecast_output,
        cluster_name=cluster_name,
        node_name=node_name,
        extra_labels=extra_labels
    )


def convert_multi_cluster_forecasts(
    multi_cluster_toto_output: Dict[str, Dict[str, Any]],
    cluster_node_mapping: Optional[Dict[str, str]] = None,
    global_extra_labels: Optional[Dict[str, str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pure function to convert multiple clusters' TOTO forecasts to cluster-grouped format.
    
    Args:
        multi_cluster_toto_output: Dict mapping cluster names to TOTO outputs
        cluster_node_mapping: Optional cluster to node mapping
        global_extra_labels: Optional global labels
        
    Returns:
        Cluster-grouped forecast results
    """
    converter = ForecastFormatConverter()
    return converter.convert_multi_cluster_forecasts(
        multi_cluster_toto_output=multi_cluster_toto_output,
        cluster_node_mapping=cluster_node_mapping,
        global_extra_labels=global_extra_labels
    )
