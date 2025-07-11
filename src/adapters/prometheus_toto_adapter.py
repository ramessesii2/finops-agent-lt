"""
PrometheusToTotoAdapter: Direct conversion from raw Prometheus JSON to TOTO tensor format.

This adapter eliminates the inefficient Raw JSON → TimeSeries → Tensors conversion
by going directly from Raw JSON → Tensors, filtered by cluster name.
"""
import logging
from typing import Dict, Any, List
import torch
import pandas as pd

# Import TOTO's MaskedTimeseries - conditional import to handle missing toto module
try:
    from toto.data.util.dataset import MaskedTimeseries
except ImportError:
    # Create a mock class for testing when toto module is not available
    class MaskedTimeseries:
        def __init__(self, series, padding_mask, id_mask, timestamp_seconds, time_interval_seconds):
            self.series = series
            self.padding_mask = padding_mask
            self.id_mask = id_mask
            self.timestamp_seconds = timestamp_seconds
            self.time_interval_seconds = time_interval_seconds


class PrometheusToTotoAdapter:
    """Adapter to convert raw Prometheus JSON directly to TOTO tensor format."""
    
    def __init__(self):
        """Initialize the adapter."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def convert_to_toto_format(
        self, 
        prometheus_data: Dict[str, List[Dict[str, Any]]], 
        cluster_name: str
    ) -> MaskedTimeseries:
        """
        Convert raw Prometheus data to TOTO MaskedTimeseries format for a specific cluster.
        
        Args:
            prometheus_data: Raw Prometheus query results
            cluster_name: Name of the cluster to filter for
            
        Returns:
            MaskedTimeseries: TOTO-compatible tensor format
            
        Raises:
            ValueError: If data is empty, malformed, or cluster not found
        """
        self._validate_input(prometheus_data, cluster_name)
        
        # Extract cluster-specific data from all metrics
        cluster_data = self._extract_cluster_data(prometheus_data, cluster_name)
        
        if not cluster_data:
            raise ValueError(f"No data found for cluster: {cluster_name}")
        
        # Build node-metric combinations first for both tensor building and metadata
        node_metric_combinations = []
        for metric_name, metric_values in cluster_data['metrics'].items():
            # Get all unique nodes for this metric
            nodes_for_metric = set(item['node_name'] for item in metric_values)
            for node_name in sorted(nodes_for_metric):  # Sort for consistent ordering
                node_metric_combinations.append((metric_name, node_name))
        
        # Convert to tensor format using node-metric combinations
        masked_timeseries = self._build_masked_timeseries(cluster_data['metrics'], node_metric_combinations)
        
        # Create variate metadata for each node-metric combination
        variate_metadata = []
        for metric_name, node_name in node_metric_combinations:
            variate_metadata.append({
                'metric_name': metric_name,
                'node_name': node_name,
                'variate_id': f"{metric_name}_{node_name}"
            })
        
        return {
            'masked_timeseries': masked_timeseries,
            'node_names': cluster_data['node_names'],
            'metric_names': list(cluster_data['metrics'].keys()),
            'variate_metadata': variate_metadata
        }
    
    def _validate_input(self, prometheus_data: Dict[str, List[Dict[str, Any]]], cluster_name: str):
        """Validate input data."""
        if not prometheus_data:
            raise ValueError("Empty Prometheus data provided")
        
        if not cluster_name:
            raise ValueError("Cluster name cannot be empty")
    
    def _extract_cluster_data(
        self, 
        prometheus_data: Dict[str, List[Dict[str, Any]]], 
        cluster_name: str
    ) -> Dict[str, Any]:
        """
        Extract time series data for the specified cluster from raw Prometheus results.
        
        Returns:
            Dict containing 'metrics' (timestamp-value data) and 'node_names' (extracted node names)
        """
        cluster_data = {}
        node_names = set()  # Track unique node names for this cluster
        
        for metric_name, metric_results in prometheus_data.items():
            metric_values = []
            
            for result in metric_results:
                try:
                    # Check if this result is for our target cluster
                    metric_labels = result.get("metric", {})
                    if metric_labels.get("clusterName") == cluster_name:
                        # Extract node name with priority: "node" > "nodename" > "instance"
                        # OpenCost metrics use "node", Kubernetes metrics use "nodename" or "instance"
                        node_name = (
                            metric_labels.get("node") or 
                            metric_labels.get("nodename") or 
                            metric_labels.get("instance")
                        )
                        if node_name:
                            node_names.add(node_name)
                        
                        # Check for required 'values' key - this is malformed data if missing
                        if "values" not in result:
                            raise ValueError(f"Malformed Prometheus data: missing 'values' key in result for {metric_name}")
                        
                        values = result.get("values", [])
                        if not values:
                            continue
                            
                        # Extract timestamp-value pairs with node name
                        for timestamp, value in values:
                            try:
                                metric_values.append({
                                    'timestamp': int(timestamp),
                                    'value': float(value),
                                    'node_name': node_name or 'cluster-aggregate'
                                })
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Skipping invalid value in {metric_name}: {e}")
                                continue
                                
                except KeyError as e:
                    raise ValueError(f"Malformed Prometheus data: missing key {e}")
            
            if metric_values:
                # Sort by timestamp to ensure chronological order
                metric_values.sort(key=lambda x: x['timestamp'])
                cluster_data[metric_name] = metric_values
        
        return {
            'metrics': cluster_data,
            'node_names': list(node_names) if node_names else ['cluster-aggregate']
        }
    
    def _build_masked_timeseries(self, cluster_data: Dict[str, List[Dict[str, Any]]], node_metric_combinations: List[tuple]) -> MaskedTimeseries:
        """
        Build MaskedTimeseries tensor from cluster data.
        
        Args:
            cluster_data: Dict mapping metric names to list of timestamp-value-node dictionaries
            
        Returns:
            MaskedTimeseries: TOTO-compatible tensor format
        """
        if not cluster_data:
            raise ValueError("No cluster data to convert")
        
        # Get all unique timestamps and sort them
        all_timestamps = set()
        for metric_values in cluster_data.values():
            all_timestamps.update(item['timestamp'] for item in metric_values)
        sorted_timestamps = sorted(all_timestamps)
        
        if len(sorted_timestamps) < 2:
            # If we only have one timestamp, assume 1-hour interval
            time_interval = 3600
        else:
            # Calculate time interval from first two timestamps
            time_interval = sorted_timestamps[1] - sorted_timestamps[0]
        
        # Use passed node-metric combinations to preserve node-specific data
        # node_metric_combinations is now passed as parameter to avoid duplication
        n_variates = len(node_metric_combinations)
        n_timesteps = len(sorted_timestamps)
        
        # Initialize tensors
        series_data = torch.zeros((n_variates, n_timesteps), dtype=torch.float32)
        padding_mask = torch.full((n_variates, n_timesteps), True, dtype=torch.bool)
        id_mask = torch.zeros((n_variates, n_timesteps), dtype=torch.float32)
        
        timestamp_to_idx = {ts: idx for idx, ts in enumerate(sorted_timestamps)}
        
        for variate_idx, (metric_name, node_name) in enumerate(node_metric_combinations):
            metric_values = cluster_data[metric_name]
            
            # Only process values for this specific node
            for item in metric_values:
                if item['node_name'] == node_name:
                    timestamp = item['timestamp']
                    value = item['value']
                    time_idx = timestamp_to_idx[timestamp]
                    
                    series_data[variate_idx, time_idx] = value
                    padding_mask[variate_idx, time_idx] = False
        
        # Build timestamp tensors
        timestamp_tensor = torch.tensor(sorted_timestamps, dtype=torch.int64)
        timestamp_seconds = timestamp_tensor.unsqueeze(0).expand(n_variates, n_timesteps)
        
        # Time interval tensor (same for all variates)
        time_interval_seconds = torch.full((n_variates,), time_interval, dtype=torch.int64)
        
        return MaskedTimeseries(
            series=series_data,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )
