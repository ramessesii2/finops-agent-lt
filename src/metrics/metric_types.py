"""Metric type configuration and classification."""
from typing import Set, Dict, Any
from enum import Enum


class MetricAggregationLevel(Enum):
    """Metric aggregation levels."""
    CLUSTER = "cluster"
    NODE = "node"
    POD = "pod"
    CONTAINER = "container"


class MetricTypeClassifier:
    """Classifier for metric types and aggregation levels."""
    
    # Cluster-level metrics: Single entry per cluster with node="cluster-aggregate"
    CLUSTER_LEVEL_METRICS: Set[str] = {
        # Core resource metrics (aggregated across cluster)
        'cost_usd_per_cluster',
        'cpu_pct_per_cluster', 
        'mem_pct_per_cluster',
        
        # Capacity planning metrics (cluster totals)
        'node_count_per_cluster',
        'mem_total_gb_per_cluster',
        
        # Network I/O metrics (cluster aggregates)
        # 'network_receive_bytes',
        # 'network_transmit_bytes',
        # 'disk_read_bytes',
        # 'disk_write_bytes',
        
        # Storage capacity metrics (cluster totals)
        # 'disk_total_gb',
        # 'disk_used_gb',
    }
    
    # Node-level metrics: Multiple entries per node with actual node names
    NODE_LEVEL_METRICS: Set[str] = {
        # Per-node resource metrics
        'cost_usd_per_node',
        'cpu_pct_per_node',
        'mem_pct_per_node',
        'cpu_total_cores_per_node',
        # 'disk_usage_per_node',
        # 'network_io_per_node',
    }
    
    # Pod-level metrics: Multiple entries per pod (future expansion)
    POD_LEVEL_METRICS: Set[str] = {
        # Future pod-level metrics
        # 'pod_cpu_usage',
        # 'pod_memory_usage',
        # 'pod_restart_count',
    }
    
    # Container-level metrics: Multiple entries per container (future expansion)
    CONTAINER_LEVEL_METRICS: Set[str] = {
        # Future container-level metrics
        # 'container_cpu_usage',
        # 'container_memory_usage',
    }
    
    @classmethod
    def get_aggregation_level(cls, metric_name: str) -> MetricAggregationLevel:
        """
        Determine the aggregation level for a given metric.
        
        Args:
            metric_name: Name of the metric to classify
            
        Returns:
            MetricAggregationLevel: The appropriate aggregation level
            
        Raises:
            ValueError: If metric name is not recognized
        """
        if metric_name in cls.CLUSTER_LEVEL_METRICS:
            return MetricAggregationLevel.CLUSTER
        elif metric_name in cls.NODE_LEVEL_METRICS:
            return MetricAggregationLevel.NODE
        elif metric_name in cls.POD_LEVEL_METRICS:
            return MetricAggregationLevel.POD
        elif metric_name in cls.CONTAINER_LEVEL_METRICS:
            return MetricAggregationLevel.CONTAINER
        else:
            # Default to cluster level for unknown metrics
            return MetricAggregationLevel.CLUSTER
    
    @classmethod
    def is_cluster_level(cls, metric_name: str) -> bool:
        """Check if a metric is cluster-level (single entry per cluster)."""
        return metric_name in cls.CLUSTER_LEVEL_METRICS
    
    @classmethod
    def is_node_level(cls, metric_name: str) -> bool:
        """Check if a metric is node-level (multiple entries per node)."""
        return metric_name in cls.NODE_LEVEL_METRICS
    
    @classmethod
    def get_node_name_for_metric(cls, metric_name: str, actual_node_name: str = None) -> str:
        """
        Get the appropriate node name for a metric based on its aggregation level.
        
        Args:
            metric_name: Name of the metric
            actual_node_name: The actual node name from Prometheus data
            
        Returns:
            str: "cluster-aggregate" for cluster-level metrics, actual node name for node-level metrics
        """
        if cls.is_cluster_level(metric_name):
            return "cluster-aggregate"
        elif cls.is_node_level(metric_name):
            return actual_node_name or "unknown-node"
        else:
            # Default behavior for unknown metrics
            return "cluster-aggregate"
    
    @classmethod
    def get_metrics_by_level(cls, metric_names: Set[str]) -> Dict[MetricAggregationLevel, Set[str]]:
        """
        Group metrics by their aggregation level.
        
        Args:
            metric_names: Set of metric names to classify
            
        Returns:
            Dict mapping aggregation levels to sets of metric names
        """
        result = {
            MetricAggregationLevel.CLUSTER: set(),
            MetricAggregationLevel.NODE: set(),
            MetricAggregationLevel.POD: set(),
            MetricAggregationLevel.CONTAINER: set(),
        }
        
        for metric_name in metric_names:
            level = cls.get_aggregation_level(metric_name)
            result[level].add(metric_name)
        
        return result
    
    @classmethod
    def get_all_metrics(cls) -> Set[str]:
        """Get all known metrics across all aggregation levels."""
        return (
            cls.CLUSTER_LEVEL_METRICS |
            cls.NODE_LEVEL_METRICS |
            cls.POD_LEVEL_METRICS |
            cls.CONTAINER_LEVEL_METRICS
        )

def get_cluster_level_metrics() -> Set[str]:
    """Get set of cluster-level metrics."""
    return MetricTypeClassifier.CLUSTER_LEVEL_METRICS.copy()


def get_node_level_metrics() -> Set[str]:
    """Get set of node-level metrics."""
    return MetricTypeClassifier.NODE_LEVEL_METRICS.copy()


def is_cluster_level_metric(metric_name: str) -> bool:
    """Check if metric is cluster-level."""
    return MetricTypeClassifier.is_cluster_level(metric_name)


def is_node_level_metric(metric_name: str) -> bool:
    """Check if metric is node-level."""
    return MetricTypeClassifier.is_node_level(metric_name)


# Export commonly used items
__all__ = [
    'MetricAggregationLevel',
    'MetricTypeClassifier',
    'get_cluster_level_metrics',
    'get_node_level_metrics', 
    'is_cluster_level_metric',
    'is_node_level_metric',
]
