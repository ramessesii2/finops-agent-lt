"""
PromQL Query Configuration for Forecasting Agent

Simple, centralized PromQL query definitions.
Following coding-conventions.md: "Keep it simple!"
"""
from typing import Dict


# === CLUSTER-LEVEL QUERIES ===
CLUSTER_QUERIES = {
    "cost_usd_per_cluster": "sum(node_total_hourly_cost) by (clusterName)",
    "cpu_pct_per_cluster": "100 * (1 - avg by (clusterName) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))",
    "mem_pct_per_cluster": "100 * (1 - avg by (clusterName) (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))",
    "node_count_per_cluster": "count by (clusterName) (kube_node_status_condition{condition=\"Ready\",status=\"true\"})",
    "mem_total_gb_per_cluster": "sum by (clusterName) (node_memory_MemTotal_bytes) / 1024 / 1024 / 1024",
}

# === NODE-LEVEL QUERIES ===
NODE_QUERIES = {
    "cost_usd_per_node": "sum by (clusterName, node) (node_total_hourly_cost)",
    "cpu_pct_per_node": "100 * (1 - avg by (clusterName, nodename) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))",
    "mem_pct_per_node": "100 * (1 - avg by (clusterName, nodename) (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))",
    "cpu_total_cores_per_node": "sum by (clusterName, instance) (machine_cpu_cores)",
}


def get_all_queries() -> Dict[str, str]:
    """Get all PromQL queries."""
    return {**CLUSTER_QUERIES, **NODE_QUERIES}


def get_query(metric_name: str) -> str:
    """Get PromQL query for a metric."""
    all_queries = get_all_queries()
    if metric_name not in all_queries:
        raise KeyError(f"PromQL query not found for metric: {metric_name}")
    return all_queries[metric_name]
