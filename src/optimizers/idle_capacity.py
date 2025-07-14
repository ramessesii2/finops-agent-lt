from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from optimizers.base import BaseOptimizer
import logging
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Prometheus metric for node idle capacity recommendations
NODE_IDLE_CAPACITY_SAVINGS = Gauge(
    'node_idle_capacity_potential_savings',
    'Potential daily savings from node idle capacity optimization',
    ['node', 'action', 'type']
)

class IdleCapacityOptimizer(BaseOptimizer):
    """Identifies and recommends idle capacity reductions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize idle capacity optimizer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.idle_threshold = config.get('idle_threshold', 0.5)  # 50% idle threshold
        self.min_savings = config.get('min_savings', 100.0)  # Minimum daily savings
        self.lookback_days = config.get('lookback_days', 7)  # Days to analyze
        
    def analyze(self, 
                metrics: pd.DataFrame,
                forecasts: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict[str, Any]]:
        """Analyze node metrics to identify idle capacity.
        
        Args:
            metrics: DataFrame with historical metrics
            forecasts: Optional dictionary of forecast DataFrames
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if metrics.empty:
            logger.warning("No metrics data provided for idle capacity analysis")
            return recommendations
        
        logger.debug(f"Starting idle capacity analysis with {len(metrics)} metrics records")
        
        # Step 1: Extract and process CPU metrics
        cpu_metrics = self._extract_cpu_metrics(metrics)
        
        # Step 2: Extract and process memory metrics  
        memory_metrics = self._extract_memory_metrics(metrics)
        
        # Step 3: Extract cost metrics
        cost_metrics = self._extract_cost_metrics(metrics)
        
        # Step 4: Group by node and calculate utilization
        node_utilizations = self._calculate_node_utilizations(cpu_metrics, memory_metrics, cost_metrics)
        logger.info(f"Calculated utilizations for {len(node_utilizations)} nodes")
        
        # Debug: Print node utilizations
        for node, util_data in node_utilizations.items():
            logger.debug(f"Node {node}: CPU={util_data['cpu_utilization']:.2%}, "
                       f"Memory={util_data['memory_utilization']:.2%}, "
                       f"Cost=${util_data['cost']:.2f}/hour")
        
        # Step 5: Generate recommendations
        for node, util_data in node_utilizations.items():
            should_recommend = self._should_recommend_optimization(util_data)
            logger.info(f"Node {node}: should_recommend={should_recommend}")
            
            if should_recommend:
                recommendation = self._create_recommendation(node, util_data)
                recommendations.append(recommendation)
                logger.info(f"Created recommendation for node {node}")
        
        # Sort recommendations by potential savings
        recommendations.sort(key=lambda x: x['recommended_value']['potential_savings'], 
                           reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _extract_cpu_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Extract and process CPU metrics from raw data."""
        cpu_data = metrics[metrics['metric_name'] == 'node_cpu_seconds_total'].copy()
        
        if cpu_data.empty:
            logger.warning("No CPU metrics found")
            return pd.DataFrame()
        
        # Extract node information from labels
        cpu_data['node'] = cpu_data['labels'].apply(
            lambda x: x.get('instance', x.get('node', 'unknown'))
        )
        
        # Extract CPU mode from labels
        cpu_data['mode'] = cpu_data['labels'].apply(
            lambda x: x.get('mode', 'unknown')
        )
        
        # Group by node and timestamp to calculate utilization
        cpu_utilizations = []
        
        for node in cpu_data['node'].unique():
            node_data = cpu_data[cpu_data['node'] == node]
            
            # Group by timestamp to get all modes for each time point
            for timestamp in node_data.index.unique():
                time_data = node_data.loc[timestamp]
                
                # Handle single row vs multiple rows
                if isinstance(time_data, pd.Series):
                    time_data = pd.DataFrame([time_data])
                
                # Calculate total CPU time and idle time
                total_cpu = time_data['value'].sum()
                idle_cpu = time_data[time_data['mode'] == 'idle']['value'].sum()
                
                # Calculate utilization: (total - idle) / total
                if total_cpu > 0:
                    utilization = (total_cpu - idle_cpu) / total_cpu
                    # Normalize to 0-1 range and handle edge cases
                    utilization = max(0.0, min(1.0, utilization))
                else:
                    utilization = 0.0
                
                cpu_utilizations.append({
                    'node': node,
                    'cpu_utilization': utilization,
                    'timestamp': timestamp
                })
        
        return pd.DataFrame(cpu_utilizations)
    
    def _extract_memory_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Extract and process memory metrics from raw data."""
        # Try different memory metric combinations
        memory_data = []
        
        # Option 1: Total and Available memory
        total_memory = metrics[metrics['metric_name'] == 'node_memory_MemTotal_bytes'].copy()
        available_memory = metrics[metrics['metric_name'] == 'node_memory_MemAvailable_bytes'].copy()
        
        # Option 2: Total and Used memory (alternative approach)
        used_memory = metrics[metrics['metric_name'] == 'node_memory_MemUsed_bytes'].copy()
        
        # Option 3: Working set memory (container-based)
        working_set = metrics[metrics['metric_name'] == 'container_memory_working_set_bytes'].copy()
        
        if not total_memory.empty:
            # Extract node information
            total_memory['node'] = total_memory['labels'].apply(
                lambda x: x.get('instance', x.get('node', 'unknown'))
            )
            available_memory['node'] = available_memory['labels'].apply(
                lambda x: x.get('instance', x.get('node', 'unknown'))
            )
            
            # Calculate memory utilization for each node
            for node in total_memory['node'].unique():
                node_total = total_memory[total_memory['node'] == node]
                node_available = available_memory[available_memory['node'] == node]
                
                if not node_total.empty and not node_available.empty:
                    # Calculate utilization: (total - available) / total
                    total_val = node_total['value'].iloc[0]
                    available_val = node_available['value'].iloc[0]
                    
                    if total_val > 0:
                        utilization = (total_val - available_val) / total_val
                        utilization = max(0.0, min(1.0, utilization))  # Normalize to 0-1
                        
                        memory_data.append({
                            'node': node,
                            'memory_utilization': utilization,
                            'timestamp': node_total.index[0]
                        })
        
        # If no memory data found, try alternative approaches
        if not memory_data and not used_memory.empty:
            logger.info("Using alternative memory calculation with used memory")
            used_memory['node'] = used_memory['labels'].apply(
                lambda x: x.get('instance', x.get('node', 'unknown'))
            )
            
            for node in used_memory['node'].unique():
                node_used = used_memory[used_memory['node'] == node]
                if not node_used.empty:
                    # Assume 100% utilization if we only have used memory
                    # This is a conservative estimate
                    memory_data.append({
                        'node': node,
                        'memory_utilization': 0.8,  # Conservative estimate
                        'timestamp': node_used.index[0]
                    })
        
        if not memory_data:
            logger.warning("No memory metrics found, using default utilization")
            # If no memory metrics at all, use a default value
            # This ensures the optimizer can still work
            all_nodes = set()
            if not total_memory.empty:
                all_nodes.update(total_memory['node'].unique())
            if not used_memory.empty:
                all_nodes.update(used_memory['node'].unique())
            
            for node in all_nodes:
                memory_data.append({
                    'node': node,
                    'memory_utilization': 0.5,  # Default 50% utilization
                    'timestamp': pd.Timestamp.now()
                })
        
        return pd.DataFrame(memory_data)
    
    def _extract_cost_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Extract cost metrics from raw data."""
        cost_data = metrics[metrics['metric_name'] == 'node_total_hourly_cost'].copy()
        
        if cost_data.empty:
            logger.warning("No cost metrics found, using default cost")
            return pd.DataFrame()
        
        # Extract node information
        cost_data['node'] = cost_data['labels'].apply(
            lambda x: x.get('instance', x.get('node', 'unknown'))
        )
        
        # Create result DataFrame with timestamp from index
        result_data = []
        for idx, row in cost_data.iterrows():
            result_data.append({
                'node': row['node'],
                'cost': row['value'], 
                'timestamp': idx
            })
        
        return pd.DataFrame(result_data)
    
    def _calculate_node_utilizations(self, 
                                   cpu_metrics: pd.DataFrame, 
                                   memory_metrics: pd.DataFrame,
                                   cost_metrics: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate utilization statistics for each node."""
        node_utilizations = {}
        
        # Get all unique nodes
        all_nodes = set()
        if not cpu_metrics.empty:
            all_nodes.update(cpu_metrics['node'].unique())
        if not memory_metrics.empty:
            all_nodes.update(memory_metrics['node'].unique())
        if not cost_metrics.empty:
            all_nodes.update(cost_metrics['node'].unique())
        
        for node in all_nodes:
            node_data = {
                'cpu_utilization': 0.0,
                'memory_utilization': 0.0,
                'cost': 0.0,
                'cpu_std': 0.0,
                'memory_std': 0.0,
                'data_points': 0
            }
            
            # Calculate CPU utilization
            if not cpu_metrics.empty:
                node_cpu = cpu_metrics[cpu_metrics['node'] == node]
                if not node_cpu.empty:
                    node_data['cpu_utilization'] = node_cpu['cpu_utilization'].mean()
                    node_data['cpu_std'] = node_cpu['cpu_utilization'].std()
                    node_data['data_points'] = len(node_cpu)
            
            # Calculate memory utilization
            if not memory_metrics.empty:
                node_memory = memory_metrics[memory_metrics['node'] == node]
                if not node_memory.empty:
                    node_data['memory_utilization'] = node_memory['memory_utilization'].mean()
                    node_data['memory_std'] = node_memory['memory_utilization'].std()
            
            # Get cost data
            if not cost_metrics.empty:
                node_cost = cost_metrics[cost_metrics['node'] == node]
                if not node_cost.empty:
                    node_data['cost'] = node_cost['cost'].mean()
            
            node_utilizations[node] = node_data
        
        return node_utilizations
    
    def _should_recommend_optimization(self, util_data: Dict[str, Any]) -> bool:
        """Determine if optimization should be recommended for a node."""
        cpu_util = util_data['cpu_utilization']
        mem_util = util_data['memory_utilization']
        cost = util_data['cost']
        
        # Check if utilization is below threshold
        is_idle = (cpu_util < self.idle_threshold or mem_util < self.idle_threshold)
        
        # Calculate potential savings
        max_util = max(cpu_util, mem_util)
        potential_savings = cost * (1 - max_util) * 24  # Daily savings
        
        # Check if savings meet minimum threshold
        has_significant_savings = potential_savings >= self.min_savings
        
        logger.debug(f"Node optimization check: CPU={cpu_util:.2%}, Memory={mem_util:.2%}, "
                    f"Cost=${cost:.2f}/hour, MaxUtil={max_util:.2%}, "
                    f"PotentialSavings=${potential_savings:.2f}/day, "
                    f"IdleThreshold={self.idle_threshold}, MinSavings=${self.min_savings}")
        logger.debug(f"  is_idle={is_idle}, has_significant_savings={has_significant_savings}")
        
        return is_idle and has_significant_savings
    
    def _create_recommendation(self, node: str, util_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an optimization recommendation for a node."""
        cpu_util = util_data['cpu_utilization']
        mem_util = util_data['memory_utilization']
        cost = util_data['cost']
        cpu_std = util_data['cpu_std']
        mem_std = util_data['memory_std']
        
        # Calculate potential savings
        max_util = max(cpu_util, mem_util)
        potential_savings = cost * (1 - max_util) * 24  # Daily savings
                
        # Calculate confidence based on utilization stability
        confidence = 1 - min(1, (cpu_std + mem_std) / 2)
                
        # Determine action based on utilization level
        if max_util < 0.3:
            action = 'remove'
        else:
            action = 'downsize'
        
        return {
                    'type': 'node_scaling',
                    'resource': 'node',
                    'node': node,
                    'current_value': {
                        'cpu_utilization': cpu_util,
                        'memory_utilization': mem_util,
                'cost': cost
                    },
                    'recommended_value': {
                'action': action,
                        'potential_savings': potential_savings
                    },
                    'confidence': confidence,
                    'details': {
                        'cpu_std': cpu_std,
                        'mem_std': mem_std,
                'lookback_days': self.lookback_days,
                'data_points': util_data['data_points']
            }
        }
    
    def validate_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Validate if a recommendation is safe to apply.
        
        Args:
            recommendation: Recommendation dictionary to validate
            
        Returns:
            True if recommendation is safe, False otherwise
        """
        # Check confidence threshold
        if recommendation['confidence'] < self.config.get('min_confidence', 0.8):
            return False
            
        # Check minimum savings threshold
        if recommendation['recommended_value']['potential_savings'] < self.min_savings:
            return False
            
        # Check utilization thresholds
        current = recommendation['current_value']
        if current['cpu_utilization'] > 0.7 or current['memory_utilization'] > 0.7:
            return False
            
        return True
    
    def get_optimization_types(self) -> List[str]:
        """Get list of optimization types supported by this optimizer.
        
        Returns:
            List of supported optimization types
        """
        return ['node_scaling']
    
    def get_required_metrics(self) -> List[str]:
        """
        Return Kubernetes/Prometheus metric names needed for idle‑capacity analysis.
        """
        return [
            # CPU usage per node (rate over mode!="idle")
            "node_cpu_seconds_total",
            # Memory availability / total bytes per node
            "node_memory_MemAvailable_bytes",
            "node_memory_MemTotal_bytes",
            # Cost per node from OpenCost (optional, falls back to 0 if missing)
            "node_total_hourly_cost"
        ]
    
    def get_optimization_interval(self) -> int:
        """Get recommended interval between optimization runs (in minutes).
        
        Returns:
            Recommended interval in minutes
        """
        return 60  # Check hourly
    
    def get_optimization_thresholds(self) -> Dict[str, float]:
        """Get optimization thresholds.
        
        Returns:
            Dictionary of threshold values for different metrics
        """
        return {
            'min_utilization': self.idle_threshold,
            'min_savings': self.min_savings,
            'min_confidence': self.config.get('min_confidence', 0.8)
        }
    
    def expose_recommendations_metrics(self, recommendations: List[Dict[str, Any]]):
        """Expose node idle capacity recommendations as Prometheus metrics."""
        for rec in recommendations:
            node = rec['node']
            action = rec['recommended_value']['action']
            rec_type = rec['type']
            savings = rec['recommended_value']['potential_savings']
            NODE_IDLE_CAPACITY_SAVINGS.labels(node=node, action=action, type=rec_type).set(savings) 