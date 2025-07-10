from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class BaseOptimizer(ABC):
    """Base class for all optimization strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the optimizer with configuration.
        
        Args:
            config: Optimizer-specific configuration dictionary
        """
        self.config = config
        
    @abstractmethod
    def analyze(self, 
                metrics: pd.DataFrame,
                forecasts: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict[str, Any]]:
        """Analyze metrics and forecasts to generate optimization recommendations.
        
        Args:
            metrics: DataFrame with historical metrics
            forecasts: Optional dictionary of forecast DataFrames
            
        Returns:
            List of recommendation dictionaries, each containing:
            - type: Type of optimization (e.g., 'node_scaling', 'pod_rightsizing')
            - resource: Resource being optimized (e.g., 'cpu', 'memory')
            - current_value: Current resource value
            - recommended_value: Recommended resource value
            - potential_savings: Estimated cost savings
            - confidence: Confidence score (0-1)
            - details: Additional optimization details
        """
        pass
    
    @abstractmethod
    def validate_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Validate if a recommendation is safe to apply.
        
        Args:
            recommendation: Recommendation dictionary to validate
            
        Returns:
            True if recommendation is safe, False otherwise
        """
        pass
    
    def get_optimization_types(self) -> List[str]:
        """Get list of optimization types supported by this optimizer.
        
        Returns:
            List of supported optimization types
        """
        return ['node_scaling', 'pod_rightsizing']  # Default types
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics required for optimization.
        
        Returns:
            List of required metric names
        """
        return ['cpu_usage', 'memory_usage', 'cost']  # Default metrics
    
    def get_optimization_interval(self) -> int:
        """Get recommended interval between optimization runs (in minutes).
        
        Returns:
            Recommended interval in minutes
        """
        return 60  # Default to hourly optimization
    
    def get_optimization_thresholds(self) -> Dict[str, float]:
        """Get optimization thresholds.
        
        Returns:
            Dictionary of threshold values for different metrics
        """
        return {
            'min_utilization': 0.5,  # 50% minimum utilization
            'min_savings': 100.0,    # $100 minimum daily savings
            'min_confidence': 0.8     # 80% minimum confidence
        } 