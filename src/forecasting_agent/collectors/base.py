from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta

class BaseCollector(ABC):
    """Base class for all metric collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the collector with configuration.
        
        Args:
            config: Collector-specific configuration dictionary
        """
        self.config = config
        self.last_collection: Optional[datetime] = None
        
    @abstractmethod
    def collect_metrics(self,
                       start_time: datetime,
                       end_time: datetime,
                       metrics: List[str]) -> pd.DataFrame:
        """Collect metrics for the specified time range.
        
        Args:
            start_time: Start of the collection period
            end_time: End of the collection period
            metrics: List of metric names to collect
            
        Returns:
            DataFrame with columns:
            - timestamp: Datetime index
            - metric_name: Name of the metric
            - value: Metric value
            - labels: Dictionary of metric labels
        """
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics.
        
        Returns:
            List of metric names that can be collected
        """
        pass
    
    def validate_metrics(self, metrics: List[str]) -> List[str]:
        """Validate that requested metrics are available.
        
        Args:
            metrics: List of metric names to validate
            
        Returns:
            List of valid metric names
            
        Raises:
            ValueError: If any requested metric is not available
        """
        available = self.get_available_metrics()
        invalid = [m for m in metrics if m not in available]
        if invalid:
            raise ValueError(f"Metrics not available: {invalid}")
        return metrics
    
    def get_collection_interval(self) -> timedelta:
        """Get the recommended collection interval.
        
        Returns:
            Recommended time between collections
        """
        return timedelta(minutes=5)  # Default to 5 minutes
    
    def health_check(self) -> Dict[str, Any]:
        """Check collector health.
        
        Returns:
            Dictionary with health status and details
        """
        return {
            "status": "healthy",
            "last_collection": self.last_collection,
            "available_metrics": len(self.get_available_metrics())
        } 