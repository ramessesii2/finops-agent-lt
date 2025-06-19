from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Base class for all forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with configuration.
        
        Args:
            config: Model-specific configuration dictionary
        """
        self.config = config
        self.model = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to historical data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'value']
        """
        pass
    
    @abstractmethod
    def forecast(self, 
                horizon: int, 
                frequency: str = '1D',
                return_components: bool = False) -> Dict[str, Any]:
        """Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of periods to forecast
            frequency: Frequency of the forecast (e.g., '1D' for daily)
            return_components: Whether to return forecast components
            
        Returns:
            Dictionary containing:
            - 'forecast': DataFrame with forecast values
            - 'lower_bound': DataFrame with lower confidence bounds
            - 'upper_bound': DataFrame with upper confidence bounds
            - 'components': Optional DataFrame with forecast components
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                test_data: pd.DataFrame,
                metrics: Optional[list] = None) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame with actual values
            metrics: List of metrics to compute (e.g., ['mae', 'rmse'])
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Model saving not implemented")
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Model loading not implemented") 