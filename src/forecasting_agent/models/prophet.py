from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from .base import BaseModel

class ProphetModel(BaseModel):
    """Prophet-based forecasting model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Prophet model with configuration.
        
        Args:
            config: Prophet-specific configuration dictionary
        """
        super().__init__(config)
        self.model = Prophet(
            changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=config.get('holidays_prior_scale', 10.0),
            seasonality_mode=config.get('seasonality_mode', 'multiplicative'),
            daily_seasonality=config.get('daily_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', True),
            yearly_seasonality=config.get('yearly_seasonality', True),
        )
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit Prophet model to historical data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'value']
        """
        # Prepare data for Prophet
        df = data.copy()
        df.columns = ['ds', 'y']
        
        # Add any additional regressors if specified
        if 'regressors' in self.config:
            for regressor in self.config['regressors']:
                if regressor in data.columns:
                    self.model.add_regressor(regressor)
        
        # Fit the model
        self.model.fit(df)
        
    def forecast(self, 
                horizon: int, 
                frequency: str = '1D',
                return_components: bool = False) -> Dict[str, Any]:
        """Generate forecasts using Prophet.
        
        Args:
            horizon: Number of periods to forecast
            frequency: Frequency of the forecast (e.g., '1D' for daily)
            return_components: Whether to return forecast components
            
        Returns:
            Dictionary containing forecast results
        """
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq=frequency,
            include_history=True
        )
        
        # Add any additional regressors if specified
        if 'regressors' in self.config:
            for regressor in self.config['regressors']:
                if regressor in self.config.get('regressor_forecasts', {}):
                    future[regressor] = self.config['regressor_forecasts'][regressor]
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Prepare results with both original Prophet format and renamed format
        results = {
            # Original Prophet format for plotting
            'prophet_forecast': forecast,
            # Renamed format for compatibility
            'forecast': forecast[['ds', 'yhat']].rename(columns={'ds': 'timestamp', 'yhat': 'value'}),
            'lower_bound': forecast[['ds', 'yhat_lower']].rename(columns={'ds': 'timestamp', 'yhat_lower': 'value'}),
            'upper_bound': forecast[['ds', 'yhat_upper']].rename(columns={'ds': 'timestamp', 'yhat_upper': 'value'})
        }
        
        # Add components if requested
        if return_components:
            components = forecast[['ds', 'trend', 'seasonal', 'holidays']]
            components = components.rename(columns={'ds': 'timestamp'})
            results['components'] = components
            
        return results
    
    def evaluate(self, 
                test_data: pd.DataFrame,
                metrics: Optional[list] = None) -> Dict[str, float]:
        """Evaluate Prophet model performance.
        
        Args:
            test_data: DataFrame with actual values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape']
            
        # Prepare test data
        test_df = test_data.copy()
        test_df.columns = ['ds', 'y']
        
        # Generate predictions for test period
        forecast = self.model.predict(test_df[['ds']])
        
        # Calculate metrics
        results = {}
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(y_true - y_pred))
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        if 'mape' in metrics:
            results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
        return results
    
    def plot_forecast(self, forecast: pd.DataFrame, save_path: Optional[str] = None) -> Any:
        """Plot the forecast using Prophet's built-in plotting.
        
        Args:
            forecast: Prophet forecast DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = self.model.plot(forecast)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_components(self, forecast: pd.DataFrame, save_path: Optional[str] = None) -> Any:
        """Plot forecast components using Prophet's built-in plotting.
        
        Args:
            forecast: Prophet forecast DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = self.model.plot_components(forecast)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save(self, path: str) -> None:
        """Save Prophet model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
            
    def load(self, path: str) -> None:
        """Load Prophet model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = Prophet.load(path) 