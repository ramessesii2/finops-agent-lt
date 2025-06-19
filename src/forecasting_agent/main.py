import asyncio
import logging
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime, timedelta
import pandas as pd
import time
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter

from .collectors.prometheus import PrometheusCollector
from .models.prophet import ProphetModel
from .optimizers.idle_capacity import IdleCapacityOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
# best practice: encode units in the name to avoid mixing series.
FORECAST_COST_USD = Gauge(
    'forecast_cluster_cost_usd',
    'Forecasted cluster cost (USD) for a given horizon',
    ['cluster', 'horizon'] # horizon="7d|30d|90d"
)
FORECAST_COST_QUANTILE = Gauge(
    'forecast_cluster_cost_usd_quantile',
    'Forecast cost quantiles (USD)',
    ['cluster', 'horizon', 'quantile']  # quantile="0.10|0.50|0.90"
)
FORECAST_POINT_TS = Gauge(
    'forecast_timestamp_seconds',
    'Unix timestamp the forecast value refers to',
    ['cluster', 'horizon']
)
FORECAST_MAPE = Gauge(
    'forecast_mape',
    'Mean-absolute-percentage-error for the previous horizon',
    ['cluster', 'horizon']
)
OPTIMIZATION_SAVINGS = Gauge(
    'optimization_potential_savings',
    'Potential savings from optimizations',
    ['cluster', 'type']
)
COLLECTION_ERRORS = Counter(
    'collection_errors_total',
    'Total number of metric collection errors',
    ['collector']
)

class ForecastingAgent:
    """Main forecasting agent class."""
    
    def __init__(self, config_path: str):
        """Initialize the forecasting agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.collector = self._init_collector()
        self.optimizer = self._init_optimizer()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _init_collector(self) -> PrometheusCollector:
        """Initialize metrics collector.
        
        Returns:
            Initialized collector instance
        """
        return PrometheusCollector(self.config['collector'])
        
    def _init_optimizer(self) -> IdleCapacityOptimizer:
        """Initialize optimizer.
        
        Returns:
            Initialized optimizer instance
        """
        return IdleCapacityOptimizer(self.config['optimizer'])
        
    def collect_metrics(self) -> pd.DataFrame:
        """Collect metrics from configured sources.
        
        Returns:
            DataFrame with collected metrics
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config['collector']['lookback_days'])
            
            metrics = self.collector.get_required_metrics()
            return self.collector.collect_metrics(start_time, end_time, metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            COLLECTION_ERRORS.labels(collector=self.collector.__class__.__name__).inc()
            raise
            
    def generate_forecast(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """Generate forecasts using the configured model.
        
        Args:
            metrics: DataFrame with historical metrics
            
        Returns:
            Dictionary with forecast results
        """
        try:
            forecasts = []
            horizon = self.config['models']['forecast_horizon']
            horizon_label = f"{horizon}d" if isinstance(horizon, int) else horizon
            # Group by metric_name
            grouped = metrics.groupby('metric_name')
            
            for metric_name, group in grouped:
                logger.info(f"Processing metric: {metric_name}")
                
                # Use 'value' column for all metrics (cost is the same as value)
                target_col = 'value'
                
                
                # Prepare data for model fitting
                df = pd.DataFrame({
                    'ds': group.index.tz_localize(None),  # Remove timezone info to avoid issues
                    'y': group[target_col].values
                })
                
                logger.info(f"Training Prophet model for {metric_name} with {len(df)} data points")
                
                # Create fresh model and fit
                model = ProphetModel(self.config['models']['prophet'])
                model.fit(df)
                
                # Generate forecast
                forecast = model.forecast(horizon=horizon, frequency='1D')
                
                logger.info(f"Generated forecast for {metric_name} with {len(forecast['forecast'])} future points")
                
                # Use Prophet's built-in plotting
                try:
                    # Create Prophet plot using original forecast format
                    plot_filename = f'forecast_{metric_name.replace("/", "_").replace(":", "_")}.png'
                    model.plot_forecast(forecast['prophet_forecast'], save_path=plot_filename)
                    
                    # Create components plot if there's enough data
                    if len(df) > 7:  # Need at least a week of data for seasonality
                        components_filename = f'components_{metric_name.replace("/", "_").replace(":", "_")}.png'
                        model.plot_components(forecast['prophet_forecast'], save_path=components_filename)
                        
                except Exception as plot_error:
                    logger.warning(f"Could not create Prophet plot for {metric_name}: {str(plot_error)}")
                
                # Update Prometheus metrics
                clusters = group['labels'].apply(lambda x: x.get('clusterName', 'unknown')).unique()
                for cluster in clusters:
                    # Get the last forecast values
                    # TODO: This is Prophet's output Structure. Generalize it for any Kats models
                    last_forecast = forecast['forecast'].iloc[-1]
                    last_lower = forecast['lower_bound'].iloc[-1]
                    last_upper = forecast['upper_bound'].iloc[-1]
                    
                    # Update cost metrics for cost-related metric names
                    if 'cost' in metric_name.lower():
                        FORECAST_COST.labels(cluster=cluster).set(last_forecast['value'])
                    FORECAST_CONFIDENCE.labels(cluster=cluster, bound='upper').set(last_upper['value'])
                    FORECAST_CONFIDENCE.labels(cluster=cluster, bound='lower').set(last_lower['value'])
                
                forecasts.append({
                    "metric_name": metric_name, 
                    "forecast": forecast,
                    "model": model
                })
            
            logger.info(f"Generated forecasts for {len(forecasts)} metrics")
            logger.info("Individual Prophet plots saved as PNG files")
            
            return {"forecasts": forecasts}
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
            
    def generate_recommendations(self, 
                               metrics: pd.DataFrame,
                               forecast: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate optimization recommendations.
        
        Args:
            metrics: DataFrame with historical metrics
            forecast: Optional forecast results
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = self.optimizer.analyze(metrics, forecast)
            
            # Update Prometheus metrics
            for rec in recommendations:
                OPTIMIZATION_SAVINGS.labels(
                    cluster=rec['node'],
                    type=rec['type']
                ).set(rec['recommended_value']['potential_savings'])
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
            
    def run(self):
        """Run the forecasting agent."""
        # Start Prometheus metrics server
        start_http_server(
            self.config['metrics']['port'],
            addr=self.config['metrics'].get('host', '0.0.0.0')
        )
        

        try:
            # Collect metrics
            metrics = self.collect_metrics()
            
            # Generate forecast
            forecast = self.generate_forecast(metrics)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(metrics, forecast)
            
            # Log results
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            # Wait for next iteration
            time.sleep(self.config['agent']['interval'])
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying
            
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forecasting Agent')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    agent = ForecastingAgent(args.config)
    #asyncio.run(agent.run())
    agent.run()
    
if __name__ == '__main__':
    main() 