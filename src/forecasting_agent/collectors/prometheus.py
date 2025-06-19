from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from prometheus_api_client.utils import parse_datetime
import requests
from prometheus_api_client import PrometheusConnect
from .base import BaseCollector
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PrometheusCollector(BaseCollector):
    """Prometheus metrics collector implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Prometheus collector with configuration.
        
        Args:
            config: Prometheus-specific configuration dictionary
        """
        super().__init__(config)
        self.prom = PrometheusConnect(
            url=config['url'],
            headers=config.get('headers', {}),
            disable_ssl=config.get('disable_ssl', False)
        )
        self._available_metrics = None
        logger.debug(f"Initialized PrometheusCollector with URL: {config['url']}")
        
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics required for collection.
        
        Returns:
            List of metric names to collect
        """
        # Default metrics for cost and resource utilization
        return [
            # OpenCost metrics
            'kubecost_cluster_management_cost',  # Hourly cost paid as a cluster management fee
            'kubecost_load_balancer_cost',  # Hourly cost of load balancer
            'kubecost_network_internet_egress_cost', # Total cost per GB of internet egress.
            'node_total_hourly_cost', # Total node cost per hour
            # Resource utilization metrics
            'node_cpu_seconds_total',  # CPU utilization
            'node_memory_MemTotal_bytes',  # Memory utilization
            'node_memory_MemAvailable_bytes',
            # 'container_cpu_usage_seconds_total',  # Container CPU usage
            # 'container_memory_working_set_bytes'  # Container memory usage
        ]
        
    def collect_metrics(self,
                       start_time: datetime,
                       end_time: datetime,
                       metrics: List[str]) -> pd.DataFrame:
        """Collect metrics from Prometheus.
        
        Args:
            start_time: Start of the collection period
            end_time: End of the collection period
            metrics: List of metric names to collect
            
        Returns:
            DataFrame with collected metrics
        """
        # Validate metrics
        metrics = self.validate_metrics(metrics)
        # Convert timestamps to Prometheus format
        # start_ts = parse_datetime(start_time)
        # end_ts = parse_datetime(end_time)
        
        logger.debug(f"Collecting metrics from {start_time} to {end_time}")
        logger.debug(f"Metrics to collect: {metrics}")
        
        # Collect metrics
        results = []
        for metric in metrics:
            try:
                # Query Prometheus
                logger.debug(f"Querying metric: {metric}")
                result = self.prom.custom_query_range(
                    query=metric,
                    start_time=start_time,
                    end_time=end_time,
                    step=self.config.get('step', '1m')
                )
                
                if not result:
                    logger.warning(f"No data returned for metric {metric} in the given time range.")
                    continue
                    
                # logger.debug(f"Raw result for {metric}: {result}")
                
                # Process results
                for series in result:
                    metric_name = series['metric'].get('__name__', metric)
                    # Extract all labels except __name__
                    labels = {k: v for k, v in series['metric'].items() if k != '__name__'}
                    
                    # Get values from the series
                    values = series.get('values', [])
                    if not values and 'value' in series:
                        values = [series['value']]
                    
                    for value in values:
                        try:
                            if not isinstance(value, (list, tuple)) or len(value) != 2:
                                logger.warning(f"Skipping malformed value for metric {metric_name}: {value}")
                                continue
                            
                            # Convert timestamp to datetime
                            ts = float(value[0])
                            timestamp = datetime.fromtimestamp(ts)
                            
                            # Convert metric value to float
                            metric_value = float(value[1])
                            
                            results.append({
                                'timestamp': timestamp,
                                'metric_name': metric_name,
                                'value': metric_value,
                                'labels': labels
                            })
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error processing value for metric {metric_name}: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"Failed to query Prometheus for metric {metric}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            logger.debug(f"Created DataFrame with {len(df)} rows")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame head:\n{df.head()}")
        else:
            logger.warning("DataFrame is empty after collecting metrics.")
            
        return df
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available Prometheus metrics.
        
        Returns:
            List of available metric names
        """
        if self._available_metrics is None:
            try:
                # Query Prometheus for all available metrics
                result = self.prom.all_metrics()
                self._available_metrics = result
            except Exception as e:
                print(f"Error getting available metrics: {str(e)}")
                self._available_metrics = []
                
        return self._available_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Check Prometheus collector health.
        
        Returns:
            Dictionary with health status and details
        """
        try:
            # Try to query Prometheus
            self.prom.health()
            status = "healthy"
        except Exception as e:
            status = f"unhealthy: {str(e)}"
            
        return {
            "status": status,
            "last_collection": self.last_collection,
            "available_metrics": len(self.get_available_metrics()),
            "prometheus_url": self.config['url']
        } 