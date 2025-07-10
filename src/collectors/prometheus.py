from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from prometheus_api_client import PrometheusConnect
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PrometheusCollector:
    """Prometheus metrics collector implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Prometheus collector with configuration.
        
        Args:
            config: Prometheus-specific configuration dictionary
        """
        self.config = config  # Store config for later use
        self.prom = PrometheusConnect(
            url=config['url'],
            headers=config.get('headers', {}),
            disable_ssl=config.get('disable_ssl', False)
        )
        # Timestamp of the most recent successful collection (for health checks)
        self.last_collection: Optional[datetime] = None
        logger.debug(f"Initialized PrometheusCollector with URL: {config['url']}")
        

    def _prom_query(self, promql: str, start_time: datetime, end_time: datetime):
        """Run a range query and return the raw Prometheus response list."""
        logger.debug("PromQL query: %s", promql)
        start = start_time.astimezone(timezone.utc).replace(tzinfo=None)
        end   = end_time.astimezone(timezone.utc).replace(tzinfo=None)
        return self.prom.custom_query_range(
            query=promql,
            start_time=start,
            end_time=end,
            step=self.config.get("step", "1m"),
        )

    def collect_metrics_timeseries(self, start_time: datetime, end_time: datetime, promq: Dict[str, str]) -> Dict[str, list]:
        """
        Collect multiple Prometheus metrics and return the Prometheus query
        results so that downstream components can decide how to post-process them.

        Args:
            start_time: Start of the query range (datetime)
            end_time: End of the query range (datetime)
            promq: Mapping from a friendly metric key to the PromQL query string

        Returns:
            Dict[str, list]: A dictionary whose keys are the provided ``metric_key``
            values and whose values are the raw result lists returned by
            ``PrometheusConnect.custom_query_range``.
        """
        results: Dict[str, list] = {}
        for metric_key, promql in promq.items():
            try:
                raw_result = self._prom_query(promql, start_time, end_time)
                results[metric_key] = raw_result
            except Exception as e:
                logger.error(f"Error collecting metric '{metric_key}': {e}")
                continue

        # If every metric returned an empty result set treat it as a failure
        if not results or all(len(v) == 0 for v in results.values()):
            raise ValueError("No metrics data collected from Prometheus.")

        # Record timestamp of successful collection for health_check()
        self.last_collection = datetime.now()
        return results
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available Prometheus metrics.
        
        Returns:
            List of available metric names
        """
        metrics = []
        try:
            # Query Prometheus for all available metrics
            metrics = self.prom.all_metrics()
        except Exception as e:
            print(f"Error getting available metrics: {str(e)}")
                
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Check Prometheus collector health.
        
        Returns:
            Dictionary with health status and details
        """
        try:
            # Use a lightweight query to check connectivity
            self.prom.check_prometheus_connection()
            status = "healthy"
        except Exception as e:
            status = f"unhealthy: {str(e)}"
            
        return {
            "status": status,
            "last_collection": self.last_collection,
            "prometheus_url": self.config['url']
        } 