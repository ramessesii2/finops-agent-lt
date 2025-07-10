import logging
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime, timedelta, timezone
import time
from darts import TimeSeries
import json
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from collectors.prometheus import PrometheusCollector
from optimizers.idle_capacity import IdleCapacityOptimizer
from adapters.forecasting.nbeats_adapter import NBEATSAdapter
from adapters.forecasting.toto_adapter import TOTOAdapter
from adapters.prometheus_toto_adapter import PrometheusToTotoAdapter
from adapters.prometheus_timeseries_adapter import PrometheusTimeseriesAdapter
from adapters.forecast_format_converter import ForecastFormatConverter
from core.metric_types import MetricTypeClassifier, MetricAggregationLevel
from core.promql_queries import get_all_queries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use centralized PromQL configuration (simple and clean)
PROMQL = get_all_queries()

exported_forecasts: Dict[str, Dict[str, Any]] = {}

def _start_forecast_server(host: str, port: int):
    """Start a background HTTP server exposing forecast JSON."""

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/clusters" or self.path == "/clusters/":
                # List all available clusters
                cluster_list = list(exported_forecasts.keys())
                self._send_json({
                    "clusters": cluster_list,
                    "count": len(cluster_list)
                })
            elif self.path == "/metrics" or self.path == "/metrics/":
                
                self._send_json({
                    "metrics": exported_forecasts,
                    "clusters_count": len(exported_forecasts),
                    "total_forecast_entries": sum(len(forecasts) if isinstance(forecasts, list) else 1 
                                                 for forecasts in exported_forecasts.values())
                })
            elif self.path.startswith("/metrics/"):
                # Return metrics for specific cluster
                cluster = self.path[len("/metrics/"):]
                if cluster in exported_forecasts and cluster:
                    cluster_forecasts = exported_forecasts[cluster]
                    self._send_json({
                        "cluster": cluster,
                        "forecasts": cluster_forecasts,
                        "forecast_count": len(cluster_forecasts) if isinstance(cluster_forecasts, list) else 1
                    })
                else:
                    self._send_json({"error": f"cluster '{cluster}' not found", "available_clusters": list(exported_forecasts.keys())}, status=404)
            else:
                # Health check or default endpoint
                self._send_json({
                    "message": "Forecasting Agent API",
                    "status": "running",
                    "endpoints": {
                        "/clusters": "List all available clusters",
                        "/metrics": "Get all metrics from all clusters",
                        "/metrics/{clusterName}": "Get forecasts for specific cluster"
                    },
                    "active_clusters": len(exported_forecasts)
                })

        def log_message(self, *args, **kwargs):
            # Silence default HTTP server logging
            return

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Forecast HTTP server running on {host}:{port}")

def plot_all_forecasts(cluster_timeseries, exported_forecasts):
    """Plot all clusters' forecasts (history + forecast) for each component."""

    for cluster, ts in cluster_timeseries.items():
        forecast = exported_forecasts.get(cluster)
        if not forecast:
            continue
        # Plot the forecast (median) for each component
        # Reconstruct a Darts TimeSeries for the median forecast
        median_dict = {}
        for comp, quantile_dict in forecast.items():
            median_points = quantile_dict.get("q0.50") or quantile_dict.get("q0.5")
            if median_points:
                df = pd.DataFrame(median_points)
                df["ds"] = pd.to_datetime(df["ds"])
                # Use DatetimeIndex for times
                median_dict[comp] = TimeSeries.from_times_and_values(pd.DatetimeIndex(df["ds"]), df["y"].to_numpy(), columns=[comp])
        if median_dict:
            # Combine all median component series into one multivariate TimeSeries
            median_ts = TimeSeries.from_dataframe(
                pd.concat([s.to_dataframe() for s in median_dict.values()], axis=1)
            )
            for comp in ts.components:
                plt.figure(figsize=(12, 5))
                hist_df = ts[comp].to_dataframe().reset_index()
                fcst_df = median_ts[comp].to_dataframe().reset_index()
                time_col = hist_df.columns[0]
                plt.plot(hist_df[time_col], hist_df[comp], label="History", lw=2)
                plt.plot(fcst_df[time_col], fcst_df[comp], label="Forecast", lw=2, linestyle="--")
                forecast_start = hist_df[time_col].iloc[-1]
                axv = mdates.date2num(pd.Timestamp(forecast_start))
                if hasattr(axv, 'item'):
                    axv = float(axv.item())
                else:
                    axv = float(axv)
                plt.axvline(axv, color="k", linestyle=":", label="Forecast Start")
                plt.title(f"{comp} - Full Horizon Forecast for {cluster}")
                plt.xlabel("Time")
                plt.ylabel(comp)
                plt.legend()
                plt.tight_layout()
                plt.show()

class ForecastingAgent:
    """Main forecasting agent class."""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.collector = self._init_collector()
        self.optimizer = self._init_optimizer()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_collector(self) -> PrometheusCollector:
        return PrometheusCollector(self.config['collector'])

    def _init_optimizer(self) -> IdleCapacityOptimizer:
        return IdleCapacityOptimizer(self.config['optimizer'])

    def collect_raw_metrics(self) -> Dict[str, Any]:
        """Collect raw metrics from Prometheus."""
        try:
            end_time = datetime.now(tz=timezone.utc)
            scrape_step = self.config['collector'].get('step', '5m')
            start_time = end_time - timedelta(days=self.config['collector'].get('lookback_days', 4))
            # Get raw results from PrometheusCollector
            return self.collector.collect_metrics_timeseries(start_time, end_time, PROMQL)
        except Exception as e:
            logger.error(f"Error collecting raw metrics: {str(e)}")
            raise
    
    def convert_to_timeseries(self, raw_results: Dict[str, Any]) -> Dict[str, TimeSeries]:
        """Convert raw Prometheus results to per-cluster TimeSeries."""
        try:
            freq = self.config['models'].get('freq', "h")
            return PrometheusTimeseriesAdapter.prometheus_results_to_timeseries(raw_results, freq=freq)
        except Exception as e:
            logger.error(f"Error converting to timeseries: {str(e)}")
            raise
    
    def collect_metrics_timeseries(self) -> Dict[str, TimeSeries]:
        """Collect metrics and convert to TimeSeries (convenience method)."""
        raw_results = self.collect_raw_metrics()
        return self.convert_to_timeseries(raw_results)

    def generate_forecast_NBEATS(self, series: TimeSeries) -> dict:
        """Generate NBEATS forecast with horizon calculation and JSON formatting."""
        try:
            step = self.config['collector'].get('step', 'h')
            horizon_days = self.config['models'].get('forecast_horizon', 30)
            if step in ['h', '1h', 'hour']:
                horizon = horizon_days * 24
            elif step in ['m', '1m', 'min']:
                horizon = horizon_days * 24 * 60
            else:
                horizon = horizon_days
                
            model_config = self.config['models'].get('nbeats', {})
            adapter = NBEATSAdapter(model_config)
            adapter.fit(series)
            
            likelihood = model_config.get('likelihood', None)
            num_samples = model_config.get('num_samples', 100) if likelihood else 1
            quantiles = self.config['models'].get('quantiles', [0.1, 0.5, 0.9])
            
            # Use adapter's forecast method with quantiles for JSON formatting
            return adapter.forecast(horizon, num_samples=num_samples, quantiles=quantiles)
        except Exception as e:
            logger.error(f"Error generating NBEATS forecast: {str(e)}")
            raise

    def generate_forecast_TOTO(self, raw_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate TOTO forecast for all clusters using clean architecture.
        - PrometheusToTotoAdapter: converts raw JSON to MaskedTimeseries tensors
        - TOTOAdapter.forecast(): pure forecasting logic only
        - ForecastFormatConverter: transforms TOTO output to cluster-grouped format
        
        Args:
            raw_results: Raw Prometheus query results containing data for all clusters
            
        Returns:
            Dict with cluster-grouped format: {
                cluster_name: [
                    {
                        "metric": {
                            "__name__": "cpu_usage_forecast",
                            "clusterName": "prod-eu1",
                            "node": "aggregated",
                            "quantile": "0.50",
                            "horizon": "7d"
                        },
                        "values": [0.27, 0.29, 0.31],
                        "timestamps": [1719955200000, 1719958800000, 1719962400000]
                    },
                    ...
                ]
            }
        """
        try:
            # Step 1: Convert raw Prometheus JSON to multi-cluster tensors (clean data conversion)
            prometheus_to_toto = PrometheusToTotoAdapter()
            
            # Extract cluster names from raw results
            cluster_names = self._extract_cluster_names_from_raw_results(raw_results)
            if not cluster_names:
                raise ValueError("No clusters found in raw Prometheus results")
            
            # Convert each cluster individually using the existing single-cluster method
            multi_cluster_data = {}
            for cluster_name in cluster_names:
                try:
                    # Convert raw results to TOTO tensor format for each cluster
                    conversion_result = prometheus_to_toto.convert_to_toto_format(
                        prometheus_data=raw_results,
                        cluster_name=cluster_name
                    )
                    
                    # Extract metric names from the raw data for this cluster
                    metric_names = self._extract_metric_names_for_cluster(raw_results, cluster_name)
                    
                    multi_cluster_data[cluster_name] = {
                        'masked_timeseries': conversion_result['masked_timeseries'],
                        'metric_names': metric_names,
                        'node_names': conversion_result['node_names']
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to convert data for cluster {cluster_name}: {str(e)}")
                    continue
            
            if not multi_cluster_data:
                raise ValueError("No clusters could be successfully converted")
            
            # Step 3: TOTO forecasting for each cluster 
            model_config = self.config['models'].get('toto', {})
            toto_adapter = TOTOAdapter(model_config)
            quantiles = self.config['models'].get('quantiles', [0.1, 0.5, 0.9])
            forecast_horizon_days = self.config['models'].get('forecast_horizon', 7)
            
            cluster_toto_forecasts = {}
            for cluster_name, cluster_info in multi_cluster_data.items():
                logger.debug(f"Generating TOTO forecast for cluster: {cluster_name}")
                
                # Calculate future steps based on input series time intervals
                masked_timeseries = cluster_info['masked_timeseries']
                time_interval_seconds = masked_timeseries.time_interval_seconds[0].item()  # Get interval from first variate
                
                # Convert forecast horizon from days to number of timesteps
                seconds_per_day = 24 * 60 * 60
                forecast_horizon_seconds = forecast_horizon_days * seconds_per_day
                future_steps = int(forecast_horizon_seconds / time_interval_seconds)
                
                logger.debug(f"Cluster {cluster_name}: time_interval={time_interval_seconds}s, "
                           f"forecast_horizon={forecast_horizon_days} days, future_steps={future_steps}")
                
                toto_forecast = toto_adapter.forecast(
                    series=masked_timeseries,
                    horizon=future_steps,  # Use calculated timesteps instead of raw days
                    quantiles=quantiles,
                    metric_names=cluster_info['metric_names']
                )
                
                cluster_toto_forecasts[cluster_name] = toto_forecast
            
            # Step 4: Transform TOTO output to cluster-grouped format (clean separation)
            format_converter = ForecastFormatConverter()
            cluster_grouped_results = {}
            
            for cluster_name, toto_output in cluster_toto_forecasts.items():
                logger.debug(f"Converting forecast format for cluster: {cluster_name}")
                
                # Get actual node names for this cluster from our extracted data
                cluster_info = multi_cluster_data.get(cluster_name, {})
                node_names = cluster_info.get('node_names', ['cluster-aggregate'])
                metric_names = cluster_info.get('metric_names', [])
                
                metrics_by_level = MetricTypeClassifier.get_metrics_by_level(set(metric_names))
                
                cluster_forecasts = []
                
                # Handle cluster-level metrics (single entry with cluster-aggregate)
                cluster_level_metrics = metrics_by_level[MetricAggregationLevel.CLUSTER]
                if cluster_level_metrics:
                    # Filter TOTO output to only cluster-level metrics to avoid duplication
                    cluster_filtered_output = self._filter_toto_output_by_metrics(
                        toto_output, list(cluster_level_metrics)
                    )
                    cluster_aggregate_forecasts = format_converter.convert_to_cluster_grouped_format(
                        toto_forecast_output=cluster_filtered_output,
                        cluster_name=cluster_name,
                        node_name="cluster-aggregate",  # Single entry for cluster-level metrics
                        extra_labels={}
                    )
                    cluster_forecasts.extend(cluster_aggregate_forecasts)
                    logger.debug(f"Added {len(cluster_aggregate_forecasts)} cluster-level forecasts for cluster {cluster_name}")
                
                # Handle node-level metrics (multiple entries per node)
                node_level_metrics = metrics_by_level[MetricAggregationLevel.NODE]
                if node_level_metrics:
                    # Filter TOTO output to only node-level metrics to avoid duplication
                    node_filtered_output = self._filter_toto_output_by_metrics(
                        toto_output, list(node_level_metrics)
                    )
                    for node_name in node_names:
                        if node_name != 'cluster-aggregate':  # Skip aggregate for node-level metrics
                            node_specific_forecasts = format_converter.convert_to_cluster_grouped_format(
                                toto_forecast_output=node_filtered_output,
                                cluster_name=cluster_name,
                                node_name=node_name,  # Use actual node name for node-level metrics
                                extra_labels={}
                            )
                            cluster_forecasts.extend(node_specific_forecasts)
                            logger.debug(f"Added {len(node_specific_forecasts)} node-level forecasts for node {node_name}")
                
                cluster_grouped_results[cluster_name] = cluster_forecasts
            
            logger.info(f"Successfully generated cluster-grouped forecasts for {len(cluster_grouped_results)} clusters")
            return cluster_grouped_results
            
        except Exception as e:
            logger.error(f"Error generating TOTO forecast: {str(e)}")
            raise
    
    def _filter_toto_output_by_metrics(self, toto_output: Dict[str, Any], selected_metrics: List[str]) -> Dict[str, Any]:
        """
        Filter TOTO forecast output to include only selected metrics.
        
        This prevents metric duplication by ensuring cluster-level and node-level
        processing only work with their respective metric types.
        
        Args:
            toto_output: Complete TOTO forecast output with all metrics
            selected_metrics: List of metric names to include in filtered output
            
        Returns:
            Filtered TOTO output containing only the selected metrics
        """
        # Filter quantiles to only include selected metrics
        filtered_quantiles = {}
        original_quantiles = toto_output.get('quantiles', {})
        
        for metric_name in selected_metrics:
            if metric_name in original_quantiles:
                filtered_quantiles[metric_name] = original_quantiles[metric_name]
        
        # Create filtered output preserving all other fields
        filtered_output = {
            'quantiles': filtered_quantiles,
            'timestamps': toto_output.get('timestamps', []),
            'metric_names': selected_metrics,
            'horizon': toto_output.get('horizon', 168),
            'time_interval_seconds': toto_output.get('time_interval_seconds', 3600)
        }
        
        # Preserve any additional fields that might be present
        for key, value in toto_output.items():
            if key not in filtered_output:
                filtered_output[key] = value
        
        logger.debug(f"Filtered TOTO output: {len(selected_metrics)} metrics selected from {len(original_quantiles)} total")
        return filtered_output
    
    def _extract_cluster_names_from_raw_results(self, raw_results: Dict[str, Any]) -> List[str]:
        """Extract unique cluster names from raw Prometheus results."""
        cluster_names = set()
        
        for metric_name, metric_results in raw_results.items():
            if isinstance(metric_results, list):
                for result in metric_results:
                    if isinstance(result, dict) and 'metric' in result:
                        cluster_name = result['metric'].get('clusterName')
                        if cluster_name:
                            cluster_names.add(cluster_name)
        
        return list(cluster_names)
    
    def _extract_metric_names_for_cluster(self, raw_results: Dict[str, Any], cluster_name: str) -> List[str]:
        """Extract metric names that have data for the specified cluster."""
        metric_names = []
        
        for metric_name, metric_results in raw_results.items():
            if isinstance(metric_results, list):
                # Check if this metric has data for the specified cluster
                for result in metric_results:
                    if isinstance(result, dict) and 'metric' in result:
                        if result['metric'].get('clusterName') == cluster_name:
                            metric_names.append(metric_name)
                            break  # Found data for this metric and cluster
        
        return metric_names

    def validate_forecasts(self, cluster_timeseries: Dict[str, TimeSeries]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Validate forecast accuracy using 70/30 train/test split."""
        from validation import ForecastValidator
        
        validator = ForecastValidator(train_ratio=0.7)
        model_type = self.config['models'].get('type', 'nbeats')
        
        if model_type == 'toto':
            from adapters.forecasting.toto_adapter import TOTOAdapter
            adapter_class = TOTOAdapter
            model_config = self.config['models'].get('toto', {})
        else:
            from adapters.forecasting.nbeats_adapter import NBEATSAdapter
            adapter_class = NBEATSAdapter
            model_config = self.config['models'].get('nbeats', {})
        
        # Add quantiles to config
        model_config['quantiles'] = self.config['models'].get('quantiles', [0.1, 0.5, 0.9])
        
        logger.info(f"Starting validation with {model_type} model...")
        results = validator.validate_all_clusters(adapter_class, cluster_timeseries, model_config)
        
        # Log summary
        summary_df = validator.summarize_validation_results(results)
        if not summary_df.empty:
            logger.info(f"Validation completed for {len(summary_df)} components across {len(results)} clusters")
        
        return results
    
    def run(self):
        """Run the forecasting agent continuously."""
        global exported_forecasts
        metrics_conf = self.config['metrics']
        api_port = metrics_conf.get('forecast_api_port', metrics_conf['port'])
        api_host = metrics_conf.get('forecast_api_host', metrics_conf.get('host', '0.0.0.0'))
        _start_forecast_server(api_host, api_port)
        
        # Check if validation is enabled
        enable_validation = self.config.get('validation', {}).get('enabled', False)
        validation_interval = self.config.get('validation', {}).get('interval_cycles', 5)
        cycle_count = 0
        
        while True:
            try:
                # Check Prometheus health before collecting metrics
                health = self.collector.health_check()
                if "unhealthy" in health["status"]:
                    logger.warning(f"Prometheus collector health check failed: {health['status']}")
                    logger.info("Retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                logger.info("Collecting metrics and generating forecasts...")
                
                # Collect raw Prometheus data for clean architecture
                raw_results = self.collect_raw_metrics()
                
                cluster_timeseries = None
                if enable_validation and cycle_count % validation_interval == 0:
                    cluster_timeseries = self.collect_metrics_timeseries()
                    logger.info("Running forecast validation...")
                    try:
                        validation_results = self.validate_forecasts(cluster_timeseries)
                        logger.info("Validation completed successfully")
                    except Exception as e:
                        logger.error(f"Validation failed: {str(e)}")
                
                # Generate forecasts using clean architecture
                model_type = self.config['models'].get('type', 'nbeats')
                
                if model_type == 'toto':
                    # Clean architecture: Raw JSON → Tensors → TOTO forecast → Cluster-grouped format
                    logger.info("Generating TOTO forecasts using clean architecture...")
                    cluster_grouped_forecasts = self.generate_forecast_TOTO(raw_results)
                    
                    # Update exported forecasts with cluster-grouped results
                    for cluster_name, forecast_entries in cluster_grouped_forecasts.items():
                        exported_forecasts[cluster_name] = forecast_entries
                        logger.info(f"Generated clean architecture forecast for cluster: {cluster_name}")
                        
                else:
                    # For NBEATS, fall back to TimeSeries-based approach
                    if cluster_timeseries is None:
                        cluster_timeseries = self.collect_metrics_timeseries()
                    
                    for cluster, ts in cluster_timeseries.items():
                        forecast = self.generate_forecast_NBEATS(ts)
                        exported_forecasts[cluster] = forecast
                        logger.info(f"Generated NBEATS forecast for cluster: {cluster}")
                
                api_url = f"http://{api_host}:{api_port}/metrics/{{clustername}}"
                logger.info(f"Forecast metrics updated and available at {api_url}")
                # To plot all forecasts, uncomment the following line:
                # plot_all_forecasts(cluster_timeseries, exported_forecasts)
                
                cycle_count += 1
                logger.info(f"Waiting {self.config['agent']['interval']}s before next collection...")
                time.sleep(self.config['agent']['interval'])
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forecasting Agent')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    agent = ForecastingAgent(args.config)
    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info("Forecasting agent stopped by user")
    
if __name__ == '__main__':
    main() 
