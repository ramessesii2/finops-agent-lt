import logging
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime, timedelta, timezone
import time
from darts import TimeSeries
import json
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collectors.prometheus import PrometheusCollector
from .optimizers.idle_capacity import IdleCapacityOptimizer
from forecasting_agent.adapters.forecasting.nbeats_adapter import NBEATSAdapter
from forecasting_agent.adapters.forecasting.toto_adapter import TOTOAdapter
from forecasting_agent.adapters.prometheus_timeseries_adapter import PrometheusTimeseriesAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROMQL = {
    # OpenCost: per-cluster node spend USD/h
    "cost_usd": (
        "sum(node_total_hourly_cost) by (clusterName)"
    ),
    # CPU % = (non-idle / total) * 100, 5-minute rate, summed per cluster
    "cpu_pct": (
        "100 * sum(rate(node_cpu_seconds_total{mode!=\"idle\"}[5m])) by (clusterName) "
        "/ sum(rate(node_cpu_seconds_total[5m])) by (clusterName)"
    ),
    # Memory % = (MemTotal-MemAvailable)/MemTotal * 100
    "mem_pct": (
        "100 * ( "
        "  sum(node_memory_MemTotal_bytes) by (clusterName) - "
        "  sum(node_memory_MemAvailable_bytes) by (clusterName) "
        ") / sum(node_memory_MemTotal_bytes) by (clusterName)"
    ),
}

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
            if self.path == "/metrics" or self.path == "/metrics/":
                # list clusters
                self._send_json({"clusters": list(exported_forecasts.keys())})
            elif self.path.startswith("/metrics/"):
                cluster = self.path[len("/metrics/"):]
                if cluster in exported_forecasts and cluster:
                    self._send_json(exported_forecasts[cluster])
                else:
                    self._send_json({"error": "cluster not found"}, status=404)
            else:
               self._send_json({"message": "OK"})

        def log_message(self, *args, **kwargs):
            # Silence default HTTP server logging
            return

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Forecast HTTP server running on {host}:{port}")

def plot_all_forecasts(cluster_timeseries, exported_forecasts):
    """Plot all clusters' forecasts (history + forecast) for each component."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from darts import TimeSeries
    import pandas as pd

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
            # Add extra window (scrape interval) so rate[...] has warm-up samples
            scrape_step = self.config['collector'].get('step', '5m')
            step_minutes = int(scrape_step.rstrip('m')) if scrape_step.endswith('m') else 5
            pad = timedelta(minutes=max(5, step_minutes))
            start_time = end_time - timedelta(days=self.config['collector'].get('lookback_days', 4)) - pad
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

    def generate_forecast_TOTO(self, series: TimeSeries) -> dict:
        """Generate forecast using Datadog Toto model based on config."""
        try:
            step = self.config['collector'].get('step', 'h')
            horizon_days = self.config['models'].get('forecast_horizon', 30)
            if step in ['h', '1h', 'hour']:
                horizon = horizon_days * 24
            elif step in ['m', '1m', 'min']:
                horizon = horizon_days * 24 * 60
            else:
                horizon = horizon_days
            model_config = self.config['models'].get('toto', {})
            adapter = TOTOAdapter(model_config)
            quantiles = self.config['models'].get('quantiles', [0.1, 0.5, 0.9])
            return adapter.forecast(series, horizon, quantiles)
        except Exception as e:
            logger.error(f"Error generating Toto forecast: {str(e)}")
            raise

    def run(self):
        """Run the forecasting agent continuously."""
        global exported_forecasts
        metrics_conf = self.config['metrics']
        api_port = metrics_conf.get('forecast_api_port', metrics_conf['port'])
        api_host = metrics_conf.get('forecast_api_host', metrics_conf.get('host', '0.0.0.0'))
        _start_forecast_server(api_host, api_port)
        
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
                cluster_timeseries = self.collect_metrics_timeseries()
                # Generate forecast for each cluster
                for cluster, ts in cluster_timeseries.items():
                    if self.config['models'].get('type', 'nbeats') == 'toto':
                        forecast = self.generate_forecast_TOTO(ts)
                    else:
                        forecast = self.generate_forecast_NBEATS(ts)
                    exported_forecasts[cluster] = forecast
                    logger.info(f"Generated forecast for cluster: {cluster}")
                
                api_url = f"http://{api_host}:{api_port}/metrics/{{clustername}}"
                logger.info(f"Forecast metrics updated and available at {api_url}")
                # To plot all forecasts, uncomment the following line:
                # plot_all_forecasts(cluster_timeseries, exported_forecasts)
                
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
