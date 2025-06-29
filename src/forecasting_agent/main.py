import asyncio
import logging
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime, timedelta
import pandas as pd
import time
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
import json
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collectors.prometheus import PrometheusCollector
from .optimizers.idle_capacity import IdleCapacityOptimizer
from forecasting_agent.adapters.forecasting.nbeats_adapter import NBEATSAdapter
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
            if self.path.startswith("/metrics/"):
                cluster = self.path[len("/metrics/"):]
                if cluster in exported_forecasts:
                    self._send_json(exported_forecasts[cluster])
                else:
                    self._send_json({"error": "cluster not found"}, status=404)
            elif self.path == "/metrics" or self.path == "/metrics/":
                self._send_json({"clusters": list(exported_forecasts.keys())})
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

    def collect_metrics_timeseries(self) -> Dict[str, TimeSeries]:
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config['collector']['lookback_days'])
            # Get raw results from PrometheusCollector
            results = self.collector.collect_metrics_timeseries(start_time, end_time, PROMQL)
            # Convert to per-cluster TimeSeries
            return PrometheusTimeseriesAdapter.prometheus_results_to_timeseries(results, freq=self.config['models']['nbeats'].get('freq', "h"))
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            # COLLECTION_ERRORS.labels(collector=self.collector.__class__.__name__).inc()
            raise

    def generate_forecast_NBEATS(self, series: TimeSeries) -> dict:
        try:
            # Ensure forecast horizon is 30 days ahead
            step = self.config['collector'].get('step', 'h')
            horizon = self.config['models'].get('forecast_horizon', 30)
            if step in ['h', '1h', 'hour']:
                horizon = horizon * 24  # 30 days of hourly steps
            elif step in ['m', '1m', 'min']:
                horizon = horizon * 24 * 60      # 30 days of daily steps

            model_type = self.config['models'].get('type', 'nbeats')
            if model_type != 'nbeats':
                raise ValueError("Only NBEATS is supported.")
            model_config = self.config['models'].get('nbeats', {})
            model = NBEATSAdapter(model_config)
            model.fit(series)
            likelihood = model_config.get('likelihood', None)
            num_samples = model_config.get('num_samples', 100) if likelihood else 1
            forecast = model.forecast(horizon, num_samples=num_samples)
            quantiles = self.config['models'].get('quantiles', [0.1, 0.5, 0.9])
            results = {}
            if isinstance(forecast, list):
                for i, ts in enumerate(forecast):
                    if isinstance(ts, list):
                        for j, ts_inner in enumerate(ts):
                            comp = f"component_{i}_{j}"
                            comp_results = {}
                            values = ts_inner.values(copy=False)
                            timestamps = pd.to_datetime(ts_inner.time_index).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
                            for q in quantiles:
                                if getattr(ts_inner, 'n_samples', 1) > 1:
                                    q_values = np.quantile(values, q, axis=1)
                                else:
                                    q_values = values[:, 0]
                                comp_results[f"q{q:.2f}"] = [
                                    {"ds": t, "y": float(v)} for t, v in zip(timestamps, q_values)
                                ]
                            results[comp] = comp_results
                    else:
                        comp = f"component_{i}"
                        comp_results = {}
                        values = ts.values(copy=False)
                        timestamps = pd.to_datetime(ts.time_index).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
                        for q in quantiles:
                            if getattr(ts, 'n_samples', 1) > 1:
                                q_values = np.quantile(values, q, axis=1)
                            else:
                                q_values = values[:, 0]
                            comp_results[f"q{q:.2f}"] = [
                                {"ds": t, "y": float(v)} for t, v in zip(timestamps, q_values)
                            ]
                        results[comp] = comp_results
            else:
                values = forecast.values(copy=False)
                timestamps = pd.to_datetime(forecast.time_index).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
                for comp_idx, comp in enumerate(forecast.components):
                    comp_results = {}
                    for q in quantiles:
                        if values.ndim == 3:
                            # [time, component, sample]
                            if getattr(forecast, 'n_samples', 1) > 1:
                                q_values = np.quantile(values[:, comp_idx, :], q, axis=1)
                            else:
                                q_values = values[:, comp_idx, 0]
                        elif values.ndim == 2:
                            # [time, sample] (single component)
                            if getattr(forecast, 'n_samples', 1) > 1:
                                q_values = np.quantile(values, q, axis=1)
                            else:
                                q_values = values[:, 0]
                        else:
                            raise ValueError(f"Unexpected values shape: {values.shape}")
                        comp_results[f"q{q:.2f}"] = [
                            {"ds": t, "y": float(v)} for t, v in zip(timestamps, q_values)
                        ]
                    results[comp] = comp_results
            return results
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def run(self):
        """Run the forecasting agent."""
        global exported_forecasts
        # start forecast JSON server (once)
        metrics_conf = self.config['metrics']
        api_port = metrics_conf.get('forecast_api_port', metrics_conf['port'])
        api_host = metrics_conf.get('forecast_api_host', metrics_conf.get('host', '0.0.0.0'))
        _start_forecast_server(api_host, api_port)
        try:
            cluster_timeseries = self.collect_metrics_timeseries()
            # Generate forecast for each cluster
            for cluster, ts in cluster_timeseries.items():
                forecast = self.generate_forecast_NBEATS(ts)
                exported_forecasts[cluster] = forecast
            api_url = f"http://{api_host}:{api_port}/metrics/{{clustername}}"
            logger.info(f"Forecast metrics are now ready to be ingested via the API at {api_url}")
            # To plot all forecasts, uncomment the following line:
            # plot_all_forecasts(cluster_timeseries, exported_forecasts)
            time.sleep(self.config['agent']['interval'])
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(60)

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
