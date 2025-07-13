import logging
from typing import List, Dict, Optional, Any
import time

import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries  
from toto.inference.forecaster import TotoForecaster  
from toto.model.toto import Toto 

class TOTOAdapter:
    """Adapter for Datadog's *Toto* zero-shot forecasting model.

    Converts a Darts ``TimeSeries`` into the tensor format expected by Toto
    (``MaskedTimeseries``), runs zero-shot inference, then converts the result
    back into the JSON schema.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def forecast(
        self,
        series: MaskedTimeseries,
        horizon: int,
        quantiles: Optional[List[float]] = None,
        variate_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate a horizon-step forecast for the given MaskedTimeseries.

        Parameters
        ----------
        series : MaskedTimeseries
            Historical multivariate tensor input data.
        horizon : int
            Number of future steps to predict.
        quantiles : list[float], optional
            Quantiles to emit; defaults to [0.1, 0.5, 0.9].
        variate_metadata : list[dict], optional
            Metadata for each tensor variate. Each dict should contain:
            - 'metric_name': Name of the metric (e.g., 'cost_usd_per_node')
            - 'node_name': Name of the node (e.g., 'test-cluster-cp-0')
            - 'variate_id': Unique identifier for this variate
            If not provided, generic metadata will be generated.

        Returns
        -------
        Dict containing:
            - 'samples': Raw forecast samples tensor
            - 'series': Series data organized by variate_id, each containing quantiles and metadata
            - 'timestamps': Future timestamps
            - 'variate_metadata': Metadata for each variate
            - 'horizon': Forecast horizon
            - 'time_interval_seconds': Time interval in seconds
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]
        
        # Set default variate metadata if not provided
        if variate_metadata is None:
            variate_metadata = [
                {
                    'metric_name': f'metric_{i}',
                    'node_name': f'node_{i}',
                    'variate_id': f'variate_{i}'
                }
                for i in range(series.series.shape[0])
            ]
        
        # Run TOTO inference
        forecast_result = self._run_toto_inference(series, horizon)
        
        # Generate future timestamps
        last_timestamp = series.timestamp_seconds[0, -1].item()
        time_interval = series.time_interval_seconds[0].item()
        future_timestamps = [
            int((last_timestamp + (i + 1) * time_interval) * 1000)  # Convert to milliseconds
            for i in range(horizon)
        ]
        
        # Calculate quantiles for each variate (node-metric combination)
        samples = forecast_result.samples  # Shape: (num_samples, n_variates, horizon)
        series_results = {}
        
        for variate_idx, metadata in enumerate(variate_metadata):
            variate_samples = samples[:, variate_idx, :]  # Shape: (num_samples, horizon)
            variate_quantiles = {}
            
            for q in quantiles:
                quantile_values = []
                for time_idx in range(horizon):
                    time_samples = variate_samples[:, time_idx]  # Shape: (num_samples,)
                    quantile_value = torch.quantile(time_samples, q).item()
                    quantile_values.append(self._to_scalar(quantile_value))
                variate_quantiles[f"q{q:.2f}"] = quantile_values
            
            # Use variate_id as key to ensure uniqueness
            series_results[metadata['variate_id']] = {
                'quantiles': variate_quantiles,
                'metadata': metadata
            }
        
        return {
            'samples': samples,
            'series': series_results,
            'timestamps': future_timestamps,
            'variate_metadata': variate_metadata,
            'horizon': horizon,
            'time_interval_seconds': time_interval
        }
    
    def _run_toto_inference(self, masked_timeseries: MaskedTimeseries, horizon: int):
        """Run TOTO model inference on tensor data.
        
        Args:
            masked_timeseries: Input tensor data
            horizon: Number of future steps to predict
            
        Returns:
            TOTO forecast result object
        """
        device = "cpu"
        
        try:
            # Load TOTO model
            checkpoint = self.config.get("checkpoint", "Datadog/Toto-Open-Base-1.0")
            toto_model = Toto.from_pretrained(checkpoint).to(device)
            
            if self.config.get("compile", True):
                toto_model.compile()
            
            forecaster = TotoForecaster(toto_model.model)
            
            # Run forecast
            forecast_result = forecaster.forecast(
                masked_timeseries,
                # We can set any number of timesteps into the future that we'd like to forecast. Because Toto is an autoregressive model,
                # the inference time will be longer for longer forecasts. 
                prediction_length=horizon,
                # TOTOForecaster draws samples from a predicted parametric distribution. The more samples, the more stable and accurate the prediction.
                # This is especially important if you care about accurate prediction intervals in the tails.
                num_samples=self.config.get("num_samples", 256),
                # TOTOForecaster also handles batching the samples in order to control memory usage.
                # Set samples_per_batch as high as you can without getting OOMs for maximum performance.
                # If you're doing batch inference, the effective batch size sent to the model is (batch_size x samples_per_batch).
                samples_per_batch=self.config.get("samples_per_batch", 256),
                # KV cache should significantly speed up inference, and in most cases should reduce memory usage too.
                use_kv_cache=self.config.get("use_kv_cache", True),
            )
            
            return forecast_result
            
        except Exception as e:
            self.logger.error(f"TOTO inference failed: {str(e)}")
            raise

    def _to_scalar(self, x):
        """Return plain float from numpy/PyTorch scalar or array."""
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])  # first element
        return float(x)
