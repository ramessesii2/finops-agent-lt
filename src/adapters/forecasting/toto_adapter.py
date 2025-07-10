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
        metric_names: Optional[List[str]] = None
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
        metric_names : list[str], optional
            Names of metrics for the tensor variates.

        Returns
        -------
        Dict containing:
            - 'samples': Raw forecast samples tensor
            - 'quantiles': Calculated quantile values
            - 'timestamps': Future timestamps
            - 'metric_names': Names of metrics
            - 'horizon': Forecast horizon
            - 'time_interval_seconds': Time interval in seconds
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]
        
        # Set default metric names if not provided
        if metric_names is None:
            metric_names = [f"metric_{i}" for i in range(series.series.shape[0])]
        
        # Run TOTO inference
        forecast_result = self._run_toto_inference(series, horizon)
        
        # Generate future timestamps
        last_timestamp = series.timestamp_seconds[0, -1].item()
        time_interval = series.time_interval_seconds[0].item()
        future_timestamps = [
            int((last_timestamp + (i + 1) * time_interval) * 1000)  # Convert to milliseconds
            for i in range(horizon)
        ]
        
        # Calculate quantiles for each metric
        samples = forecast_result.samples  # Shape: (num_samples, n_variates, horizon)
        quantile_results = {}
        
        for variate_idx, metric_name in enumerate(metric_names):
            metric_samples = samples[:, variate_idx, :]  # Shape: (num_samples, horizon)
            metric_quantiles = {}
            
            for q in quantiles:
                quantile_values = []
                for time_idx in range(horizon):
                    time_samples = metric_samples[:, time_idx]  # Shape: (num_samples,)
                    quantile_value = torch.quantile(time_samples, q).item()
                    quantile_values.append(self._to_scalar(quantile_value))
                metric_quantiles[f"q{q:.2f}"] = quantile_values
            
            quantile_results[metric_name] = metric_quantiles
        
        # Return pure TOTO forecast output
        return {
            'samples': samples,
            'quantiles': quantile_results,
            'timestamps': future_timestamps,
            'metric_names': metric_names,
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
                prediction_length=horizon,
                num_samples=self.config.get("num_samples", 256),
                samples_per_batch=self.config.get("samples_per_batch", 256),
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
