import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries
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
        series: TimeSeries,
        horizon: int,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """Generate a *horizon*-step forecast for ``series``.

        Parameters
        ----------
        series : darts.TimeSeries
            Historical multivariate input.
        horizon : int
            Number of future steps to predict.
        quantiles : list[float], optional
            Quantiles to emit; defaults to ``[0.1, 0.5, 0.9]``.
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]
        device = "cpu"

        n_variates = len(series.components)
        context_length = min(self.config.get("context_length", 4096), len(series))
        ctx_ts = series[-context_length:]

        # Build input tensors 
        vals = series.values(copy=False)  # shape (time, variate)
        input_series = torch.from_numpy(vals.T).to(torch.float32).to(device)

        idx = pd.to_datetime(ctx_ts.time_index)
        ts_seconds = (idx.astype("int64") // 10 ** 9).to_numpy()
        timestamp_seconds = torch.from_numpy(ts_seconds).expand((n_variates, context_length)).to(device)

        if len(idx) >= 2:
            interval_sec = int((idx[1] - idx[0]).total_seconds())
        else:
            interval_sec = 3600
        time_interval_seconds = torch.full((n_variates,), interval_sec, device=device)

        inputs = MaskedTimeseries(
            series=input_series,
            padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(input_series),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # Run Toto zero-shot forecast.

        try:
            checkpoint = self.config.get("checkpoint", "Datadog/Toto-Open-Base-1.0")
            toto_model = Toto.from_pretrained(checkpoint).to(device)
            if self.config.get("compile", True):
                toto_model.compile()
            forecaster = TotoForecaster(toto_model.model)

            forecast_obj = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=self.config.get("num_samples", 256),
                samples_per_batch=self.config.get("samples_per_batch", 256),
                use_kv_cache=self.config.get("use_kv_cache", True),
            )

            # samples: (batch, n_variates, horizon, num_samples) – squeeze batch dim
            samples_np = np.squeeze(forecast_obj.samples.cpu().numpy(), axis=0)
        except Exception as exc:  # pragma: no cover
            self.logger.error("Toto inference failed: %s – falling back to naive forecast", exc)

        # Build output JSON
        start_dt = pd.to_datetime(series.time_index[-1]) + pd.to_timedelta(interval_sec, unit="s")
        ds_out = pd.date_range(start_dt, periods=horizon, freq=f"{interval_sec}s").strftime("%Y-%m-%dT%H:%M:%SZ")
        # print(samples_np.shape)  
        results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        for var_idx, comp in enumerate(series.components):
            comp_samples = samples_np[var_idx]  # (horizon, num_samples)
            comp_dict: Dict[str, List[Dict[str, float]]] = {}
            for q in quantiles:
                q_vals = np.quantile(comp_samples, q, axis=-1) # (horizon,)
                comp_dict[f"q{q:.2f}"] = [
                    {
                        "ds": t, 
                        "y": self._to_scalar(v)
                    } 
                    for t, v in zip(ds_out, q_vals)
                ]
            results[comp] = comp_dict

        return results

    def _to_scalar(self, x):
        """Return plain float from numpy/PyTorch scalar or array."""
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])  # first element
        return float(x)
