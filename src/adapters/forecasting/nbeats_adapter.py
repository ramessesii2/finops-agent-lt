from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from typing import Optional
from darts.metrics import mape, smape, rmse
from functools import reduce
import numpy as np
import operator

class NBEATSAdapter:
    def __init__(self, config: dict):
        self.input_chunk_length = config.get("input_chunk_length", 24)
        self.output_chunk_length = config.get("output_chunk_length", 12)
        self.n_epochs = config.get("n_epochs", 100)
        # Handle likelihood specified as a string in the YAML config by instantiating the
        # corresponding Darts Likelihood class. Darts expects an *instance* of a Likelihood,
        # not a plain string, otherwise attribute access like ``num_parameters`` will fail.
        likelihood_cfg = config.get("likelihood", None)
        if isinstance(likelihood_cfg, str):
            try:
                from darts.utils.likelihood_models import (
                    QuantileRegression,
                    PoissonLikelihood,
                    GaussianLikelihood,
                )
                _lk_map = {
                    "quantileregression": QuantileRegression,
                    "poisson": PoissonLikelihood,
                    "gaussian": GaussianLikelihood,
                }
                LikelihoodClass = _lk_map.get(likelihood_cfg.lower())
                if LikelihoodClass is None:
                    raise ValueError(
                        f"Unsupported likelihood string '{likelihood_cfg}'. "
                        "Supported: QuantileRegression, Poisson, Gaussian"
                    )
                # Instantiate with sensible defaults (QR gets default quantiles 0.1,0.5,0.9)
                likelihood_cfg = LikelihoodClass()
            except (ImportError, ValueError) as exc:
                # Fallback: disable likelihood and warn – ensures model still trains.
                import logging
                logging.getLogger(__name__).warning(
                    "Could not instantiate likelihood '%s': %s. Proceeding without probabilistic forecasting.",
                    likelihood_cfg,
                    exc,
                )
                likelihood_cfg = None
        self.likelihood = likelihood_cfg
        self.model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=config.get("random_state", 42),
            activation=config.get("activation", "ReLU"),
            generic_architecture=config.get("generic_architecture", True),
            likelihood=self.likelihood,
        )
        # Separate scalers for target and covariates to avoid data leakage
        self.target_scaler = Scaler()
        self.cov_scaler = Scaler()
        self.series_scaled = None
        self.past_covariates_scaled = None

    def fit(self, series: TimeSeries, past_covariates: Optional[TimeSeries] = None):
        """Fit the underlying N-BEATS model.
        If the provided series is shorter than ``input_chunk_length + output_chunk_length`` the
        ``input_chunk_length`` is automatically reduced so that the model can be trained instead
        of failing with ``ValueError`` coming from Darts.
        """
        # Dynamically shrink ``input_chunk_length`` when the training set is too small.
        ts_length = series.n_timesteps if hasattr(series, "n_timesteps") else len(series)
        min_required = self.input_chunk_length + self.output_chunk_length
        if ts_length < min_required:
            # Prevent negative or zero length
            adjusted_input_len = max(1, ts_length - self.output_chunk_length)
            import logging
            logging.getLogger(__name__).warning(
                "Training series length (%d) is smaller than the configured input (%d) + output (%d) = %d. "
                "Reducing input_chunk_length to %d so the model can be trained.",
                ts_length,
                self.input_chunk_length,
                self.output_chunk_length,
                min_required,
                adjusted_input_len,
            )
            # Re-initialise the underlying model with the updated input length while preserving the rest
            self.input_chunk_length = adjusted_input_len
            self.model = NBEATSModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                n_epochs=self.n_epochs,
                random_state=self.model.random_state if hasattr(self.model, "random_state") else 42,
                activation=self.model.activation if hasattr(self.model, "activation") else "ReLU",
                generic_architecture=self.model.generic_architecture if hasattr(self.model, "generic_architecture") else True,
                likelihood=self.likelihood,
            )

        # Cast to float32 only when running on Apple Silicon (Metal / MPS backend).
        # MPS does not support float64 tensors, whereas CUDA/CPU backends do. Detect
        # MPS at runtime so other systems can continue using the original dtype.
        try:
            import torch
            running_on_mps = torch.backends.mps.is_available()
        except Exception:
            running_on_mps = False

        if running_on_mps:
            series = series.astype(np.float32)
            if past_covariates is not None:
                past_covariates = past_covariates.astype(np.float32)

        # Proceed with the usual scaling + fitting workflow
        self.series_scaled = self.target_scaler.fit_transform(series)
        if past_covariates is not None:
            self.past_covariates_scaled = self.cov_scaler.fit_transform(past_covariates)
            self.model.fit(self.series_scaled, past_covariates=self.past_covariates_scaled, verbose=True)
        else:
            self.model.fit(self.series_scaled, verbose=True)

    def forecast(self, n: int, num_samples: int = 1, past_covariates: Optional[TimeSeries] = None, quantiles: Optional[list] = None):
        """Generate forecast and return in JSON format matching other adapters."""
        import pandas as pd
        
        if past_covariates is not None:
            pc_scaled = self.cov_scaler.transform(past_covariates)
            forecast_scaled = self.model.predict(n, num_samples=num_samples, past_covariates=pc_scaled)
        else:
            forecast_scaled = self.model.predict(n, num_samples=num_samples)
        # Inverse transform to original scale
        forecast = self.target_scaler.inverse_transform(forecast_scaled)
        
        # If quantiles requested, format as JSON like other adapters
        if quantiles is not None:
            return self._format_forecast_json(forecast, quantiles)
        
        return forecast
    
    def _format_forecast_json(self, forecast, quantiles):
        """Format forecast as JSON matching the schema used by other adapters."""
        import pandas as pd
        import numpy as np
        
        results = {}
        values = forecast.values(copy=False)
        timestamps = pd.to_datetime(forecast.time_index).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        
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

    def backtest(self, series: TimeSeries, past_covariates: Optional[TimeSeries] = None,
                 start: float = 0.8, stride: int = 1, metric: str = "mape"):
        """
        Perform rolling-origin backtesting and return (score, forecast).
        """
        target_scaled = self.target_scaler.transform(series)
        cov_scaled = self.cov_scaler.transform(past_covariates) if past_covariates is not None else None

        hist_forecasts = self.model.historical_forecasts(
            target_scaled,
            past_covariates=cov_scaled,
            forecast_horizon=self.output_chunk_length,
            stride=stride,
            start=start,
            retrain=False,
            verbose=True
        )

        if isinstance(hist_forecasts, list):
            forecast_scaled = reduce(operator.add, hist_forecasts)
        else:
            forecast_scaled = hist_forecasts

        forecast = self.target_scaler.inverse_transform(forecast_scaled)
        target_overlap = series.slice_intersect(forecast)

        if metric == "mape":
            score = mape(target_overlap, forecast)
        elif metric == "smape":
            score = smape(target_overlap, forecast)
        elif metric == "rmse":
            score = rmse(target_overlap, forecast)
        else:
            raise ValueError(f"Unsupported metric {metric}")

        return score, forecast

    def evaluate(self, actual: TimeSeries, forecast: TimeSeries, metric: str = "mape"):
        """
        Evaluate a forecast against actual data using common error metrics.
        """
        actual_overlap = actual.slice_intersect(forecast)
        if metric == "mape":
            return mape(actual_overlap, forecast)
        elif metric == "smape":
            return smape(actual_overlap, forecast)
        elif metric == "rmse":
            return rmse(actual_overlap, forecast)
        else:
            raise ValueError(f"Unsupported metric {metric}")

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = NBEATSModel.load(path)