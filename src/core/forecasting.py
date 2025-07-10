from abc import ABC, abstractmethod
import pandas as pd

class ForecastingModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def forecast(self, horizon: int, frequency: str) -> dict:
        pass

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, metrics: list = None) -> dict:
        pass

    @abstractmethod
    def plot_forecast(self, forecast: dict, save_path: str = None):
        pass