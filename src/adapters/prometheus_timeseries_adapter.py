import pandas as pd
from typing import Dict, List
from darts import TimeSeries

class PrometheusTimeseriesAdapter:
    """
    Adapter to convert raw Prometheus query results to Darts TimeSeries.
    """
    @staticmethod
    def _result_to_df(result: list, metric_key: str) -> pd.DataFrame:
        rows = []
        for series in result:
            name = series["metric"].get("__name__", metric_key)
            labels = {k: v for k, v in series["metric"].items() if k != "__name__"}
            values = series.get("values") or ([series["value"]] if "value" in series else [])
            for ts, v in values:
                try:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(float(ts), unit="s"),
                            "metric_name": name,
                            "value": float(v),
                            "labels": labels,
                        }
                    )
                except (ValueError, TypeError):
                    continue
        return pd.DataFrame(rows)

    @staticmethod
    def _merge_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        labels_df = pd.json_normalize(df["labels"].tolist())
        df = pd.concat([df.drop(columns=["labels"]), labels_df], axis=1)
        if "clusterName" in df.columns and "cluster" not in df.columns:
            df = df.rename(columns={"clusterName": "cluster"})
        df = df.sort_values("timestamp")
        return df.reset_index(drop=True)

    @classmethod
    def prometheus_results_to_timeseries(cls, results: Dict[str, list], freq: str) -> Dict[str, TimeSeries]:
        """
        Convert a dict of raw Prometheus results to a dict of Darts TimeSeries per cluster.
        Args:
            results: Dict mapping metric keys to raw Prometheus result lists
        Returns:
            Dict[str, TimeSeries] mapping cluster name to its TimeSeries
        """ 
        all_dfs = []
        for metric_key, result in results.items():
            df = cls._result_to_df(result, metric_key)
            df = cls._merge_by_cluster(df)
            all_dfs.append(df)
        if not all_dfs:
            raise ValueError("No metrics data to convert.")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # Pivot to wide format: index=[timestamp, cluster], columns=metric_name, values=y
        merged_df = merged_df.rename(columns={"timestamp": "ds", "value": "y"})
        clusters = merged_df["cluster"].unique()
        ts_dict = {}
        for cluster in clusters:
            cluster_df = merged_df[merged_df["cluster"] == cluster]
            # Pivot so each metric is a column
            wide = cluster_df.pivot_table(index="ds", columns="metric_name", values="y")
            wide = wide.sort_index()
            # Build TimeSeries
            ts = TimeSeries.from_dataframe(wide, fill_missing_dates=True, freq=freq)
            ts_dict[cluster] = ts
        return ts_dict 