import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller

def regularize_timeseries(df, freq=None, value_col='y'):
    """
    Ensures the DataFrame has a regular datetime index in 'ds' and fills missing values using Darts' MissingValuesFiller.
    - freq: e.g. 'D' for daily, 'H' for hourly. If None, tries to infer.
    - value_col: the column with the time series values.
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds']).dt.normalize()  # Normalize to midnight for daily data
    df = df.drop_duplicates(subset='ds')
    df = df.sort_values('ds')
    if freq is None:
        freq = pd.infer_freq(df['ds'])
    if freq is None:
        freq = 'D'  # Default to daily if cannot infer
    # Create TimeSeries with missing dates filled
    series = TimeSeries.from_dataframe(df, time_col='ds', value_cols=value_col, fill_missing_dates=True, freq=freq)
    # Fill missing values using Darts' transformer
    series = MissingValuesFiller().transform(series)
    # Convert back to DataFrame for compatibility
    out_df = series.pd_dataframe().reset_index().rename(columns={'time': 'ds', series.columns[0]: value_col})
    return out_df

def dataframe_to_timeseries(df: pd.DataFrame, time_col: str = 'ds', value_col: str = 'y', freq: str = 'D') -> TimeSeries:
    """
    Convert a tidy ``pandas.DataFrame`` to a Darts ``TimeSeries``.

    Handles multiple metrics/labels per timestamp by pivoting to wide format,
    so each unique metric/label combination becomes a separate component in the resulting TimeSeries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least ``time_col`` and ``value_col``.
    time_col : str, optional
        Name of the datetime column. Defaults to ``'ds'``.
    value_col : str, optional
        Name of the value column. Defaults to ``'y'``.
    freq : str, optional
        Pandas/Darts frequency string used when filling missing dates.
        Defaults to daily (``'D'``).

    Returns
    -------
    darts.TimeSeries
        A gap-free ``TimeSeries`` with missing dates filled and missing
        values forward-filled using :class:`darts.dataprocessing.transformers.MissingValuesFiller`.
        Each unique metric/label combination becomes a separate component.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing_cols = {time_col, value_col} - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Identify columns that define a unique series (e.g., metric_name, labels)
    other_cols = [c for c in df.columns if c not in {time_col, value_col, "timestamp"}]
    df = df[[time_col, value_col] + other_cols].copy() 

    # Build a unique column identifier from metric name / labels to keep multiple series.
    if other_cols:
        df['series_id'] = df[other_cols].astype(str).agg('_'.join, axis=1)
    else:
        df['series_id'] = 'value'

    # Pivot to wide format (one column per metric/label combination)
    pivot_df = (
        df.pivot(index=time_col, columns='series_id', values=value_col)
          .sort_index()
          .reset_index()
    )

    # Create TimeSeries – use all numeric columns automatically
    series = TimeSeries.from_dataframe(
        pivot_df,
        time_col=time_col,
        value_cols=None,  # all columns except time become components
        fill_missing_dates=True,
        freq=freq,
    )
    series = MissingValuesFiller().transform(series)
    print(f"Series head:\n{series.head()}") 
    return series

