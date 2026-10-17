"""Baseline forecasting models: recent average and time-of-week seasonal benchmarks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


class RecentAverageBaseline:
    """Predicts future liquidity flows using recent historical observations (e.g. lag-1 or rolling 24h)."""

    def __init__(self, method: str = "rolling_24h") -> None:
        self.method = method
        self.last_value: float = 0.0
        self.rolling_mean: float = 0.0

    def fit(self, series: pd.Series) -> RecentAverageBaseline:
        """Fit baseline on historical values."""
        clean = series.dropna().astype(float)
        if len(clean) == 0:
            self.last_value = 0.0
            self.rolling_mean = 0.0
        else:
            self.last_value = float(clean.iloc[-1])
            self.rolling_mean = float(clean.iloc[-24:].mean()) if len(clean) >= 24 else float(clean.mean())
        return self

    def predict(self, horizon_hours: int) -> float:
        """Predict flow for a given lead time."""
        if self.method == "last_value":
            return max(0.0, self.last_value)
        return max(0.0, self.rolling_mean)


class TimeOfWeekSeasonalBaseline:
    """Time-of-week seasonal baseline.

    Computes empirical average deposit and withdrawal flows conditioned on
    (day_of_week in [0..6], hour_of_day in [0..23]). Provides a strong cyclical benchmark.
    """

    def __init__(self) -> None:
        # Table of shape (7, 24)
        self.seasonal_table: np.ndarray = np.zeros((7, 24), dtype=float)
        self.global_mean: float = 0.0

    def fit(self, df: pd.DataFrame, target_col: str) -> TimeOfWeekSeasonalBaseline:
        """Fit seasonal table on historical data without future information."""
        work_df = df[[TIMESTAMP_COL, target_col]].dropna().copy()
        if not pd.api.types.is_datetime64_any_dtype(work_df[TIMESTAMP_COL]):
            work_df[TIMESTAMP_COL] = pd.to_datetime(work_df[TIMESTAMP_COL], utc=True)

        work_df["dow"] = work_df[TIMESTAMP_COL].dt.dayofweek
        work_df["hour"] = work_df[TIMESTAMP_COL].dt.hour
        work_df["target"] = work_df[target_col].astype(float)

        self.global_mean = float(work_df["target"].mean()) if len(work_df) > 0 else 0.0

        # Group by (dow, hour)
        grouped = work_df.groupby(["dow", "hour"])["target"].mean()

        for dow in range(7):
            for hr in range(24):
                if (dow, hr) in grouped.index:
                    self.seasonal_table[dow, hr] = float(grouped.loc[(dow, hr)])
                else:
                    self.seasonal_table[dow, hr] = self.global_mean

        return self

    def predict(self, target_timestamp: pd.Timestamp | pd.DatetimeIndex | pd.Series) -> np.ndarray:
        """Predict for target timestamp(s)."""
        if isinstance(target_timestamp, pd.Timestamp):
            dow = target_timestamp.dayofweek
            hr = target_timestamp.hour
            return np.array([max(0.0, self.seasonal_table[dow, hr])])

        # Array/Series of timestamps
        if isinstance(target_timestamp, pd.DatetimeIndex):
            dows = target_timestamp.dayofweek.values
            hrs = target_timestamp.hour.values
        else:
            dows = target_timestamp.dt.dayofweek.values
            hrs = target_timestamp.dt.hour.values

        preds = np.zeros(len(target_timestamp), dtype=float)
        for i in range(len(target_timestamp)):
            preds[i] = max(0.0, self.seasonal_table[dows[i], hrs[i]])
        return preds

