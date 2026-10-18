"""Feature engineering for treasury liquidity forecasting with strict lookahead-leakage prevention."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    NET_FLOW_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
DEFAULT_ROLLING_WINDOWS = [6, 24, 72, 168]


class FeatureEngineer:
    """Constructs lag, rolling, and temporal features strictly without future information leakage."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ) -> None:
        self.lags = lags or DEFAULT_LAGS
        self.rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS

    def create_features(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generates feature set for the given dataframe.

        CRITICAL LEAKAGE RULE:
        All lag and rolling features at row t are derived strictly from rows <= t-1.
        Row t values of target columns are NEVER accessed to compute row t predictors.
        """
        targets = target_cols or [DEPOSITS_COL, WITHDRAWALS_COL, NET_FLOW_COL]
        featured = df.copy()

        # Ensure timestamps are parsed
        if not pd.api.types.is_datetime64_any_dtype(featured[TIMESTAMP_COL]):
            featured[TIMESTAMP_COL] = pd.to_datetime(featured[TIMESTAMP_COL], utc=True)

        dt = featured[TIMESTAMP_COL].dt

        # 1. Temporal calendar features (exogenous, known in advance for any target timestamp)
        featured["hour_of_day"] = dt.hour
        featured["day_of_week"] = dt.dayofweek
        featured["day_of_month"] = dt.day
        featured["month"] = dt.month
        featured["is_weekend"] = (dt.dayofweek >= 5).astype(int)
        featured["is_business_hour"] = ((dt.hour >= 9) & (dt.hour <= 17) & (dt.dayofweek < 5)).astype(int)

        # Cyclical sinusoidal embeddings
        featured["sin_hour"] = np.sin(2.0 * np.pi * dt.hour / 24.0)
        featured["cos_hour"] = np.cos(2.0 * np.pi * dt.hour / 24.0)
        featured["sin_dow"] = np.sin(2.0 * np.pi * dt.dayofweek / 7.0)
        featured["cos_dow"] = np.cos(2.0 * np.pi * dt.dayofweek / 7.0)

        # 2. Lag features: lag_k(x)_t = x_{t-k}
        for col in targets:
            if col in featured.columns:
                series = featured[col].astype(float)
                for lag in self.lags:
                    featured[f"{col}_lag_{lag}"] = series.shift(lag)

        # 3. Rolling statistics strictly on lagged series: rolling_stat(x.shift(1))
        # Shifting by 1 guarantees row t does NOT see row t in the rolling calculation!
        for col in targets:
            if col in featured.columns:
                # shift by 1 first to prevent contemporaneous leakage
                lagged_series = featured[col].astype(float).shift(1)
                for window in self.rolling_windows:
                    roll = lagged_series.rolling(window=window, min_periods=max(2, window // 4))
                    featured[f"{col}_roll_mean_{window}h"] = roll.mean()
                    featured[f"{col}_roll_std_{window}h"] = roll.std().fillna(0.0)
                    featured[f"{col}_roll_min_{window}h"] = roll.min()
                    featured[f"{col}_roll_max_{window}h"] = roll.max()

        # 4. Ratios and interaction features
        for col in [DEPOSITS_COL, WITHDRAWALS_COL]:
            lag1 = featured.get(f"{col}_lag_1")
            roll24 = featured.get(f"{col}_roll_mean_24h")
            if lag1 is not None and roll24 is not None:
                featured[f"{col}_ratio_lag1_to_roll24"] = np.where(roll24 > 0, lag1 / roll24, 1.0)

        return featured

    def get_feature_column_names(self, df: pd.DataFrame) -> List[str]:
        """Extract all predictor feature column names (excluding timestamps, targets, and raw index)."""
        exclude = {
            TIMESTAMP_COL,
            DEPOSITS_COL,
            WITHDRAWALS_COL,
            NET_FLOW_COL,
            "index",
        }
        return [c for c in df.columns if c not in exclude and not c.endswith("_target")]

    @staticmethod
    def assert_no_lookahead_leakage(
        original_df: pd.DataFrame,
        featured_df: pd.DataFrame,
        perturbed_index: int,
        target_col: str = DEPOSITS_COL,
    ) -> bool:
        """Rigorous leakage verification test.

        Modifies the raw value at perturbed_index in original_df, recomputes features,
        and verifies that NO feature at any row t <= perturbed_index changed.
        """
        engineer = FeatureEngineer()
        features_before = engineer.create_features(original_df)

        modified_df = original_df.copy()
        modified_df.loc[perturbed_index, target_col] += 100_000_000.0  # Massive perturbation
        features_after = engineer.create_features(modified_df)

        feature_cols = engineer.get_feature_column_names(features_before)

        # Check rows <= perturbed_index: features at row t must be identical
        for row_idx in range(perturbed_index + 1):
            before_vals = features_before.loc[row_idx, feature_cols].values
            after_vals = features_after.loc[row_idx, feature_cols].values
            # Handle NaN equality
            diff = np.abs(np.nan_to_num(before_vals) - np.nan_to_num(after_vals))
            if np.any(diff > 1e-6):
                return False  # Leakage detected!

        return True  # No leakage detected
