"""Data preprocessing module for timezone conversion, unit normalization, gap imputation, and field reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from reserve_juggler.contracts import (
    DataQualityStatus,
    EventPublisher,
    LearningEvent,
    NoOpEventPublisher,
)
from reserve_juggler.data.schema import (
    DEFAULT_ASSUMPTIONS,
    DEPOSITS_COL,
    MARKET_HOURS_OPEN_COL,
    NET_FLOW_COL,
    RATE_COLUMNS,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_MATURE_HOUR,
    REPO_SWEEP_OPEN_COL,
    REPO_SWEEP_OPEN_HOUR,
    TIER1_EARLY_EXIT_FEE_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_DURATION_COL,
    TIER2_BOND_PRICE_COL,
    TIER2_BOND_YIELD_COL,
    TIER2_EMERGENCY_HAIRCUT_COL,
    TIER3_EMERGENCY_SLIPPAGE_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


class DatasetPreprocessor:
    """Cleans, normalizes, and prepares raw datasets for modeling and simulation."""

    def __init__(
        self,
        operating_timezone: str = "UTC",
        source_timezone: str = "UTC",
        event_publisher: Optional[EventPublisher] = None,
    ) -> None:
        self.operating_timezone = operating_timezone
        self.source_timezone = source_timezone
        self.publisher = event_publisher or NoOpEventPublisher()
        self.imputed_timestamps: List[datetime] = []

    def preprocess(
        self,
        df: pd.DataFrame,
        fill_hourly_gaps: bool = True,
        imputation_method: str = "interpolate",
    ) -> pd.DataFrame:
        """Preprocesses the raw dataframe:
        - Parses and standardizes timestamps to UTC.
        - Removes exact duplicates and sorts chronologically.
        - Fills gaps in hourly timeline if requested, recording all imputed timestamps.
        - Normalizes percentage rates (e.g. 4.8% -> 0.048).
        - Reconciles net_flow = deposits - withdrawals.
        - Sets default columns and financial penalty parameters.
        """
        clean_df = df.copy()

        # 1. Parse timestamps to UTC
        if not pd.api.types.is_datetime64_any_dtype(clean_df[TIMESTAMP_COL]):
            clean_df[TIMESTAMP_COL] = pd.to_datetime(clean_df[TIMESTAMP_COL])

        if clean_df[TIMESTAMP_COL].dt.tz is None:
            clean_df[TIMESTAMP_COL] = clean_df[TIMESTAMP_COL].dt.tz_localize(self.source_timezone).dt.tz_convert("UTC")
        else:
            clean_df[TIMESTAMP_COL] = clean_df[TIMESTAMP_COL].dt.tz_convert("UTC")

        # 2. Sort chronologically and drop duplicate timestamps
        clean_df = clean_df.sort_values(by=TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="first").reset_index(drop=True)

        # 3. Fill hourly gaps if enabled
        if fill_hourly_gaps and len(clean_df) > 1:
            full_index = pd.date_range(
                start=clean_df[TIMESTAMP_COL].iloc[0],
                end=clean_df[TIMESTAMP_COL].iloc[-1],
                freq="1h",
                tz="UTC",
                name=TIMESTAMP_COL,
            )
            clean_df = clean_df.set_index(TIMESTAMP_COL).reindex(full_index)
            missing_mask = clean_df[DEPOSITS_COL].isna()
            if missing_mask.any():
                missing_indices = clean_df.index[missing_mask].tolist()
                self.imputed_timestamps = [ts.to_pydatetime() for ts in missing_indices]

                # Impute numeric columns
                numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
                if imputation_method == "interpolate":
                    clean_df[numeric_cols] = clean_df[numeric_cols].interpolate(method="time").bfill().ffill()
                else:
                    clean_df[numeric_cols] = clean_df[numeric_cols].ffill().bfill()

                # Emit imputation learning event
                now = datetime.now(timezone.utc)
                evt = LearningEvent(
                    event_id=f"imp_{int(now.timestamp())}_{len(self.imputed_timestamps)}",
                    event_type="data_gap_imputed",
                    source_module="reserve_juggler.data.preprocessor",
                    event_timestamp=now,
                    recording_timestamp=now,
                    payload={
                        "imputed_count": len(self.imputed_timestamps),
                        "first_imputed": str(self.imputed_timestamps[0]),
                        "last_imputed": str(self.imputed_timestamps[-1]),
                        "imputation_method": imputation_method,
                    },
                )
                self.publisher.publish(evt)

            clean_df = clean_df.reset_index()

        # 4. Normalize percentage columns to decimal rates
        for col in RATE_COLUMNS.intersection(clean_df.columns):
            s = clean_df[col].astype(float)
            if (s > 1.0).any():
                clean_df[col] = s / 100.0

        # 5. Populate defaults for supplementary columns if not present
        for col, default_val in DEFAULT_ASSUMPTIONS.items():
            if col not in clean_df.columns:
                clean_df[col] = default_val

        # 6. Reconcile net_flow = deposits_usd - withdrawals_usd
        clean_df[NET_FLOW_COL] = clean_df[DEPOSITS_COL].astype(float) - clean_df[WITHDRAWALS_COL].astype(float)

        # 7. Reconcile repo sweep timing window
        hours = clean_df[TIMESTAMP_COL].dt.hour
        if REPO_SWEEP_OPEN_COL not in clean_df.columns:
            clean_df[REPO_SWEEP_OPEN_COL] = (hours == REPO_SWEEP_OPEN_HOUR).astype(int)
        else:
            clean_df[REPO_SWEEP_OPEN_COL] = clean_df[REPO_SWEEP_OPEN_COL].astype(int)

        if REPO_SWEEP_MATURE_COL not in clean_df.columns:
            clean_df[REPO_SWEEP_MATURE_COL] = (hours == REPO_SWEEP_MATURE_HOUR).astype(int)
        else:
            clean_df[REPO_SWEEP_MATURE_COL] = clean_df[REPO_SWEEP_MATURE_COL].astype(int)

        # 8. Market hours open (US financial markets: weekdays 09:30-16:00 ET -> ~14:00 to 21:00 UTC)
        if MARKET_HOURS_OPEN_COL not in clean_df.columns:
            is_weekday = clean_df[TIMESTAMP_COL].dt.dayofweek < 5
            is_market_hour = (hours >= 14) & (hours < 21)
            clean_df[MARKET_HOURS_OPEN_COL] = (is_weekday & is_market_hour).astype(int)

        return clean_df
