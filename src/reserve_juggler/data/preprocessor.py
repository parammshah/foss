"""Data preprocessing module for timezone conversion, unit normalization, gap imputation, and field reconciliation."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from reserve_juggler.contracts import (
    DataQualityStatus,
    EventPublisher,
    LearningEvent,
    NoOpEventPublisher,
)
from reserve_juggler.data.schema import (
    ACTUAL_TO_CANONICAL_COLUMN_MAP,
    CANONICAL_TO_ACTUAL_COLUMN_MAP,
    DEFAULT_ASSUMPTIONS,
    DEPOSITS_COL,
    EMERGENCY_OVERDRAFT_PENALTY_APR_COL,
    MARKET_HOURS_OPEN_COL,
    NET_FLOW_COL,
    PERCENTAGE_COLUMNS,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_MATURE_HOUR,
    REPO_SWEEP_OPEN_COL,
    REPO_SWEEP_OPEN_HOUR,
    TIER1_EARLY_EXIT_FEE_COL,
    TIER1_EARLY_REDEMPTION_HAIRCUT_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_DURATION_COL,
    TIER2_BOND_LIQUIDATION_HAIRCUT_COL,
    TIER2_BOND_MOD_DURATION_COL,
    TIER2_BOND_PRICE_COL,
    TIER2_BOND_YIELD_COL,
    TIER2_EMERGENCY_HAIRCUT_COL,
    TIER3_EMERGENCY_SLIPPAGE_COL,
    TIER3_STOCK_LIQUIDATION_HAIRCUT_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


def resolve_dataset_path(path: Optional[str | Path] = None) -> Path:
    """Resolves dataset path from argument, environment variable, or default location.

    Does not hardcode user-specific paths into library logic.
    Raises FileNotFoundError if an explicitly specified path does not exist.
    """
    if path is not None:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                f"Specified dataset path does not exist: {target.resolve()}\n"
                f"Please ensure the file exists or configure RESERVE_JUGGLER_DATA_PATH."
            )
        return target

    # Check environment variable
    env_path = os.environ.get("RESERVE_JUGGLER_DATA_PATH")
    if env_path:
        target = Path(env_path)
        if not target.exists():
            raise FileNotFoundError(
                f"RESERVE_JUGGLER_DATA_PATH was set to '{env_path}', but file does not exist.\n"
                f"Please verify the environment variable."
            )
        return target

    # Check local benchmark dataset fallback
    default_local = Path("data/challenge_dataset_8760.csv")
    if default_local.exists():
        return default_local

    raise FileNotFoundError(
        "No dataset path provided and default data/challenge_dataset_8760.csv not found.\n"
        "Specify path via --input, function parameter, or RESERVE_JUGGLER_DATA_PATH environment variable."
    )


def load_treasury_dataset(
    path: Optional[str | Path] = None,
    preprocessor: Optional[DatasetPreprocessor] = None,
) -> pd.DataFrame:
    """Loads and preprocesses treasury dataset from resolved path."""
    resolved_path = resolve_dataset_path(path)
    df = pd.read_csv(resolved_path)
    prep = preprocessor or DatasetPreprocessor()
    return prep.preprocess(df)


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
        self.normalization_log: Dict[str, str] = {}

    def preprocess(
        self,
        df: pd.DataFrame,
        fill_hourly_gaps: bool = True,
        imputation_method: str = "interpolate",
    ) -> pd.DataFrame:
        """Preprocesses the raw dataframe:
        - Parses and standardizes timestamps to UTC.
        - Removes exact duplicates and sorts chronologically while tracking original ordering.
        - Fills gaps in hourly timeline if requested, recording all imputed timestamps.
        - Normalizes percentage rates once at the ingestion boundary (e.g. 4.8% -> 0.048, 0.5% -> 0.005).
        - Preserves original net_flow_usd and creates separate validated_net_flow_usd.
        - Sets bi-directional column aliases for seamless compatibility across actual and canonical schemas.
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

                numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
                if imputation_method == "interpolate":
                    clean_df[numeric_cols] = clean_df[numeric_cols].interpolate(method="time").bfill().ffill()
                else:
                    clean_df[numeric_cols] = clean_df[numeric_cols].ffill().bfill()

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

        # 4. Detect percentage scale and normalize rates once at boundary
        # If repo rate or bond yield has mean > 1.0, the entire file is on 0-100% scale
        is_percentage_scale = False
        for probe_col in [TIER1_REPO_RATE_COL, TIER2_BOND_YIELD_COL]:
            if probe_col in clean_df.columns:
                probe_mean = float(pd.to_numeric(clean_df[probe_col], errors="coerce").mean())
                if probe_mean > 1.0:
                    is_percentage_scale = True
                    break

        if is_percentage_scale:
            # Normalize all rate/penalty columns by 100.0
            for col in PERCENTAGE_COLUMNS.intersection(clean_df.columns):
                s = clean_df[col].astype(float)
                clean_df[col] = s / 100.0
                self.normalization_log[col] = "Divided by 100.0 (converted whole percentage points to decimal fraction)"
        else:
            for col in PERCENTAGE_COLUMNS.intersection(clean_df.columns):
                self.normalization_log[col] = "Preserved as decimal fraction (values already <= 1.0)"

        # 5. Populate bi-directional column aliases
        for actual_col, canon_col in ACTUAL_TO_CANONICAL_COLUMN_MAP.items():
            if actual_col in clean_df.columns and canon_col not in clean_df.columns:
                clean_df[canon_col] = clean_df[actual_col]
            elif canon_col in clean_df.columns and actual_col not in clean_df.columns:
                clean_df[actual_col] = clean_df[canon_col]

        # 6. Populate defaults for supplementary columns if missing
        for col, default_val in DEFAULT_ASSUMPTIONS.items():
            if col not in clean_df.columns:
                clean_df[col] = default_val

        # Also populate actual-name defaults if canonical was populated
        for canon_col, actual_col in CANONICAL_TO_ACTUAL_COLUMN_MAP.items():
            if canon_col in clean_df.columns and actual_col not in clean_df.columns:
                clean_df[actual_col] = clean_df[canon_col]

        # 7. Cash flow reconciliation without overwriting original net flow
        calc_net = clean_df[DEPOSITS_COL].astype(float) - clean_df[WITHDRAWALS_COL].astype(float)
        clean_df["validated_net_flow_usd"] = calc_net
        if NET_FLOW_COL not in clean_df.columns:
            clean_df[NET_FLOW_COL] = calc_net

        # 8. Reconcile repo sweep timing window
        hours = clean_df[TIMESTAMP_COL].dt.hour
        if REPO_SWEEP_OPEN_COL not in clean_df.columns:
            clean_df[REPO_SWEEP_OPEN_COL] = (hours == REPO_SWEEP_OPEN_HOUR).astype(int)
        else:
            clean_df[REPO_SWEEP_OPEN_COL] = clean_df[REPO_SWEEP_OPEN_COL].astype(int)

        if REPO_SWEEP_MATURE_COL not in clean_df.columns:
            clean_df[REPO_SWEEP_MATURE_COL] = (hours == REPO_SWEEP_MATURE_HOUR).astype(int)
        else:
            clean_df[REPO_SWEEP_MATURE_COL] = clean_df[REPO_SWEEP_MATURE_COL].astype(int)

        # 9. Market hours open (weekdays 14:00 to 21:00 UTC)
        if MARKET_HOURS_OPEN_COL not in clean_df.columns:
            is_weekday = clean_df[TIMESTAMP_COL].dt.dayofweek < 5
            is_market_hour = (hours >= 14) & (hours < 21)
            clean_df[MARKET_HOURS_OPEN_COL] = (is_weekday & is_market_hour).astype(int)

        return clean_df
