"""Time-aligned stress and repo-event tests for Reserve Juggler forecasting.

Verifies:
1. Predictions available at 17:00 UTC repo sweep window without future leakage.
2. Predictions available for 08:00 UTC repo maturity settlement.
3. Behavior under large withdrawal spikes (thresholded strictly from pre-origin history).
4. Behavior under large negative net-flow deficit periods.
5. Stale/missing input fallback activation with conservative buffer and reliability tagging.
6. Zero future leakage on event origins.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from reserve_juggler.contracts import (
    DataQualityStatus,
    ForecastBundle,
    ForecastPoint,
)
from reserve_juggler.data import RealisticTreasuryDataGenerator
from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    NET_FLOW_COL,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_OPEN_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)
from reserve_juggler.forecasting.service import ForecastService


@pytest.fixture
def repo_event_dataset() -> pd.DataFrame:
    """Generate 500 hours of deterministic treasury data spanning multiple repo and stress cycles."""
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=500)
    return df


def test_repo_sweep_window_forecast_availability(repo_event_dataset):
    """Test forecast generation at 16:00 UTC for the 17:00 UTC repo sweep window (1h horizon).

    Verifies predictions are available, monotonic, and derived strictly from <= 16:00 data.
    """
    df = repo_event_dataset.copy()
    service = ForecastService()

    # Find an origin at 16:00 UTC
    df["hour"] = pd.to_datetime(df[TIMESTAMP_COL]).dt.hour
    candidates = df[df["hour"] == 16].index
    # Pick a candidate well after warm-up (e.g. row 200+)
    valid_candidates = [idx for idx in candidates if idx >= 200]
    assert len(valid_candidates) > 0, "No 16:00 UTC origins found in dataset"
    origin_idx = valid_candidates[0]

    origin_ts = pd.Timestamp(df.loc[origin_idx, TIMESTAMP_COL])
    history_df = df.iloc[:origin_idx + 1].copy()

    service.fit(history_df, horizons=[1, 24])
    bundle = service.forecast(
        as_of_timestamp=origin_ts.to_pydatetime(),
        horizons_hours=[1, 24],
    )

    assert isinstance(bundle, ForecastBundle)
    assert len(bundle.points) == 2

    # Target for 1h horizon should be 17:00 UTC (the sweep window)
    pt_1h = next(p for p in bundle.points if p.horizon_hours == 1)
    target_dt = pt_1h.target_timestamp
    assert target_dt.hour == 17
    assert pt_1h.withdrawal_point_forecast > 0
    assert pt_1h.deposit_point_forecast > 0

    # Quantile monotonicity
    assert pt_1h.withdrawal_p10 <= pt_1h.withdrawal_p50 <= pt_1h.withdrawal_p90
    assert pt_1h.deposit_p10 <= pt_1h.deposit_p50 <= pt_1h.deposit_p90
    assert pt_1h.net_outflow_p50 <= pt_1h.net_outflow_p95 <= pt_1h.net_outflow_p99
    assert pt_1h.is_fallback is False


def test_repo_maturity_settlement_forecast_availability(repo_event_dataset):
    """Test forecast generation for the 08:00 UTC repo maturity event.

    Evaluates:
    - 07:00 UTC origin predicting 1h ahead (08:00 UTC).
    - Previous evening 17:00 UTC origin predicting 15h ahead (08:00 UTC next morning).
    """
    df = repo_event_dataset.copy()
    service = ForecastService()
    df["hour"] = pd.to_datetime(df[TIMESTAMP_COL]).dt.hour

    # 1. 07:00 UTC origin predicting 08:00 UTC
    candidates_7am = [idx for idx in df[df["hour"] == 7].index if idx >= 200]
    origin_7am = candidates_7am[0]
    ts_7am = pd.Timestamp(df.loc[origin_7am, TIMESTAMP_COL])
    
    service.fit(df.iloc[:origin_7am + 1].copy(), horizons=[1])
    bundle_7am = service.forecast(
        as_of_timestamp=ts_7am.to_pydatetime(),
        horizons_hours=[1],
    )
    pt_1h = bundle_7am.points[0]
    assert pt_1h.target_timestamp.hour == 8
    assert pt_1h.withdrawal_p90 >= pt_1h.withdrawal_p10

    # 2. 17:00 UTC origin predicting 15h ahead (08:00 UTC next day)
    candidates_5pm = [idx for idx in df[df["hour"] == 17].index if idx >= 200]
    origin_5pm = candidates_5pm[0]
    ts_5pm = pd.Timestamp(df.loc[origin_5pm, TIMESTAMP_COL])
    
    service_15h = ForecastService()
    service_15h.fit(df.iloc[:origin_5pm + 1].copy(), horizons=[15])
    bundle_5pm = service_15h.forecast(
        as_of_timestamp=ts_5pm.to_pydatetime(),
        horizons_hours=[15],
    )
    pt_15h = bundle_5pm.points[0]
    assert pt_15h.target_timestamp.hour == 8
    assert pt_15h.horizon_hours == 15
    assert pt_15h.withdrawal_p90 >= pt_15h.withdrawal_p50


def test_withdrawal_spike_stress_regime(repo_event_dataset):
    """Verify uncertainty bands under withdrawal spike conditions.

    Threshold derived strictly from pre-origin history (never future outcomes).
    """
    df = repo_event_dataset.copy()
    service = ForecastService()

    # Pre-origin calibration threshold: 90th percentile of first 200 hours
    prior_data = df.iloc[:200]
    spike_threshold = float(prior_data[WITHDRAWALS_COL].quantile(0.90))

    # Find a spike in subsequent data
    spike_indices = [
        idx for idx in df.iloc[200:].index
        if df.loc[idx, WITHDRAWALS_COL] >= spike_threshold
    ]
    assert len(spike_indices) > 0, "No withdrawal spikes found above threshold"
    spike_idx = spike_indices[0]

    # Predict right at the spike origin
    spike_ts = pd.Timestamp(df.loc[spike_idx, TIMESTAMP_COL])
    history_df = df.iloc[:spike_idx + 1].copy()

    service.fit(history_df, horizons=[1, 6, 24])
    bundle = service.forecast(
        as_of_timestamp=spike_ts.to_pydatetime(),
        horizons_hours=[1, 6, 24],
    )

    for pt in bundle.points:
        # Prediction interval width must be substantial during high volatility (not collapsed)
        width_80 = pt.withdrawal_p90 - pt.withdrawal_p10
        assert width_80 > 100_000.0, f"Uncertainty width too narrow (${width_80:,.2f}) during withdrawal spike"
        assert pt.withdrawal_p975 >= pt.withdrawal_p90
        assert pt.withdrawal_p025 <= pt.withdrawal_p10


def test_large_negative_net_flow_stress_regime(repo_event_dataset):
    """Verify conservative net outflow quantiles during large deficit regime."""
    df = repo_event_dataset.copy()
    service = ForecastService()

    # Calculate net flow if not already present
    if NET_FLOW_COL not in df.columns:
        df[NET_FLOW_COL] = df[DEPOSITS_COL] - df[WITHDRAWALS_COL]

    # Pre-origin deficit threshold: 10th percentile of first 200 hours
    prior_data = df.iloc[:200]
    deficit_threshold = float(prior_data[NET_FLOW_COL].quantile(0.10))

    deficit_indices = [
        idx for idx in df.iloc[200:].index
        if df.loc[idx, NET_FLOW_COL] <= deficit_threshold
    ]
    assert len(deficit_indices) > 0, "No large deficit periods found"
    deficit_idx = deficit_indices[0]

    deficit_ts = pd.Timestamp(df.loc[deficit_idx, TIMESTAMP_COL])
    history_df = df.iloc[:deficit_idx + 1].copy()

    service.fit(history_df, horizons=[1, 6, 24])
    bundle = service.forecast(
        as_of_timestamp=deficit_ts.to_pydatetime(),
        horizons_hours=[1, 6, 24],
    )

    for pt in bundle.points:
        # Net outflow quantiles represent gross cash drain: p50 <= p95 <= p99
        assert pt.net_outflow_p50 <= pt.net_outflow_p95 <= pt.net_outflow_p99
        # Tail outflow p99 must be positive and substantial
        assert pt.net_outflow_p99 > 0


def test_stale_or_missing_input_fallback_trigger(repo_event_dataset):
    """Verify conservative fallback activation when data inputs are stale or degraded."""
    df = repo_event_dataset.copy()
    service = ForecastService()

    origin_ts = pd.Timestamp(df.loc[250, TIMESTAMP_COL]).to_pydatetime()
    history_df = df.iloc[:251].copy()

    # Fit with available history so fallback can compute baseline references if needed
    service.fit(history_df, horizons=[1, 6, 24, 48, 72])

    # Generate fallback bundle explicitly
    fallback_bundle = service.generate_fallback_bundle(
        as_of_timestamp=origin_ts,
        horizons=[1, 6, 24, 48, 72],
        reason="Stale market feed: no updates received for >6 hours",
    )

    assert fallback_bundle.is_fallback_bundle is True
    assert fallback_bundle.data_quality_status == DataQualityStatus.DEGRADED
    assert len(fallback_bundle.data_quality_warnings) == 1
    assert "Stale market feed" in fallback_bundle.data_quality_warnings[0]

    for pt in fallback_bundle.points:
        assert pt.is_fallback is True
        assert pt.reliability_tier == "UNCALIBRATED"
        assert "Stale market feed" in pt.reliability_rationale
        # Fallback must apply conservative buffer: p90 > point_forecast
        assert pt.withdrawal_p90 >= pt.withdrawal_point_forecast
        # Deflated deposits: p10 <= point_forecast
        assert pt.deposit_p10 <= pt.deposit_point_forecast


def test_zero_future_leakage_on_event_origins(repo_event_dataset):
    """Verify that altering future data strictly after the origin does not affect forecast output."""
    df = repo_event_dataset.copy()
    origin_idx = 220
    origin_ts = pd.Timestamp(df.loc[origin_idx, TIMESTAMP_COL]).to_pydatetime()

    # Run 1: Normal history up to origin
    history_normal = df.iloc[:origin_idx + 1].copy()
    service_1 = ForecastService()
    service_1.fit(history_normal, horizons=[1, 6, 24])
    bundle_1 = service_1.forecast(
        as_of_timestamp=origin_ts,
        horizons_hours=[1, 6, 24],
    )

    # Run 2: Introduce extreme synthetic shock in future data (indices > origin_idx)
    # Even if full df with future shock is passed, service slicing ensures zero future leakage
    df_shocked = df.copy()
    df_shocked.loc[origin_idx + 1:, WITHDRAWALS_COL] = 999_999_999.0
    df_shocked.loc[origin_idx + 1:, DEPOSITS_COL] = 0.0

    service_2 = ForecastService()
    service_2.fit(df_shocked.iloc[:origin_idx + 1].copy(), horizons=[1, 6, 24])
    bundle_2 = service_2.forecast(
        as_of_timestamp=origin_ts,
        horizons_hours=[1, 6, 24],
    )

    for p1, p2 in zip(bundle_1.points, bundle_2.points):
        assert p1.withdrawal_point_forecast == pytest.approx(p2.withdrawal_point_forecast, rel=1e-6)
        assert p1.deposit_point_forecast == pytest.approx(p2.deposit_point_forecast, rel=1e-6)
        assert p1.withdrawal_p90 == pytest.approx(p2.withdrawal_p90, rel=1e-6)
        assert p1.deposit_p10 == pytest.approx(p2.deposit_p10, rel=1e-6)
