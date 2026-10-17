"""Tests for forecast validation, uncertainty calibration, and Team 2 optimizer integration readiness."""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from reserve_juggler.contracts import (
    DataQualityStatus,
    ForecastBundle,
    ForecastPoint,
)
from reserve_juggler.data.feature_engineering import FeatureEngineer
from reserve_juggler.data.schema import DEPOSITS_COL, TIMESTAMP_COL, WITHDRAWALS_COL
from reserve_juggler.evaluation.backtest import RollingOriginBacktest
from reserve_juggler.evaluation.metrics import evaluate_forecast_suite, winkler_score
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.data import RealisticTreasuryDataGenerator


@pytest.fixture
def synthetic_treasury_df() -> pd.DataFrame:
    """Generate 200 hours of deterministic synthetic treasury data."""
    gen = RealisticTreasuryDataGenerator(seed=123)
    return gen.generate(hours=200)


def test_forecast_point_contract_backward_compatibility():
    """Verify that ForecastPoint can be serialized, deserialized, and exposes new fields with defaults."""
    now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    target = now + timedelta(hours=24)

    pt = ForecastPoint(
        as_of_timestamp=now,
        target_timestamp=target,
        horizon_hours=24,
        deposit_p10=5_000_000.0,
        deposit_p50=8_000_000.0,
        deposit_p90=12_000_000.0,
        withdrawal_p10=6_000_000.0,
        withdrawal_p50=9_000_000.0,
        withdrawal_p90=14_000_000.0,
        net_outflow_p50=1_000_000.0,
        net_outflow_p95=4_000_000.0,
        net_outflow_p99=6_000_000.0,
        reliability_score=0.92,
        model_version="test-v1",
        # New fields
        deposit_point_forecast=8_000_000.0,
        withdrawal_point_forecast=9_000_000.0,
        deposit_p25=6_500_000.0,
        deposit_p75=10_000_000.0,
        deposit_p025=4_000_000.0,
        deposit_p975=13_500_000.0,
        withdrawal_p25=7_500_000.0,
        withdrawal_p75=11_500_000.0,
        withdrawal_p025=5_000_000.0,
        withdrawal_p975=16_000_000.0,
        calibration_sample_size=96,
        training_cutoff_timestamp=now,
        reliability_tier="HIGH",
        reliability_rationale="Calibrated on historical residuals",
        is_fallback=False,
    )

    json_str = pt.model_dump_json()
    reconstructed = ForecastPoint.model_validate_json(json_str)

    assert reconstructed.horizon_hours == 24
    assert reconstructed.deposit_point_forecast == 8_000_000.0
    assert reconstructed.reliability_tier == "HIGH"
    assert reconstructed.is_fallback is False
    assert reconstructed.calibration_sample_size == 96


def test_forecast_bundle_warnings_and_fallback_status():
    """Verify ForecastBundle serialization with quality warnings and fallback flags."""
    now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    target = now + timedelta(hours=1)

    pt = ForecastPoint(
        as_of_timestamp=now,
        target_timestamp=target,
        horizon_hours=1,
        deposit_p10=1_000.0,
        deposit_p50=2_000.0,
        deposit_p90=3_000.0,
        withdrawal_p10=1_500.0,
        withdrawal_p50=2_500.0,
        withdrawal_p90=4_000.0,
        net_outflow_p50=500.0,
        net_outflow_p95=1_500.0,
        net_outflow_p99=2_000.0,
        model_version="test-v1",
    )

    bundle = ForecastBundle(
        forecast_timestamp=now,
        points=[pt],
        data_quality_status=DataQualityStatus.DEGRADED,
        data_quality_warnings=["Stale market feed: repo rate frozen for 12 hours."],
        is_fallback_bundle=True,
    )

    assert bundle.data_quality_status == DataQualityStatus.DEGRADED
    assert len(bundle.data_quality_warnings) == 1
    assert bundle.is_fallback_bundle is True

    # Check serialization round-trip
    b_json = bundle.model_dump_json()
    b_loaded = ForecastBundle.model_validate_json(b_json)
    assert b_loaded.is_fallback_bundle is True
    assert b_loaded.data_quality_warnings[0].startswith("Stale market feed")


def test_fallback_forecast_generation_policy(synthetic_treasury_df):
    """Verify that generate_fallback_bundle produces conservative, inflated withdrawals and deflated deposits."""
    svc = ForecastService()
    now = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)

    fallback = svc.generate_fallback_bundle(
        as_of_timestamp=now,
        horizons_hours=[1, 6, 24],
        reason="Upstream market data gateway timed out.",
    )

    assert fallback.is_fallback_bundle is True
    assert fallback.data_quality_status == DataQualityStatus.DEGRADED
    assert len(fallback.data_quality_warnings) == 1
    assert len(fallback.points) == 3

    for pt in fallback.points:
        assert pt.is_fallback is True
        assert pt.reliability_tier == "UNCALIBRATED"
        assert pt.reliability_score <= 0.50
        # Withdrawal point forecast is conservatively elevated above base
        assert pt.withdrawal_point_forecast > pt.deposit_point_forecast
        # Conservative monotonic ordering
        assert pt.withdrawal_p10 <= pt.withdrawal_p50 <= pt.withdrawal_p90


def test_chronological_train_test_separation_and_holdout(synthetic_treasury_df):
    """Verify that backtest respects expanding window, holdout cutoff, and identical model evaluation timestamps."""
    backtester = RollingOriginBacktest()

    initial_train = 60
    holdout = 30
    step = 12
    max_evals = 5

    report = backtester.run(
        df=synthetic_treasury_df,
        target_cols=[DEPOSITS_COL, WITHDRAWALS_COL],
        horizons=[1, 6],
        initial_train_size=initial_train,
        roll_step_hours=step,
        max_evaluations=max_evals,
        holdout_hours=holdout,
    )

    assert report.total_evaluations <= max_evals
    assert report.holdout_hours == holdout
    assert report.holdout_start_timestamp is not None

    # Check each horizon has identical model evaluation samples
    for h_res in report.horizon_results:
        assert h_res.sample_count == report.total_evaluations
        assert h_res.unique_target_timestamps_count == report.total_evaluations
        # All 4 models have valid non-negative MAE
        assert h_res.recent_baseline_mae >= 0.0
        assert h_res.seasonal_baseline_mae >= 0.0
        assert h_res.ridge_ml_mae >= 0.0
        assert h_res.or_ensemble_mae >= 0.0
        # Intervals are populated
        assert h_res.picp_80 is not None
        assert h_res.picp_50 is not None
        assert h_res.picp_95 is not None
        assert h_res.winkler_80 is not None


def test_winkler_score_properties():
    """Verify Winkler score penalizes misses and rewards sharpness."""
    y = np.array([100.0, 100.0, 100.0])
    # Perfect narrow interval
    score_tight = winkler_score(y, lower_bound=[90.0, 90.0, 90.0], upper_bound=[110.0, 110.0, 110.0], alpha=0.20)
    assert score_tight == 20.0  # width = 20, 0 misses

    # Wider interval
    score_wide = winkler_score(y, lower_bound=[50.0, 50.0, 50.0], upper_bound=[150.0, 150.0, 150.0], alpha=0.20)
    assert score_wide == 100.0  # width = 100, 0 misses

    # Missed interval (penalty applied)
    score_miss = winkler_score(y, lower_bound=[110.0, 110.0, 110.0], upper_bound=[120.0, 120.0, 120.0], alpha=0.20)
    # width=10, under=10, penalty = 10 + (2/0.2)*10 = 110
    assert score_miss == pytest.approx(110.0)
    assert score_miss > score_tight


def test_regime_evaluation_insufficient_evidence_handling(synthetic_treasury_df):
    """Verify that regime evaluation correctly reports Insufficient Evidence when N < 10."""
    backtester = RollingOriginBacktest()

    report = backtester.run(
        df=synthetic_treasury_df,
        target_cols=[DEPOSITS_COL],
        horizons=[1],
        initial_train_size=60,
        roll_step_hours=12,
        max_evaluations=5,  # Only 5 samples total!
    )

    h_res = report.horizon_results[0]
    assert h_res.sample_count == 5
    assert len(h_res.regime_metrics) > 0

    # Since N=5 < 10, all regimes should safely report Insufficient evidence
    for r_name, r_data in h_res.regime_metrics.items():
        assert r_data["status"] == "Insufficient evidence (N < 10)"
        assert r_data["sample_count"] <= 5
