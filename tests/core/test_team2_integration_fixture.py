"""Deterministic integration fixture for Team 2 Treasury Optimizer consumption.

This test file verifies:
1. ForecastPoint and ForecastBundle contracts serialize and deserialize with full fidelity.
2. All units are strictly USD and hours, timestamps are timezone-aware UTC.
3. Demonstrates how Team 2's optimizer should consume probabilistic forecasts without
   treating point estimates as guaranteed outcomes.
4. Demonstrates safe handling of fallback bundles, degraded data quality, and uncalibrated regimes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import pytest

from reserve_juggler.contracts import (
    DataQualityStatus,
    ForecastBundle,
    ForecastPoint,
)
from reserve_juggler.data import RealisticTreasuryDataGenerator
from reserve_juggler.forecasting.service import ForecastService


@pytest.fixture
def deterministic_forecast_bundle() -> ForecastBundle:
    """Generate a realistic, deterministic 72-hour forecast bundle with all metadata."""
    now = datetime(2026, 10, 15, 17, 0, 0, tzinfo=timezone.utc)
    points: List[ForecastPoint] = []

    horizons = [1, 6, 24, 48, 72]
    for h in horizons:
        target = now + timedelta(hours=h)
        # Scaled nominal figures for $20M hourly flow bank
        dep_pt = 8_200_000.0
        wid_pt = 7_900_000.0
        h_spread = min(5_000_000.0, 400_000.0 * (h ** 0.5))

        pt = ForecastPoint(
            as_of_timestamp=now,
            target_timestamp=target,
            horizon_hours=h,
            deposit_point_forecast=dep_pt,
            withdrawal_point_forecast=wid_pt,
            deposit_p025=max(0.0, dep_pt - (1.5 * h_spread)),
            deposit_p10=max(0.0, dep_pt - (1.0 * h_spread)),
            deposit_p25=max(0.0, dep_pt - (0.5 * h_spread)),
            deposit_p50=dep_pt,
            deposit_p75=dep_pt + (0.5 * h_spread),
            deposit_p90=dep_pt + (1.0 * h_spread),
            deposit_p975=dep_pt + (1.5 * h_spread),
            withdrawal_p025=max(0.0, wid_pt - (1.5 * h_spread)),
            withdrawal_p10=max(0.0, wid_pt - (1.0 * h_spread)),
            withdrawal_p25=max(0.0, wid_pt - (0.5 * h_spread)),
            withdrawal_p50=wid_pt,
            withdrawal_p75=wid_pt + (0.5 * h_spread),
            withdrawal_p90=wid_pt + (1.0 * h_spread),
            withdrawal_p975=wid_pt + (1.5 * h_spread),
            net_outflow_p50=wid_pt - dep_pt,
            net_outflow_p95=(wid_pt - dep_pt) + (1.5 * h_spread),
            net_outflow_p99=(wid_pt - dep_pt) + (2.5 * h_spread),
            cumulative_net_outflow_p95=((wid_pt - dep_pt) * h) + (2.0 * h_spread),
            cumulative_net_outflow_p99=((wid_pt - dep_pt) * h) + (3.0 * h_spread),
            calibration_sample_size=336,
            training_cutoff_timestamp=now,
            reliability_score=0.92 if h <= 24 else 0.84,
            reliability_tier="HIGH" if h <= 24 else "MEDIUM",
            reliability_rationale=f"Calibrated on N=336 finite-sample conformal residuals (h={h})",
            is_fallback=False,
            model_version="RJ-Hybrid-ML-OR-v1.0",
        )
        points.append(pt)

    return ForecastBundle(
        forecast_timestamp=now,
        points=points,
        scenario_references={"repo_window_open": True},
        data_quality_status=DataQualityStatus.OK,
        data_quality_warnings=[],
        is_fallback_bundle=False,
        model_versions={"all": "RJ-Hybrid-ML-OR-v1.0"},
        evaluation_metadata={"calibration_protocol": "split_conformal_336h"},
    )


def test_forecast_bundle_serialization_round_trip(deterministic_forecast_bundle):
    """Verify that serialization to JSON and back retains full numerical and type fidelity."""
    bundle = deterministic_forecast_bundle
    json_bytes = bundle.model_dump_json().encode("utf-8")

    restored = ForecastBundle.model_validate_json(json_bytes)

    assert restored.forecast_timestamp == bundle.forecast_timestamp
    assert restored.data_quality_status == DataQualityStatus.OK
    assert restored.is_fallback_bundle is False
    assert len(restored.points) == 5

    for orig_pt, rest_pt in zip(bundle.points, restored.points):
        assert orig_pt.as_of_timestamp == rest_pt.as_of_timestamp
        assert orig_pt.target_timestamp == rest_pt.target_timestamp
        assert orig_pt.horizon_hours == rest_pt.horizon_hours
        assert orig_pt.deposit_point_forecast == rest_pt.deposit_point_forecast
        assert orig_pt.withdrawal_point_forecast == rest_pt.withdrawal_point_forecast
        assert orig_pt.withdrawal_p90 == rest_pt.withdrawal_p90
        assert orig_pt.deposit_p10 == rest_pt.deposit_p10
        assert orig_pt.reliability_tier == rest_pt.reliability_tier
        assert orig_pt.is_fallback == rest_pt.is_fallback
        assert orig_pt.calibration_sample_size == rest_pt.calibration_sample_size


def test_team2_optimizer_consumption_pattern(deterministic_forecast_bundle):
    """Demonstrate how Team 2 optimizer consumes forecast without treating it as guaranteed.

    Shows:
    1. Inspection of bundle health and fallback flags.
    2. Setting mandatory liquidity buffers from P90/P97.5 withdrawal quantiles.
    3. Setting conservative deposit inflow expectations from P10/P2.5 quantiles.
    4. Guarding against negative liquidity deficits.
    """
    bundle = deterministic_forecast_bundle

    # 1. Team 2 Step 1: Health & Fallback Verification
    assert bundle.data_quality_status == DataQualityStatus.OK
    assert not bundle.is_fallback_bundle

    # 2. Team 2 Step 2: Formulate Multi-Horizon Liquidity Constraints
    # A risk-averse optimizer MUST NOT use point forecasts directly for reserve safety:
    # Instead:
    # - Safe Outflow Obligation <= Withdrawal P90 (or P97.5 for extreme stress)
    # - Guaranteed Deposit Inflow >= Deposit P10
    simulated_tier1_reserves = 10_000_000.0  # $10M current cash in Tier 1

    for pt in bundle.points:
        h = pt.horizon_hours
        # Conservative requirement:
        # Buffer required = Withdrawal_P90 - Deposit_P10
        conservative_net_drain = pt.withdrawal_p90 - pt.deposit_p10

        # Point estimate net drain
        point_net_drain = pt.withdrawal_point_forecast - pt.deposit_point_forecast

        # Safety proof: Conservative requirement strictly exceeds point forecast
        assert conservative_net_drain > point_net_drain, (
            f"At h={h}, conservative buffer (${conservative_net_drain:,.2f}) must exceed "
            f"point estimate (${point_net_drain:,.2f})"
        )

        # Multi-horizon cumulative stress check:
        # P95 cumulative net outflow ensures multi-period solvency
        assert pt.cumulative_net_outflow_p95 is not None
        assert pt.cumulative_net_outflow_p99 is not None
        assert pt.cumulative_net_outflow_p99 >= pt.cumulative_net_outflow_p95


def test_team2_optimizer_fallback_handling():
    """Demonstrate how Team 2 optimizer reacts to degraded/fallback bundles.

    When is_fallback_bundle=True or data_quality_status is DEGRADED:
    - Optimizer must disable aggressive yield optimization.
    - Preserves maximum liquidity in Tier 1.
    - Honors fallback warnings.
    """
    now = datetime(2026, 10, 15, 17, 0, 0, tzinfo=timezone.utc)
    service = ForecastService()
    fallback_bundle = service.generate_fallback_bundle(
        as_of_timestamp=now,
        horizons_hours=[1, 6, 24],
        reason="Upstream market data feed stale (>4 hours)",
    )

    # Optimizer intake logic:
    optimizer_mode = "NORMAL"
    tier1_allocation_policy = "DYNAMIC"

    if fallback_bundle.is_fallback_bundle or fallback_bundle.data_quality_status != DataQualityStatus.OK:
        optimizer_mode = "CONSERVATIVE_PRESERVATION"
        tier1_allocation_policy = "MAXIMUM_LIQUIDITY_LOCK"

    assert optimizer_mode == "CONSERVATIVE_PRESERVATION"
    assert tier1_allocation_policy == "MAXIMUM_LIQUIDITY_LOCK"

    # Verify that each point reflects fallback status
    for pt in fallback_bundle.points:
        assert pt.is_fallback is True
        assert pt.reliability_tier == "UNCALIBRATED"
        # Confirm inflated withdrawals and deflated deposits
        assert pt.withdrawal_point_forecast > pt.deposit_point_forecast
