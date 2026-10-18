"""Unit tests for feature engineering and mathematical verification of zero lookahead leakage."""

import numpy as np
import pandas as pd
import pytest
from reserve_juggler.data import FeatureEngineer, RealisticTreasuryDataGenerator, DEPOSITS_COL, WITHDRAWALS_COL


def test_feature_engineering_columns():
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=200)

    engineer = FeatureEngineer(lags=[1, 2, 6, 24], rolling_windows=[6, 24])
    featured = engineer.create_features(df)

    # Check temporal features exist
    for col in ["hour_of_day", "day_of_week", "sin_hour", "cos_hour", "is_weekend"]:
        assert col in featured.columns

    # Check lags exist
    for lag in [1, 2, 6, 24]:
        assert f"{DEPOSITS_COL}_lag_{lag}" in featured.columns
        assert f"{WITHDRAWALS_COL}_lag_{lag}" in featured.columns

    # Check rolling stats exist
    for win in [6, 24]:
        assert f"{DEPOSITS_COL}_roll_mean_{win}h" in featured.columns
        assert f"{DEPOSITS_COL}_roll_std_{win}h" in featured.columns


def test_zero_lookahead_leakage_assertion():
    """Verify that a massive perturbation in target at row T has ZERO impact on predictors for rows <= T."""
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=300)

    # Test perturbation at multiple index points (e.g., 50, 100, 150)
    for t in [50, 100, 150]:
        no_leakage = FeatureEngineer.assert_no_lookahead_leakage(
            original_df=df,
            featured_df=None,
            perturbed_index=t,
            target_col=DEPOSITS_COL,
        )
        assert no_leakage, f"Lookahead leakage detected when perturbing index {t}!"


def test_future_shock_zero_leakage_regression():
    """Verify that extreme future shocks occurring after the decision timestamp have ZERO effect on earlier forecasts."""
    from reserve_juggler.forecasting.service import ForecastService
    from reserve_juggler.data.schema import TIMESTAMP_COL

    gen = RealisticTreasuryDataGenerator(seed=123)
    df = gen.generate(hours=200)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    # Decision timestamp at row 100
    cutoff_idx = 100
    as_of_time = pd.Timestamp(df.iloc[cutoff_idx][TIMESTAMP_COL]).to_pydatetime()

    # Time-correct training slice
    train_slice = df[df[TIMESTAMP_COL] <= as_of_time].copy()
    assert train_slice[TIMESTAMP_COL].max() <= as_of_time, "Training slice contains future timestamps!"

    service_baseline = ForecastService()
    service_baseline.fit(train_slice, horizons=[1, 6, 24, 48, 72], as_of_timestamp=as_of_time)
    bundle_baseline = service_baseline.forecast(as_of_time)

    # Introduce extreme future shocks (> 100x values) strictly in future observations (rows > 100)
    future_shocked_df = df.copy()
    future_shocked_df.loc[cutoff_idx + 1:, WITHDRAWALS_COL] = 999_999_999.0
    future_shocked_df.loc[cutoff_idx + 1:, DEPOSITS_COL] = 0.0

    # Under strict temporal protocol, fit on data sliced up to as_of_time
    train_slice_shocked = future_shocked_df[future_shocked_df[TIMESTAMP_COL] <= as_of_time].copy()
    assert train_slice_shocked[TIMESTAMP_COL].max() <= as_of_time, "Shocked training slice leaked future observations!"

    service_shocked = ForecastService()
    service_shocked.fit(train_slice_shocked, horizons=[1, 6, 24, 48, 72], as_of_timestamp=as_of_time)
    bundle_shocked = service_shocked.forecast(as_of_time)

    # Assert that all forecasts across all horizons are bit-for-bit invariant to future shocks
    for h in [1, 6, 24, 48, 72]:
        pt_base = bundle_baseline.get_point(h)
        pt_shock = bundle_shocked.get_point(h)
        assert pt_base is not None and pt_shock is not None
        assert pt_base.deposit_point_forecast == pytest.approx(pt_shock.deposit_point_forecast, rel=1e-9)
        assert pt_base.withdrawal_point_forecast == pytest.approx(pt_shock.withdrawal_point_forecast, rel=1e-9)
        assert pt_base.withdrawal_p90 == pytest.approx(pt_shock.withdrawal_p90, rel=1e-9)
        assert pt_base.deposit_p10 == pytest.approx(pt_shock.deposit_p10, rel=1e-9)
        assert pt_base.net_outflow_p95 == pytest.approx(pt_shock.net_outflow_p95, rel=1e-9)


def test_temporal_cutoff_enforcement_in_orchestrator():
    """Verify that DecisionOrchestrator rejects historical data containing timestamps beyond as_of_timestamp."""
    from reserve_juggler.integration.orchestrator import DecisionOrchestrator
    from reserve_juggler.integration.contracts import TreasuryState
    from reserve_juggler.data.schema import TIMESTAMP_COL

    gen = RealisticTreasuryDataGenerator(seed=99)
    df = gen.generate(hours=100)

    # Set as_of_time to hour 50, but pass all 100 rows
    as_of_time = pd.Timestamp(df.iloc[50][TIMESTAMP_COL]).to_pydatetime()
    state = TreasuryState(
        timestamp=as_of_time,
        operational_cash_usd=50_000_000.0,
    )
    orchestrator = DecisionOrchestrator()

    with pytest.raises(AssertionError, match="Temporal leakage detected"):
        orchestrator.run_cycle(
            current_state=state,
            as_of_timestamp=as_of_time,
            historical_df=df,  # Has 100 rows, but as_of_time is row 50!
        )

