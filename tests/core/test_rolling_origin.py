"""Unit tests for chronological rolling-origin backtesting."""

import pytest
from reserve_juggler.contracts import InMemoryEventPublisher
from reserve_juggler.data import RealisticTreasuryDataGenerator, DEPOSITS_COL
from reserve_juggler.evaluation.backtest import RollingOriginBacktest


def test_rolling_origin_backtest_execution():
    gen = RealisticTreasuryDataGenerator(seed=42)
    # Generate 150 hours for fast, focused backtest unit testing
    df = gen.generate(hours=150)

    publisher = InMemoryEventPublisher()
    backtester = RollingOriginBacktest(event_publisher=publisher)

    report = backtester.run(
        df=df,
        target_cols=[DEPOSITS_COL],
        horizons=[1, 6],
        initial_train_size=80,
        roll_step_hours=12,
        max_evaluations=4,
    )

    assert report.total_evaluations == 4
    assert len(report.horizon_results) == 2  # 2 horizons for DEPOSITS_COL

    for h_res in report.horizon_results:
        assert h_res.target_variable == DEPOSITS_COL
        assert h_res.sample_count == 4
        # All models produce valid non-negative MAE and WAPE
        assert h_res.recent_baseline_mae > 0.0
        assert h_res.seasonal_baseline_mae > 0.0
        assert h_res.ridge_ml_mae > 0.0
        assert h_res.or_ensemble_mae > 0.0
        assert 0.0 <= h_res.or_ensemble_wape <= 1.0

        # OR ensemble weights satisfy non-negativity and sum to 1
        weights = list(h_res.or_ensemble_weights.values())
        assert all(w >= 0.0 for w in weights)
        assert pytest.approx(sum(weights), abs=1e-4) == 1.0

    # Summary table renders properly
    summary = report.summary()
    assert "CHRONOLOGICAL ROLLING-ORIGIN BACKTEST RESULTS" in summary
    assert "OR Ensemble MAE" in summary

    # Telemetry events emitted
    events = publisher.get_events_by_type("forecast_evaluated")
    assert len(events) == 4 * 2  # 4 evaluations * 2 horizons
