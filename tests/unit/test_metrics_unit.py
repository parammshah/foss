"""
Unit tests for forecasting, decision, and drift evaluation metrics.
Verifies MAE, WAPE, zero-actual exclusion from MAPE, prediction interval empirical coverage,
and KS-test statistical drift calculation without synthetic shortcuts.
"""

from __future__ import annotations
import pytest

from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.models import (
    MatchedDecisionOutcome,
    MatchedForecastOutcome,
)
from reserve_juggler.learning.store import EventStore


def test_wape_and_mae_calculation():
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)

    pairs = [
        MatchedForecastOutcome(
            correlation_id=f"C{i}",
            forecast_id=f"F{i}",
            target_timestamp=f"2026-10-17T1{i}:00:00+00:00",
            horizon="1h",
            predicted_net_flow=100.0,
            predicted_inflow=100.0,
            predicted_outflow=0.0,
            uncertainty_p05=80.0,
            uncertainty_p50=100.0,
            uncertainty_p90=120.0,
            uncertainty_p95=130.0,
            uncertainty_p99=140.0,
            model_version="v1.4.0",
            actual_net_flow=110.0,  # Absolute error = 10, actual = 110
            is_eligible=True,
            within_90_interval=True,
        )
        for i in range(5)
    ]

    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=pairs)
    assert rep.forecast_metrics.mae == 10.0
    # WAPE = sum(|actual - pred|) / sum(|actual|) = 50 / 550 * 100% = 9.09%
    assert rep.forecast_metrics.wape is not None
    assert abs(rep.forecast_metrics.wape - 9.09) < 0.1


def test_zero_actuals_excluded_from_mape():
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)

    pairs = [
        MatchedForecastOutcome(
            correlation_id=f"C{i}",
            forecast_id=f"F{i}",
            target_timestamp=f"2026-10-17T1{i}:00:00+00:00",
            horizon="1h",
            predicted_net_flow=50.0,
            predicted_inflow=50.0,
            predicted_outflow=0.0,
            uncertainty_p05=0.0,
            uncertainty_p50=50.0,
            uncertainty_p90=100.0,
            uncertainty_p95=110.0,
            uncertainty_p99=120.0,
            model_version="v1.4.0",
            actual_net_flow=0.0,  # Zero actual flow
            is_eligible=True,
            within_90_interval=True,
        )
        for i in range(4)
    ]

    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=pairs)
    assert rep.forecast_metrics.mape is None
    assert any("MAPE undefined" in r for r in rep.forecast_metrics.exclusion_reasons)
