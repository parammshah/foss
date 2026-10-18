"""Unit tests for baseline, ML, and forecast service pipelines."""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest
from reserve_juggler.contracts import InMemoryEventPublisher
from reserve_juggler.data import RealisticTreasuryDataGenerator, DEPOSITS_COL, WITHDRAWALS_COL
from reserve_juggler.forecasting import (
    ForecastService,
    RecentAverageBaseline,
    RidgeMLModel,
    TimeOfWeekSeasonalBaseline,
)


def test_baseline_models():
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=100)

    recent = RecentAverageBaseline(method="rolling_24h")
    recent.fit(df[DEPOSITS_COL])
    p1 = recent.predict(1)
    assert p1 > 0.0

    seasonal = TimeOfWeekSeasonalBaseline()
    seasonal.fit(df, DEPOSITS_COL)
    ts = pd.Timestamp("2026-01-05 12:00:00", tz="UTC")
    p_seasonal = seasonal.predict(ts)
    assert len(p_seasonal) == 1
    assert p_seasonal[0] > 0.0


def test_ridge_ml_model():
    # Simple linear data with noise
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "feat_1": rng.normal(10, 2, 100),
        "feat_2": rng.normal(5, 1, 100),
    })
    y = 2.0 * X["feat_1"] + 3.0 * X["feat_2"] + rng.normal(0, 0.5, 100)

    model = RidgeMLModel(alpha=1.0)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 100
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.90
    # Strict non-negativity
    assert np.all(preds >= 0.0)


def test_forecast_service_e2e():
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=300)

    publisher = InMemoryEventPublisher()
    service = ForecastService(event_publisher=publisher)
    service.fit(df, horizons=[1, 6, 24])

    now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    bundle = service.forecast(as_of_timestamp=now, horizons_hours=[1, 6, 24])

    assert len(bundle.points) == 3
    assert bundle.get_point(1) is not None
    assert bundle.get_point(6) is not None
    assert bundle.get_point(24) is not None

    p1 = bundle.get_point(1)
    # Check quantile monotonicity
    assert p1.deposit_p10 <= p1.deposit_p50 <= p1.deposit_p90
    assert p1.withdrawal_p10 <= p1.withdrawal_p50 <= p1.withdrawal_p90
    # Check net outflow consistency
    assert p1.net_outflow_p50 == pytest.approx(p1.withdrawal_p50 - p1.deposit_p50)
    assert p1.net_outflow_p50 <= p1.net_outflow_p95 <= p1.net_outflow_p99
    # Cumulative shock monotonicity
    assert p1.cumulative_net_outflow_p95 <= p1.cumulative_net_outflow_p99

    # Check published event
    events = publisher.get_events_by_type("forecast_generated")
    assert len(events) == 1
    assert events[0].payload["points_count"] == 3
