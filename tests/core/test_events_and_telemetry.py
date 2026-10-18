"""Unit tests for injectable event publisher, telemetry events, and graceful standalone operation."""

from datetime import datetime, timezone
import pytest
from reserve_juggler.contracts import (
    InMemoryEventPublisher,
    LearningEvent,
    NoOpEventPublisher,
)
from reserve_juggler.data import DatasetValidator, RealisticTreasuryDataGenerator
from reserve_juggler.forecasting import ForecastService
from reserve_juggler.ledger import InitialTreasuryConfig, TreasuryLedger
from reserve_juggler.simulation import ActionType, SimulationEngine, TreasuryAction


def test_graceful_operation_with_no_event_sink():
    """Verify all modules run cleanly with default NoOp publisher (no event sink configured)."""
    gen = RealisticTreasuryDataGenerator(seed=99)
    df = gen.generate(hours=48)

    # 1. Validator without publisher
    validator = DatasetValidator(event_publisher=None)
    report = validator.validate(df)
    assert report.is_valid

    # 2. Ledger without publisher
    config = InitialTreasuryConfig(opening_cash_usd=10_000_000.0)
    ledger = TreasuryLedger(config, event_publisher=None)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    net, shortfall = ledger.process_hourly_step(now, 1_000_000.0, 500_000.0, repo_sweep_mature=False)
    assert shortfall == 0.0

    # 3. ForecastService without publisher
    service = ForecastService(event_publisher=None)
    service.fit(df.iloc[:36], horizons=[1])
    bundle = service.forecast(now, horizons_hours=[1])
    assert len(bundle.points) == 1

    # 4. SimulationEngine without publisher
    engine = SimulationEngine(ledger=ledger, timeline_df=df, event_publisher=None)
    result = engine.run(start_step=0, num_steps=5)
    assert result.total_steps == 5


def test_system_wide_telemetry_event_capture():
    """Verify that an attached event sink captures all domain events across the lifecycle."""
    publisher = InMemoryEventPublisher()
    captured_types = []
    publisher.subscribe(lambda e: captured_types.append(e.event_type))

    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=48)

    # Trigger data validation event
    validator = DatasetValidator(event_publisher=publisher)
    validator.validate(df)
    assert "dataset_validated" in captured_types

    # Trigger forecast generation event
    service = ForecastService(event_publisher=publisher)
    service.fit(df.iloc[:36], horizons=[1])
    now = datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
    service.forecast(now, horizons_hours=[1])
    assert "forecast_generated" in captured_types

    # Trigger simulation step events
    config = InitialTreasuryConfig(opening_cash_usd=10_000_000.0)
    ledger = TreasuryLedger(config, event_publisher=publisher)
    engine = SimulationEngine(ledger=ledger, timeline_df=df, event_publisher=publisher)
    engine.run(start_step=0, num_steps=3)
    assert "simulation_step_completed" in captured_types

    # Verify event structure conforms to Team 3 contract
    all_events = publisher.events
    assert len(all_events) >= 5
    for evt in all_events:
        assert evt.event_id is not None
        assert evt.event_timestamp.tzinfo == timezone.utc
        assert evt.recording_timestamp.tzinfo == timezone.utc
        assert isinstance(evt.payload, dict)
        assert evt.schema_version == "1.0"
