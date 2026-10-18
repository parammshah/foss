"""Unit tests for multi-hour simulation engine, actions, and policy execution."""

from datetime import datetime, timezone
import pytest
from reserve_juggler.contracts import InMemoryEventPublisher
from reserve_juggler.data import RealisticTreasuryDataGenerator
from reserve_juggler.ledger import InitialTreasuryConfig, TreasuryLedger
from reserve_juggler.simulation import ActionType, SimulationEngine, TreasuryAction


def test_simulation_engine_run_hold():
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=48)

    config = InitialTreasuryConfig(opening_cash_usd=50_000_000.0)
    publisher = InMemoryEventPublisher()
    ledger = TreasuryLedger(config, event_publisher=publisher)

    engine = SimulationEngine(
        ledger=ledger,
        timeline_df=df,
        event_publisher=publisher,
    )

    # Simple policy: HOLD every step
    result = engine.run(start_step=0, num_steps=24, policy_callback=lambda s, f: TreasuryAction(type=ActionType.HOLD))

    assert result.total_steps == 24
    assert len(result.step_records) == 24
    assert result.cumulative_deposits_usd > 0.0
    assert result.cumulative_withdrawals_usd > 0.0
    # No shortfall since opening cash is $50M
    assert result.cumulative_shortfall_usd == 0.0

    # Verify telemetry events emitted for every step
    events = publisher.get_events_by_type("simulation_step_completed")
    assert len(events) == 24


def test_simulation_with_repo_sweep_policy():
    """Policy sweeps $5M into repo at 17:00 when sweep window is open."""
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=48)

    config = InitialTreasuryConfig(opening_cash_usd=30_000_000.0)
    ledger = TreasuryLedger(config)
    engine = SimulationEngine(ledger=ledger, timeline_df=df)

    def sweep_policy(state, forecast):
        if state.timestamp.hour == 17:
            return TreasuryAction(type=ActionType.SWEEP_REPO, amount_usd=5_000_000.0)
        return TreasuryAction(type=ActionType.HOLD)

    result = engine.run(start_step=0, num_steps=36, policy_callback=sweep_policy)

    # Check that at step with hour=17, SWEEP_REPO was executed
    sweep_records = [r for r in result.step_records if r.action_taken and r.action_taken.type == ActionType.SWEEP_REPO]
    assert len(sweep_records) >= 1
    assert sweep_records[0].outcome.execution_status == "SUCCESS"
