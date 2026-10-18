"""Unit tests for treasury ledger double-entry accounting, customer flows, and credit facility."""

from datetime import datetime, timezone, timedelta
import pytest
from reserve_juggler.contracts import (
    AssetPosition,
    AssetTier,
    CreditFacility,
    Obligation,
    ObligationType,
    InMemoryEventPublisher,
)
from reserve_juggler.ledger import (
    InitialTreasuryConfig,
    TreasuryLedger,
)


def test_initial_config_no_silent_defaults():
    # Opening cash must be explicitly provided
    config = InitialTreasuryConfig(opening_cash_usd=10_000_000.0)
    ledger = TreasuryLedger(config)
    now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    state = ledger.get_state(now)
    assert state.operational_cash == 10_000_000.0
    assert len(state.asset_positions) == 0


def test_hourly_customer_flows_and_reconciliation():
    config = InitialTreasuryConfig(opening_cash_usd=25_000_000.0)
    ledger = TreasuryLedger(config)
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Hour 1: +20M deposits, -18M withdrawals
    net1, shortfall1 = ledger.process_hourly_step(
        current_timestamp=t0,
        deposits_usd=20_000_000.0,
        withdrawals_usd=18_000_000.0,
        repo_sweep_mature=False,
    )
    assert net1 == 2_000_000.0
    assert shortfall1 == 0.0
    # Cash should be 25M + 2M = 27M
    assert ledger.cash_usd == 27_000_000.0
    assert ledger.cumulative_deposits_usd == 20_000_000.0
    assert ledger.cumulative_withdrawals_usd == 18_000_000.0


def test_obligations_and_emergency_credit_draw():
    credit = CreditFacility(
        facility_id="LOC_1",
        is_available=True,
        committed_limit_usd=10_000_000.0,
        utilized_amount_usd=0.0,
        apr=0.08,
    )
    config = InitialTreasuryConfig(
        opening_cash_usd=2_000_000.0,
        credit_facility=credit,
    )
    publisher = InMemoryEventPublisher()
    ledger = TreasuryLedger(config, event_publisher=publisher)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Add scheduled merchant settlement obligation of $5M
    ledger.add_obligation(
        Obligation(
            obligation_id="MERCHANT_01",
            timestamp=t0,
            amount_usd=5_000_000.0,
            obligation_type=ObligationType.MERCHANT_SETTLEMENT,
        )
    )

    # Customer withdrawal = $1M, total demand = $1M + $5M = $6M.
    # Cash is $2M. Deficit is $4M.
    # Should draw $4M from credit facility to fulfill demand!
    net, shortfall = ledger.process_hourly_step(
        current_timestamp=t0,
        deposits_usd=0.0,
        withdrawals_usd=1_000_000.0,
        repo_sweep_mature=False,
    )

    assert shortfall == 0.0
    assert ledger.cash_usd == 0.0
    assert ledger.credit_manager.facility.utilized_amount_usd == pytest.approx(4_000_000.0)
    assert ledger.credit_manager.available_capacity_usd == pytest.approx(6_000_000.0)
    # Obligation is settled
    assert ledger.obligations[0].is_settled is True


def test_unfunded_shortfall_detection():
    # Cash = $1M, no credit facility, withdrawal demand = $5M -> $4M shortfall
    config = InitialTreasuryConfig(opening_cash_usd=1_000_000.0)
    publisher = InMemoryEventPublisher()
    ledger = TreasuryLedger(config, event_publisher=publisher)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    net, shortfall = ledger.process_hourly_step(
        current_timestamp=t0,
        deposits_usd=0.0,
        withdrawals_usd=5_000_000.0,
        repo_sweep_mature=False,
    )

    assert shortfall == pytest.approx(4_000_000.0)
    assert ledger.cash_usd == 0.0
    assert ledger.cumulative_shortfall_usd == pytest.approx(4_000_000.0)
    assert len(publisher.get_events_by_type("liquidity_shortfall")) == 1
