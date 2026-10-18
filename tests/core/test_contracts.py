"""Unit tests for Reserve Juggler canonical contracts and event publisher."""

from datetime import datetime, timezone, timedelta
import pytest
from reserve_juggler.contracts import (
    AssetPosition,
    AssetTier,
    CreditFacility,
    DataQualityStatus,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    ForecastPoint,
    InMemoryEventPublisher,
    LearningEvent,
    NoOpEventPublisher,
    Obligation,
    ObligationType,
    RiskSignal,
    TreasuryState,
    ensure_utc,
)


def test_ensure_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    utc = ensure_utc(naive)
    assert utc.tzinfo == timezone.utc
    assert utc.hour == 12

    already_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert ensure_utc(already_utc) == already_utc


def test_asset_position_and_liquidatable_value():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Tier 2 sovereign bond with 2% haircut
    pos = AssetPosition(
        asset_id="T2_UST_2Y",
        tier=AssetTier.TIER_2_BONDS,
        units=100_000.0,
        unit_price_usd=1.02,
        market_value_usd=102_000.0,
        liquidatable_value_usd=102_000.0 * (1.0 - 0.02),
        availability_timestamp=now,
        maturity_timestamp=now + timedelta(days=730),
        annual_yield_or_return=0.048,
        liquidation_haircut_rate=0.02,
        duration_years=1.9,
    )
    assert pos.market_value_usd == 102_000.0
    assert pos.liquidatable_value_usd == pytest.approx(99_960.0)
    assert pos.tier == AssetTier.TIER_2_BONDS

    # JSON serialization
    json_str = pos.model_dump_json()
    assert "T2_UST_2Y" in json_str
    reconstructed = AssetPosition.model_validate_json(json_str)
    assert reconstructed.asset_id == pos.asset_id
    assert reconstructed.liquidatable_value_usd == pos.liquidatable_value_usd


def test_treasury_state_computations():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    repo_pos = AssetPosition(
        asset_id="T1_REPO",
        tier=AssetTier.TIER_1_REPO,
        units=5_000_000.0,
        unit_price_usd=1.0,
        market_value_usd=5_000_000.0,
        liquidatable_value_usd=5_000_000.0 * (1.0 - 0.005), # 0.5% early break penalty
        annual_yield_or_return=0.043,
        early_exit_fee_rate=0.005,
        maturity_timestamp=now + timedelta(hours=20),
    )

    bond_pos = AssetPosition(
        asset_id="T2_BONDS",
        tier=AssetTier.TIER_2_BONDS,
        units=10_000_000.0,
        unit_price_usd=1.0,
        market_value_usd=10_000_000.0,
        liquidatable_value_usd=10_000_000.0 * (1.0 - 0.02),
        annual_yield_or_return=0.048,
        liquidation_haircut_rate=0.02,
        duration_years=1.9,
    )

    credit = CreditFacility(
        facility_id="EMERGENCY_LOC_1",
        is_available=True,
        committed_limit_usd=15_000_000.0,
        utilized_amount_usd=2_000_000.0,
        apr=0.075,
        fee_rate=0.0025,
    )

    obligation = Obligation(
        obligation_id="OBL_1",
        timestamp=now + timedelta(hours=2),
        amount_usd=4_000_000.0,
        obligation_type=ObligationType.MERCHANT_SETTLEMENT,
    )

    state = TreasuryState(
        timestamp=now,
        operational_cash_usd=6_000_000.0,
        asset_positions={"T1_REPO": repo_pos, "T2_BONDS": bond_pos},
        obligations=[obligation],
        credit_facility=credit,
        warnings=[],
    )

    assert state.operational_cash == 6_000_000.0
    assert state.total_investments_market_value_usd == 15_000_000.0
    assert state.total_balance_sheet_assets_usd == 21_000_000.0
    # Liquidatable = 6,000,000 + 4,975,000 + 9,800,000 = 20,775,000
    assert state.total_liquidatable_value_usd == pytest.approx(20_775_000.0)
    assert state.available_credit_usd == 13_000_000.0
    assert state.upcoming_obligations_usd == 4_000_000.0
    # Net liquidity = 20,775,000 + 13,000,000 - 4,000,000 = 29,775,000
    assert state.net_liquidity_usd == pytest.approx(29_775_000.0)


def test_forecast_bundle_and_points():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pt1 = ForecastPoint(
        as_of_timestamp=now,
        target_timestamp=now + timedelta(hours=1),
        horizon_hours=1,
        deposit_p10=18_000_000.0,
        deposit_p50=20_000_000.0,
        deposit_p90=22_000_000.0,
        withdrawal_p10=17_000_000.0,
        withdrawal_p50=19_500_000.0,
        withdrawal_p90=22_500_000.0,
        net_outflow_p50=-500_000.0,
        net_outflow_p95=2_500_000.0,
        net_outflow_p99=4_000_000.0,
        cumulative_net_outflow_p95=2_500_000.0,
        cumulative_net_outflow_p99=4_000_000.0,
        reliability_score=0.95,
        model_version="ensemble_v1.0",
    )

    bundle = ForecastBundle(
        forecast_timestamp=now,
        points=[pt1],
        scenario_references={"stress_shock": "severe_withdrawal"},
        data_quality_status=DataQualityStatus.OK,
        model_versions={"ridge": "1.0", "seasonal": "1.0"},
        evaluation_metadata={"loss": 0.042},
    )

    assert bundle.get_point(1) == pt1
    assert bundle.get_point(6) is None
    dumped = bundle.model_dump()
    assert dumped["data_quality_status"] == "OK"


def test_event_publishers():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    event = LearningEvent(
        event_id="EVT_001",
        event_type="test_event",
        source_module="tests",
        event_timestamp=now,
        recording_timestamp=now,
        payload={"message": "hello"},
    )

    # NoOp publisher shouldn't fail
    noop = NoOpEventPublisher()
    noop.publish(event)

    # InMemory publisher records and dispatches
    in_mem = InMemoryEventPublisher()
    dispatched = []
    in_mem.subscribe(lambda e: dispatched.append(e.event_id))

    in_mem.publish(event)
    assert len(in_mem.events) == 1
    assert dispatched == ["EVT_001"]
    assert len(in_mem.get_events_by_type("test_event")) == 1
    assert len(in_mem.get_events_by_type("unknown")) == 0
