"""
Reserve Juggler — 15 Mandatory Financial Safety & Adversarial Regression Tests.
Automates all cases from Section 4 of the Master Post-Integration Audit.
"""

from __future__ import annotations
import tempfile
from datetime import datetime, timezone, timedelta
import pytest

from reserve_juggler.contracts import (
    ApprovalStatus,
    TreasuryState,
    AssetPosition,
    AssetTier,
    Obligation,
    CreditFacility,
    ForecastBundle,
    ForecastPoint,
    GateStatus,
    RiskSignal,
    EventCategory,
    LearningEvent,
)
from reserve_juggler.events.models import RawEventRecord
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.integration.orchestrator import (
    DecisionOrchestrator,
    IndependentConstraintValidator,
)
from reserve_juggler.decision.models import StandardDecisionRecord
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.risk.liquidity_shield import evaluate_hard_safety_gates, assess_liquidity_reserve
from reserve_juggler.risk.capital_firewall import calculate_holding_time_gate
from reserve_juggler.optimization.optimizer import calculate_bridge_financing_cost
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value
from reserve_juggler.learning.service import LearningService
from reserve_juggler.learning.sqlite_store import SqliteEventStore


@pytest.fixture
def base_now() -> datetime:
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def make_record(dec_id: str, action: str, asset: str, amt: float, now: datetime) -> StandardDecisionRecord:
    return StandardDecisionRecord(
        decision_id=dec_id,
        timestamp=now,
        selected_action={"action_type": action, "target_asset": asset, "amount": amt},
        alternatives=[],
        score={},
        reserve={},
        eligible_surplus=0.0,
        cost_breakdown={},
        hard_gate_results={},
        reason_codes=[],
    )


def test_adv_01_customer_obligations_exceed_available_cash(base_now: datetime):
    """Case 1: Customer obligations exceed available immediate liquidity -> Fail gate, report shortfall."""
    val = IndependentConstraintValidator()
    st = TreasuryState(
        operational_cash_usd=4_000_000.0,
        upcoming_obligations_usd=10_000_000.0,
        available_credit_usd=20_000_000.0,
        as_of_timestamp=base_now,
    )
    rec = make_record("DEC-01", "KEEP_CASH", "CASH", 4_000_000.0, base_now)
    res = val.validate_proposal(st, rec)
    assert not res.passed
    assert res.shortfall_amount_usd == 6_000_000.0


def test_adv_02_withdrawal_forecast_spike_stress_lock(base_now: datetime):
    """Case 2: Withdrawal forecast spike -> Freezes discretionary investments."""
    val = IndependentConstraintValidator()
    bundle = ForecastBundle(
        points=[
            ForecastPoint(horizon_hours=1, predicted_deposits=5_000_000.0, predicted_withdrawals=25_000_000.0, uncertainty_p95=22_000_000.0),
        ],
        quality_status="DEGRADED",
        is_fallback_bundle=True,
    )
    st = TreasuryState(operational_cash_usd=30_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=base_now)
    rec = make_record("DEC-02", "BUY_BOND", "SOVEREIGN_BONDS", 5_000_000.0, base_now)
    res = val.validate_proposal(st, rec, forecast_bundle=bundle, stress_regime="ELEVATED_WITHDRAWAL_RISK")
    assert not res.passed
    assert any("Discretionary asset purchase prohibited" in r for r in res.rejection_reasons)


def test_adv_03_stale_or_missing_forecast_uses_fallback(base_now: datetime):
    """Case 3: Unfit / missing forecast inputs safely return conservative fallback."""
    fs = ForecastService()
    bundle = fs.forecast(base_now)
    assert bundle.is_fallback_bundle
    assert bundle.quality_status in ("DEGRADED", "FALLBACK")
    assert len(bundle.points) == 5


def test_adv_04_investment_breaching_reserve_floor_rejected(base_now: datetime):
    """Case 4: Investment leaving cash below reserve floor is rejected."""
    val = IndependentConstraintValidator()
    st = TreasuryState(operational_cash_usd=20_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=base_now)
    rec = make_record("DEC-04", "BUY_BOND", "SOVEREIGN_BONDS", 18_000_000.0, base_now)
    res = val.validate_proposal(st, rec)
    assert not res.passed


def test_adv_05_repo_maturity_settlement_timing_gate(base_now: datetime):
    """Case 5: Repo lockup conflicting with intermediate obligation deadline is rejected."""
    ob = Obligation(due_timestamp=base_now + timedelta(hours=4), amount_usd=3_000_000.0, is_mandatory=True)
    st = TreasuryState(operational_cash_usd=4_000_000.0, obligations=[ob], as_of_timestamp=base_now)
    fs = ForecastService()
    fb = fs.forecast(base_now)
    reserve = assess_liquidity_reserve(st, fb)
    gate_eval = evaluate_hard_safety_gates(st, reserve, proposed_investment_amount=2_500_000.0, proposed_lockup_hours=24.0)
    assert not gate_eval.passed


def test_adv_06_early_exit_vs_borrowing_cost_units():
    """Case 6: Early exit fee vs borrowing cost in consistent units."""
    bridge_cost = calculate_bridge_financing_cost(principal=10_000_000.0, apr=0.12, duration_hours=2.0)
    early_exit_fee = 10_000_000.0 * 0.005
    assert bridge_cost < early_exit_fee
    assert bridge_cost > 0.0


def test_adv_07_holding_time_and_equity_screening():
    """Case 7: Bond breakeven formula accepted, equity simplistic formula rejected."""
    bond = AssetPosition(asset_id="BOND_2Y", tier=AssetTier.TIER_2_BONDS, market_value_usd=5_000_000.0, annual_yield=0.048, haircut_rate=0.02)
    equity = AssetPosition(asset_id="STOCK", tier=AssetTier.TIER_3_EQUITIES, market_value_usd=5_000_000.0, annual_yield=0.08, haircut_rate=0.15)
    g_bond = calculate_holding_time_gate(bond, expected_holding_days=180.0)
    g_eq = calculate_holding_time_gate(equity, expected_holding_days=180.0)
    assert g_bond.passed
    assert not g_eq.passed


def test_adv_08_credit_line_capacity_enforcement(base_now: datetime):
    """Case 8: Credit draw exceeding committed facility is rejected."""
    val = IndependentConstraintValidator()
    st = TreasuryState(
        operational_cash_usd=10_000_000.0,
        upcoming_obligations_usd=5_000_000.0,
        available_credit_usd=3_000_000.0,
        as_of_timestamp=base_now,
    )
    rec = make_record("DEC-08", "DRAW_CREDIT", "CREDIT_FACILITY", 5_000_000.0, base_now)
    res = val.validate_proposal(st, rec)
    assert not res.passed
    assert any("exceeds committed facility limit" in r for r in res.rejection_reasons)


def test_adv_09_missing_tax_assumptions_labeled_provisional(base_now: datetime):
    """Case 9: Missing tax profile produces provisional ranking with audit warnings and requires human approval in orchestrator."""
    # 1. Engine level: gross yield returned, not synthetic brackets
    tax_res = evaluate_tax_adjusted_value(gross_revenue=100_000.0, instrument_id="TOKEN", tax_profile=None)
    assert tax_res.ranking_is_provisional
    assert tax_res.has_missing_classification
    assert tax_res.after_tax_net_value == 100_000.0  # Gross yield, no synthetic tax subtracted
    assert any("NO_TAX_PROFILE" in w for w in tax_res.warnings)

    # 2. Orchestration level: missing profile triggers provisional status and requires human sign-off
    orch = DecisionOrchestrator()
    st = TreasuryState(operational_cash_usd=25_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=base_now)
    res = orch.run_cycle(current_state=st, as_of_timestamp=base_now, tax_profile=None, auto_simulate_execution=True)
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res.simulated_outcome is None  # Blocked from auto-execution
    assert res.decision_record.metadata.get("ranking_is_provisional") is True
    assert res.decision_record.metadata.get("tax_profile_status") == "UNCONFIGURED_GROSS_ONLY"
    assert res.explanation is not None
    assert any("Tax Profile" in d for d in res.explanation.disclosed_missing_information)


def test_adv_10_news_deduplication_and_clustering(base_now: datetime):
    """Case 10: Duplicate news articles are clustered without duplicating risk."""
    sentinel = MacroPanicSentinel(use_synthetic_mode=False)
    e1 = RawEventRecord(event_id="E1", source="WIRE", title="Liquidity Freeze at Bank", summary="Liquidity freeze details", publication_time=base_now.isoformat(), ingestion_time=base_now.isoformat(), category=EventCategory.BANKING_STRESS.value, affected_markets=["REPO"], source_credibility=0.95)
    e2 = RawEventRecord(event_id="E2", source="WIRE", title="Liquidity Freeze at Bank", summary="Liquidity freeze details", publication_time=base_now.isoformat(), ingestion_time=base_now.isoformat(), category=EventCategory.BANKING_STRESS.value, affected_markets=["REPO"], source_credibility=0.95)
    clusters = sentinel.deduplicator.cluster_events([e1, e2])
    assert len(clusters) == 1
    assert len(clusters[0].duplicates) == 1
    assert clusters[0].reporting_count == 2


def test_adv_11_credit_draw_requires_human_approval(base_now: datetime):
    """Case 11: Credit facility draws mandate human approval gate."""
    val = IndependentConstraintValidator()
    st = TreasuryState(operational_cash_usd=25_000_000.0, upcoming_obligations_usd=5_000_000.0, available_credit_usd=50_000_000.0, as_of_timestamp=base_now)
    rec = make_record("DEC-11", "DRAW_CREDIT", "CREDIT_FACILITY", 10_000_000.0, base_now)
    res = val.validate_proposal(st, rec)
    assert res.passed
    assert res.requires_human_approval


def test_adv_12_infeasible_allocation_rejected_by_validator(base_now: datetime):
    """Case 12: Infeasible stock purchase with cash deficit rejected."""
    val = IndependentConstraintValidator()
    st = TreasuryState(operational_cash_usd=10_000_000.0, upcoming_obligations_usd=20_000_000.0, as_of_timestamp=base_now)
    rec = make_record("DEC-12", "BUY_STOCK", "EQUITIES", 5_000_000.0, base_now)
    res = val.validate_proposal(st, rec)
    assert not res.passed


def test_adv_13_idempotent_event_recording():
    """Case 13: Duplicate event recordings are ignored idempotently."""
    ls = LearningService()
    evt = LearningEvent(event_id="EVT-ADV-13", versioned_event_type="AUDIT", module="ORCHESTRATOR", payload={})
    init_cnt = len(ls.store.get_by_module("ORCHESTRATOR"))
    ls.store.record(evt, ignore_duplicates=True)
    cnt1 = len(ls.store.get_by_module("ORCHESTRATOR"))
    ls.store.record(evt, ignore_duplicates=True)
    cnt2 = len(ls.store.get_by_module("ORCHESTRATOR"))
    assert cnt1 == init_cnt + 1
    assert cnt2 == cnt1


def test_adv_14_restarts_durability_without_corruption():
    """Case 14: Restarts preserve SQLite events without data loss or duplicate records."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        s1 = SqliteEventStore(db_path=tmp.name)
        evt = LearningEvent(event_id="EVT-ADV-14", versioned_event_type="REBOOT", module="ORCHESTRATOR", payload={"val": 42})
        s1.record_event(evt)
        s1.close()

        s2 = SqliteEventStore(db_path=tmp.name)
        evts = s2.get_by_module("ORCHESTRATOR")
        s2.close()
        assert len(evts) == 1
        assert evts[0].event_id == "EVT-ADV-14"


def test_adv_15_fail_closed_on_optimizer_divergence(base_now: datetime):
    """Case 15: Subsystem exception cannot bypass safety gates (fails closed)."""
    class ExplodingOptimizer:
        def decide(self, *args, **kwargs):
            raise ArithmeticError("NUMERICAL OPTIMIZER MATRIX SINGULARITY")

    orch = DecisionOrchestrator(decision_service=ExplodingOptimizer())
    st = TreasuryState(operational_cash_usd=25_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=base_now)
    with pytest.raises(ArithmeticError):
        orch.run_cycle(current_state=st, as_of_timestamp=base_now, auto_simulate_execution=True)
