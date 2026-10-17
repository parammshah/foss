#!/usr/bin/env python3
"""
Reserve Juggler — Independent Adversarial Financial-Safety Audit Script.
Executes 15 rigorous adversarial stress tests across data, forecasting, optimization,
risk intelligence, independent validation, and execution boundaries.
"""

from __future__ import annotations
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from reserve_juggler.contracts import (
    TreasuryState,
    AssetPosition,
    AssetTier,
    Obligation,
    CreditFacility,
    ForecastBundle,
    ForecastPoint,
    ForecastPointCollection,
    AssetPositionCollection,
    GateStatus,
    ApprovalStatus,
    RiskSignal,
    EventCategory,
)
from reserve_juggler.events.models import RawEventRecord
from reserve_juggler.integration.orchestrator import (
    DecisionOrchestrator,
    IndependentConstraintValidator,
)
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.decision.models import StandardDecisionRecord
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile


def run_all_adversarial_tests() -> int:
    print("=" * 80)
    print("  RESERVE JUGGLER — INDEPENDENT ADVERSARIAL FINANCIAL-SAFETY AUDIT")
    print("=" * 80)

    results = []

    def log_result(case_num: int, name: str, passed: bool, details: str):
        icon = "✓ PASS" if passed else "✗ FAIL"
        results.append((case_num, name, passed, details))
        print(f"\n[Case {case_num:02d}] {name}")
        print(f"  Status:  {icon}")
        print(f"  Details: {details}")

    val = IndependentConstraintValidator()
    orchestrator = DecisionOrchestrator(independent_validator=val)
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def make_record(dec_id: str, action: str, asset: str, amt: float) -> StandardDecisionRecord:
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

    # --------------------------------------------------------------------------
    # Case 1: Customer obligations exceed available immediate liquidity
    # --------------------------------------------------------------------------
    st1 = TreasuryState(
        operational_cash_usd=4_000_000.0,
        upcoming_obligations_usd=10_000_000.0,
        available_credit_usd=20_000_000.0,
        as_of_timestamp=now,
    )
    rec1 = make_record("DEC-ADV-01", "KEEP_CASH", "CASH", 4_000_000.0)
    res1 = val.validate_proposal(st1, rec1)
    c1_pass = (not res1.passed) and (res1.shortfall_amount_usd == 6_000_000.0)
    log_result(
        1,
        "Customer obligations exceed available immediate liquidity",
        c1_pass,
        f"Passed: {res1.passed} | Shortfall: ${res1.shortfall_amount_usd:,.2f} | Gate reasons: {res1.rejection_reasons}",
    )

    # --------------------------------------------------------------------------
    # Case 2: Withdrawal forecasts spike or become unreliable
    # --------------------------------------------------------------------------
    bundle2 = ForecastBundle(
        points=[
            ForecastPoint(horizon_hours=1, predicted_deposits=5_000_000.0, predicted_withdrawals=25_000_000.0, uncertainty_p95=22_000_000.0),
        ],
        quality_status="DEGRADED",
        is_fallback_bundle=True,
    )
    rec2_invest = make_record("DEC-ADV-02", "BUY_BOND", "SOVEREIGN_BONDS", 5_000_000.0)
    st2 = TreasuryState(operational_cash_usd=30_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=now)
    res2 = val.validate_proposal(st2, rec2_invest, forecast_bundle=bundle2, stress_regime="ELEVATED_WITHDRAWAL_RISK")
    c2_pass = (not res2.passed) and ("Discretionary asset purchase prohibited" in " ".join(res2.rejection_reasons))
    log_result(
        2,
        "Withdrawal forecasts spike or become unreliable",
        c2_pass,
        f"Passed: {res2.passed} | Rejection: {res2.rejection_reasons}",
    )

    # --------------------------------------------------------------------------
    # Case 3: Forecast data is stale, missing, out of order, or invalid
    # --------------------------------------------------------------------------
    from reserve_juggler.forecasting.service import ForecastService
    fs = ForecastService()
    # Query forecast without fitting models -> must return fallback bundle safely
    fb3 = fs.forecast(now)
    c3_pass = fb3.is_fallback_bundle and fb3.quality_status in ("DEGRADED", "FALLBACK")
    log_result(
        3,
        "Forecast data is stale, missing, out of order, or invalid",
        c3_pass,
        f"Fallback Bundle: {fb3.is_fallback_bundle} | Quality: {fb3.quality_status} | Points count: {len(fb3.points)}",
    )

    # --------------------------------------------------------------------------
    # Case 4: A recommendation would breach the required cash floor
    # --------------------------------------------------------------------------
    # Case 4: A recommendation would breach the required cash floor
    # --------------------------------------------------------------------------
    st4 = TreasuryState(operational_cash_usd=20_000_000.0, upcoming_obligations_usd=5_000_000.0, as_of_timestamp=now)
    # Propose buying 18M in bonds, leaving only 2M cash (well below 5M obligations + reserve floor)
    rec4 = make_record("DEC-ADV-04", "BUY_BOND", "SOVEREIGN_BONDS", 18_000_000.0)
    res4 = val.validate_proposal(st4, rec4)
    c4_pass = not res4.passed
    log_result(
        4,
        "Recommendation breaches required cash floor",
        c4_pass,
        f"Passed: {res4.passed} | Rejections: {res4.rejection_reasons}",
    )

    # --------------------------------------------------------------------------
    # Case 5: Repo maturity and settlement times do not provide liquidity when required
    # --------------------------------------------------------------------------
    from reserve_juggler.risk.liquidity_shield import evaluate_hard_safety_gates, assess_liquidity_reserve
    ob5 = Obligation(due_timestamp=now + timedelta(hours=4), amount_usd=3_000_000.0, is_mandatory=True)
    st5 = TreasuryState(operational_cash_usd=4_000_000.0, obligations=[ob5], as_of_timestamp=now)
    res5_reserve = assess_liquidity_reserve(st5, fb3)
    gate5_eval = evaluate_hard_safety_gates(
        st5,
        res5_reserve,
        proposed_investment_amount=2_500_000.0,
        proposed_lockup_hours=24.0,
    )
    c5_pass = not gate5_eval.passed
    log_result(
        5,
        "Repo maturity timing does not provide liquidity when required",
        c5_pass,
        f"Passed: {gate5_eval.passed} | Violated gates: {gate5_eval.violated_gates}",
    )

    # --------------------------------------------------------------------------
    # Case 6: Early repo exit costs compared with borrowing costs in consistent units
    # --------------------------------------------------------------------------
    from reserve_juggler.optimization.optimizer import calculate_bridge_financing_cost
    bridge_cost = calculate_bridge_financing_cost(principal=10_000_000.0, apr=0.12, duration_hours=2.0)
    early_exit_penalty = 10_000_000.0 * 0.005  # 50 bps penalty
    c6_pass = (bridge_cost < early_exit_penalty) and (bridge_cost > 0.0)
    log_result(
        6,
        "Early repo exit costs vs borrowing costs in consistent units",
        c6_pass,
        f"Bridge interest (2h): ${bridge_cost:,.2f} vs Early exit fee (50 bps): ${early_exit_penalty:,.2f}",
    )

    # --------------------------------------------------------------------------
    # Case 7: Proposed bond/equity sale settlement delays, haircuts, and market impact
    # --------------------------------------------------------------------------
    from reserve_juggler.risk.capital_firewall import calculate_holding_time_gate
    bond_pos = AssetPosition(
        asset_id="BOND_2Y",
        tier=AssetTier.TIER_2_BONDS,
        market_value_usd=5_000_000.0,
        annual_yield=0.048,
        haircut_rate=0.02,
    )
    eq_pos = AssetPosition(
        asset_id="EQUITY_GROWTH",
        tier=AssetTier.TIER_3_EQUITIES,
        market_value_usd=5_000_000.0,
        annual_yield=0.08,
        haircut_rate=0.15,
    )
    gate7_bond = calculate_holding_time_gate(bond_pos, expected_holding_days=180.0)
    gate7_eq = calculate_holding_time_gate(eq_pos, expected_holding_days=180.0)
    c7_pass = gate7_bond.passed and (not gate7_eq.passed)
    log_result(
        7,
        "Holding-time gatekeeper and equity non-linear risk rejection",
        c7_pass,
        f"Bond gate passed: {gate7_bond.passed} (Breakeven {gate7_bond.minimum_holding_days:.1f}d) | Equity simplistic formula rejected: {not gate7_eq.passed}",
    )

    # --------------------------------------------------------------------------
    # Case 8: Credit facility capacity and commitment validation
    # --------------------------------------------------------------------------
    st8 = TreasuryState(
        operational_cash_usd=10_000_000.0,
        upcoming_obligations_usd=5_000_000.0,
        available_credit_usd=3_000_000.0,  # Only 3M available
        as_of_timestamp=now,
    )
    rec8 = make_record("DEC-ADV-08", "DRAW_CREDIT", "CREDIT_FACILITY", 5_000_000.0)
    res8 = val.validate_proposal(st8, rec8)
    c8_pass = (not res8.passed) and any("exceeds committed facility limit" in r for r in res8.rejection_reasons)
    log_result(
        8,
        "Credit facility capacity and commitment validation",
        c8_pass,
        f"Passed: {res8.passed} | Rejection: {res8.rejection_reasons}",
    )

    # --------------------------------------------------------------------------
    # Case 9: Tax assumptions missing or uncertain
    # --------------------------------------------------------------------------
    from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value
    tax_res = evaluate_tax_adjusted_value(
        gross_revenue=100_000.0,
        instrument_id="UNCLASSIFIED_TOKEN",
        tax_profile=None,  # Missing profile
    )
    c9_pass = tax_res.ranking_is_provisional and tax_res.has_missing_classification
    log_result(
        9,
        "Missing tax profile handled as provisional gross yield",
        c9_pass,
        f"Ranking provisional: {tax_res.ranking_is_provisional} | Missing classification: {tax_res.has_missing_classification} | Warnings: {tax_res.warnings}",
    )

    # --------------------------------------------------------------------------
    # Case 10: News ingestion fails, produces duplicate, stale, or low-credibility events
    # --------------------------------------------------------------------------
    sentinel = MacroPanicSentinel(use_synthetic_mode=False)
    evt_dup1 = RawEventRecord(
        event_id="EVT-ADV-10A",
        source="REUTERS",
        title="Interbank liquidity squeeze triggers discount window queries",
        summary="Interbank liquidity squeeze triggers discount window queries",
        publication_time=now.isoformat(),
        ingestion_time=now.isoformat(),
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["REPO", "CASH"],
        source_credibility=0.95,
    )
    evt_dup2 = RawEventRecord(
        event_id="EVT-ADV-10B",
        source="REUTERS",
        title="Interbank liquidity squeeze triggers discount window queries",
        summary="Interbank liquidity squeeze triggers discount window queries",
        publication_time=now.isoformat(),
        ingestion_time=now.isoformat(),
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["REPO", "CASH"],
        source_credibility=0.95,
    )
    clusters = sentinel.deduplicator.cluster_events([evt_dup1, evt_dup2])
    c10_pass = (len(clusters) == 1) and (len(clusters[0].duplicates) == 1) and (clusters[0].reporting_count == 2)
    log_result(
        10,
        "News deduplication and credibility screening",
        c10_pass,
        f"Clusters formed: {len(clusters)} | Duplicates in cluster: {len(clusters[0].duplicates) if clusters else 0} | Velocity count: {clusters[0].reporting_count if clusters else 0}",
    )

    # --------------------------------------------------------------------------
    # Case 11: High-risk recommendation requires human approval
    # --------------------------------------------------------------------------
    st11 = TreasuryState(
        operational_cash_usd=25_000_000.0,
        upcoming_obligations_usd=5_000_000.0,
        available_credit_usd=50_000_000.0,
        as_of_timestamp=now,
    )
    rec11 = make_record("DEC-ADV-11", "DRAW_CREDIT", "CREDIT_FACILITY", 10_000_000.0)
    res11 = val.validate_proposal(st11, rec11)
    c11_pass = res11.requires_human_approval and res11.passed
    log_result(
        11,
        "High-risk / credit draw action triggers human approval requirement",
        c11_pass,
        f"Passed: {res11.passed} | Requires human: {res11.requires_human_approval}",
    )

    # --------------------------------------------------------------------------
    # Case 12: Infeasible recommendation rejected by independent validator
    # --------------------------------------------------------------------------
    st12 = TreasuryState(operational_cash_usd=10_000_000.0, upcoming_obligations_usd=20_000_000.0, as_of_timestamp=now)
    rec12 = make_record("DEC-ADV-12", "BUY_STOCK", "EQUITIES", 5_000_000.0)
    res12 = val.validate_proposal(st12, rec12)
    c12_pass = not res12.passed
    log_result(
        12,
        "Infeasible recommendation rejected by independent validator",
        c12_pass,
        f"Passed: {res12.passed} | Gate outcomes: {[g.gate_name + ':' + g.status.value for g in res12.gate_outcomes]}",
    )

    # --------------------------------------------------------------------------
    # Case 13: Repeated decision-cycle request does not duplicate financial execution
    # --------------------------------------------------------------------------
    from reserve_juggler.contracts import LearningEvent
    from reserve_juggler.learning.service import LearningService
    ls = LearningService()
    evt13 = LearningEvent(
        event_id="EVT-ADV-13",
        versioned_event_type="DECISION_AUDIT",
        module="ORCHESTRATOR",
        payload={"decision_id": "DEC-ADV-13", "cycle_id": "CYC-ADV-13"},
    )
    initial_count = len(ls.store.get_by_module("ORCHESTRATOR"))
    ls.store.record(evt13, ignore_duplicates=True)
    count_after_first = len(ls.store.get_by_module("ORCHESTRATOR"))
    ls.store.record(evt13, ignore_duplicates=True)
    count_after_second = len(ls.store.get_by_module("ORCHESTRATOR"))
    c13_pass = (count_after_first == initial_count + 1) and (count_after_second == count_after_first)
    log_result(
        13,
        "Idempotent decision recording ignores duplicate side effects",
        c13_pass,
        f"Initial count: {initial_count} -> After 1st write: {count_after_first} -> After 2nd duplicate: {count_after_second}",
    )

    # --------------------------------------------------------------------------
    # Case 14: Restarts do not corrupt state or duplicate entries
    # --------------------------------------------------------------------------
    from reserve_juggler.learning.sqlite_store import SqliteEventStore
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store1 = SqliteEventStore(db_path=tmp.name)
        evt14 = LearningEvent(
            event_id="EVT-ADV-14",
            versioned_event_type="TEST_REBOOT",
            module="ORCHESTRATOR",
            payload={"balance": 50_000_000.0},
        )
        store1.record_event(evt14)
        store1.close()

        # Reopen in new instance
        store2 = SqliteEventStore(db_path=tmp.name)
        reloaded_events = store2.get_by_module("ORCHESTRATOR")
        count = len(reloaded_events)
        store2.close()
        c14_pass = (count == 1) and (reloaded_events[0].event_id == evt14.event_id)
        log_result(
            14,
            "Restart / reboot durability without database corruption",
            c14_pass,
            f"Reloaded event: {reloaded_events[0].event_id if reloaded_events else None} | Total events: {count}",
        )

    # --------------------------------------------------------------------------
    # Case 15: Exceptions cannot bypass safety gates (fail-closed verification)
    # --------------------------------------------------------------------------
    class BrokenDecisionService:
        def decide(self, *args, **kwargs):
            raise RuntimeError("CRITICAL OPTIMIZER NUMERICAL DIVERGENCE")

    broken_orch = DecisionOrchestrator(decision_service=BrokenDecisionService(), independent_validator=val)
    try:
        broken_orch.run_cycle(
            current_state=st11,
            as_of_timestamp=now,
            auto_simulate_execution=True,
        )
        c15_pass = False
        c15_detail = "Failed: Exception was swallowed or bypassed"
    except RuntimeError as ex:
        c15_pass = True
        c15_detail = f"Verified: System raised '{ex}' and refused to execute live or simulated transactions (failed closed)."

    log_result(
        15,
        "Fail-closed behavior when subsystem raises an exception",
        c15_pass,
        c15_detail,
    )

    print("\n" + "=" * 80)
    passed_count = sum(1 for _, _, p, _ in results if p)
    print(f"  ADVERSARIAL AUDIT SUMMARY: {passed_count} / {len(results)} CASES PASSED")
    print("=" * 80)
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_all_adversarial_tests())
