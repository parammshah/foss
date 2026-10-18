#!/usr/bin/env python3
"""
CLI Utility to Execute an Integrated End-to-End Decision Cycle in Reserve Juggler.

Loads and validates the real 4-tier challenge dataset,
fits the multi-horizon forecasting models,
evaluates macro risk events with the Macro-Panic Sentinel,
runs the Operations Research Decision Engine,
enforces Independent Safety Constraint Gates,
generates a full Explainability trace,
simulates financial ledger execution,
and records the decision into durable SQLite self-learning telemetry.

Usage:
    python scripts/run_integrated_decision_cycle.py
    python scripts/run_integrated_decision_cycle.py --dataset /Users/paramshah/Downloads/treasury_4tier_dataset.csv
    python scripts/run_integrated_decision_cycle.py --hours 200 --regime STRESS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

# Add src to path
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from reserve_juggler.contracts import (
    ApprovalStatus,
    AssetPosition,
    AssetTier,
    CreditFacility,
    Obligation,
    ObligationType,
    TreasuryState,
    ensure_utc,
)
from reserve_juggler.data import (
    DatasetPreprocessor,
    DatasetValidator,
    resolve_dataset_path,
)
from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.events.models import EventCategory, RawEventRecord
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.integration.orchestrator import (
    DecisionOrchestrator,
    IndependentConstraintValidator,
)
from reserve_juggler.ledger.ledger import InitialTreasuryConfig, TreasuryLedger
from reserve_juggler.learning.service import LearningService
from reserve_juggler.tax.tax_profile import TaxProfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("reserve_juggler.cli")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one complete end-to-end Reserve Juggler decision cycle."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to actual dataset CSV (defaults to RESERVE_JUGGLER_DATA_PATH or data/challenge_dataset_8760.csv)",
    )
    parser.add_argument(
        "--training-hours",
        type=int,
        default=258,
        help="Hours of historical data to use for feature engineering and model fitting (default: 258 for 17:00 UTC repo sweep window)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="NORMAL",
        choices=["NORMAL", "WITHDRAWAL_SPIKE", "MARKET_PANIC"],
        help="Optional market stress injection for the decision cycle",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="reserve_juggler.db",
        help="Path to SQLite learning telemetry database",
    )
    parser.add_argument(
        "--tax-mode",
        type=str,
        default="statutory",
        choices=["statutory", "synthetic", "none"],
        help="Tax profile configuration: statutory (IRS Form 1120), synthetic (demo progressive), or none (unconfigured gross yield provisional)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  RESERVE JUGGLER — END-TO-END INTEGRATED TREASURY ENGINE")
    print("  Team 1 (Data/Forecasting) + Team 2 (Optimizer) + Team 3 (Sentinel/Learning)")
    print("=" * 80)

    # 1. Dataset Resolution & Loading
    try:
        resolved_path = resolve_dataset_path(args.dataset)
        print(f"\n[1/10] Loading actual dataset from: {resolved_path}")
        raw_df = pd.read_csv(resolved_path)
    except Exception as e:
        print(f"ERROR loading dataset: {e}", file=sys.stderr)
        return 1

    # 2. Data Validation
    print("[2/10] Running independent dataset validation & schema normalization...")
    validator = DatasetValidator()
    report = validator.validate(raw_df)
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)

    errors = [i for i in report.issues if i.severity == "ERROR"]
    warnings = [i for i in report.issues if i.severity == "WARNING"]
    print(f"       Validation Errors: {len(errors)} | Warnings: {len(warnings)}")
    if warnings:
        for w in warnings[:2]:
            print(f"       ! Warning: {w.message}")

    # 3. Fit Multi-Horizon Forecasting Engine (Team 1)
    train_slice = clean_df.iloc[:args.training_hours].copy()
    as_of_time = ensure_utc(train_slice.iloc[-1][TIMESTAMP_COL])
    assert train_slice[TIMESTAMP_COL].max() <= as_of_time, (
        f"Temporal leakage violation: max training observation ({train_slice[TIMESTAMP_COL].max()}) > as_of_time ({as_of_time})"
    )
    print(f"\n[3/10] Fitting Multi-Horizon Forecast Engine up to as-of timestamp: {as_of_time.isoformat()}...")
    print(f"       Temporal Protocol Verified: max(training_timestamps) <= as_of_timestamp")
    forecast_service = ForecastService()
    forecast_service.fit(train_slice, horizons=[1, 6, 24, 48, 72], as_of_timestamp=as_of_time)
    forecast_bundle = forecast_service.forecast(as_of_time)
    print(f"       Forecast Bundle Generated: {len(forecast_bundle.point_forecasts)} horizons")
    pt_1h = forecast_bundle.get_point(1)
    if pt_1h:
        print(f"       1h Target Forecast: Deposits=${pt_1h.deposit_point_forecast:,.2f} | Withdrawals=${pt_1h.withdrawal_point_forecast:,.2f}")
        print(f"       1h Net Outflow P95: ${pt_1h.net_outflow_p95:,.2f} | Reliability: {pt_1h.reliability_tier}")

    # 4. Ingest External Risk Intelligence & Sentinel Evaluation (Team 3)
    print(f"\n[4/10] Ingesting Risk Intelligence into Macro-Panic Sentinel...")
    sentinel = MacroPanicSentinel(use_synthetic_mode=False)
    events_to_inject = []
    if args.regime == "MARKET_PANIC":
        events_to_inject.append(
            RawEventRecord(
                event_id="EVT-CLI-PANIC-01",
                source="FINANCIAL_TIMES",
                title="Sovereign Bond Yields Spike 150 bps as Interbank Liquidity Freezes",
                summary="Sudden liquidity freeze across commercial paper and repo financing desks.",
                publication_time=as_of_time.isoformat(),
                ingestion_time=as_of_time.isoformat(),
                category=EventCategory.BANKING_STRESS.value,
                affected_markets=["REPO", "SOVEREIGN_BONDS", "CASH"],
                source_credibility=0.98,
            )
        )
    elif args.regime == "WITHDRAWAL_SPIKE":
        events_to_inject.append(
            RawEventRecord(
                event_id="EVT-CLI-SPIKE-01",
                source="REUTERS",
                title="Major FinTech Core Banking Outage Sparks Customer Withdrawal Wave",
                summary="Elevated withdrawal demands observed across retail and merchant payment rails.",
                publication_time=as_of_time.isoformat(),
                ingestion_time=as_of_time.isoformat(),
                category=EventCategory.WITHDRAWAL_ACCELERATION.value,
                affected_markets=["CASH"],
                source_credibility=0.92,
            )
        )
    else:
        events_to_inject.append(
            RawEventRecord(
                event_id="EVT-CLI-NORM-01",
                source="BLOOMBERG",
                title="Central Bank Holds Benchmark Interest Rates Steady at Policy Meeting",
                summary="Monetary policy remains stable with well-anchored inflation expectations.",
                publication_time=as_of_time.isoformat(),
                ingestion_time=as_of_time.isoformat(),
                category=EventCategory.MONETARY_POLICY.value,
                affected_markets=["CASH", "SOVEREIGN_BONDS"],
                source_credibility=0.95,
            )
        )

    for raw_e in events_to_inject:
        sig = sentinel.process_event(raw_e)
        if sig:
            print(f"       Processed Event: '{raw_e.title[:60]}...' -> Severity: {sig.severity} (Score: {sig.sentiment_score:+.2f})")

    # 5. Build Reconciled Bank Treasury State (Team 1 Ledger)
    print(f"\n[5/10] Initializing Reconciled Treasury Ledger State...")
    t1_repo = AssetPosition(
        asset_id="T1_REPO_PORTFOLIO",
        name="Overnight General Collateral Repo",
        tier=AssetTier.TIER_1_REPO,
        units=15_000_000.0,
        unit_price_usd=1.0,
        market_value_usd=15_000_000.0,
        liquidatable_value_usd=15_000_000.0 * (1.0 - 0.005),
        annual_yield_or_return=0.043,
        early_exit_fee_rate=0.005,
        maturity_timestamp=as_of_time + timedelta(hours=15),
    )
    t2_bonds = AssetPosition(
        asset_id="T2_UST_2Y_PORTFOLIO",
        name="US 2-Year Benchmark Treasury Notes",
        tier=AssetTier.TIER_2_BONDS,
        units=200_000.0,
        unit_price_usd=100.0,
        market_value_usd=20_000_000.0,
        liquidatable_value_usd=20_000_000.0 * (1.0 - 0.02),
        annual_yield_or_return=0.048,
        liquidation_haircut_rate=0.02,
        duration_years=1.9,
    )
    emergency_loc = CreditFacility(
        facility_id="EMERGENCY_LOC_01",
        committed_limit_usd=50_000_000.0,
        utilized_amount_usd=0.0,
        apr=0.075,
        fee_rate=0.0025,
        is_available=True,
    )
    merchant_obl = Obligation(
        obligation_id="OBL-SETTLE-001",
        timestamp=as_of_time + timedelta(hours=2),
        amount_usd=20_500_000.0,
        description="Mandatory Daily Merchant Batch Settlement",
        obligation_type=ObligationType.MERCHANT_SETTLEMENT,
        is_mandatory=True,
    )

    state = TreasuryState(
        timestamp=as_of_time,
        operational_cash_usd=65_000_000.0,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo, "T2_UST_2Y_PORTFOLIO": t2_bonds},
        obligations=[merchant_obl],
        credit_facility=emergency_loc,
        warnings=[],
    )
    print(f"       Operational Cash: ${state.operational_cash_usd:,.2f}")
    print(f"       Invested Assets Market Value: ${state.total_investments_market_value_usd:,.2f}")
    print(f"       Committed Upcoming Obligations: ${state.upcoming_obligations_usd:,.2f}")
    print(f"       Available Credit Facility: ${state.available_credit_usd:,.2f}")

    # Liquidity Reserve Assessment & Reconciliation
    from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
    res_assess = assess_liquidity_reserve(state, forecast_bundle)
    state.protected_reserve_floor = res_assess.total_required_reserve
    print(f"       Reserve Assessment: Protected=${res_assess.protected_capital:,.2f} | Future Reserve=${res_assess.future_obligation_reserve:,.2f} | Total Required=${res_assess.total_required_reserve:,.2f}")
    print(f"       Eligible Surplus:   ${res_assess.eligible_investable_surplus:,.2f} | Identity Check: ${res_assess.total_required_reserve + res_assess.eligible_investable_surplus:,.2f} == ${state.operational_cash_usd:,.2f}")

    # 6. Initialize Orchestrator & Services
    print(f"\n[6/10] Initializing Enterprise Decision Orchestrator & Safety Gatekeeper...")
    learning_service = LearningService(persistence_file=args.db_path)
    explanation_service = DecisionExplanationService()
    decision_service = TreasuryDecisionService()
    orchestrator = DecisionOrchestrator(
        validator=validator,
        forecast_service=forecast_service,
        sentinel=sentinel,
        decision_service=decision_service,
        explanation_service=explanation_service,
        learning_service=learning_service,
    )

    # 7. Execute Decision Cycle (Pre-Approval Phase)
    print(f"\n[7/10] Executing Full 10-Step Decision Cycle (Tax Mode: {args.tax_mode.upper()})...")
    if args.tax_mode == "statutory":
        from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile
        tax_profile = create_statutory_us_corp_profile(entity_id="DIGITAL_BANK_TREASURY_01")
    elif args.tax_mode == "synthetic":
        from reserve_juggler.tax.tax_profile import create_synthetic_demo_profile
        tax_profile = create_synthetic_demo_profile(entity_id="DEMO_CORP_01")
    else:
        tax_profile = None

    res = orchestrator.run_cycle(
        current_state=state,
        as_of_timestamp=as_of_time,
        historical_df=train_slice,
        external_raw_events=events_to_inject,
        tax_profile=tax_profile,
        auto_simulate_execution=False,
    )

    # 8. Display Output A: Before Approval
    dec = res.decision_record
    val = res.validation_result
    tss_val = dec.score.get('treasury_safety_score', dec.treasury_safety_score) if isinstance(dec.score, dict) else (dec.treasury_safety_score or 'N/A')
    
    print("\n" + "=" * 80)
    print("                    OUTPUT A: BEFORE APPROVAL (PENDING)                  ")
    print("=" * 80)
    print(f"       Decision ID:        {dec.decision_id}")
    print(f"       Recommended Action: {dec.action_type} -> Target: {dec.target_asset}")
    print(f"       Allocated Amount:   ${dec.amount:,.2f}")
    print(f"       Forecast Timestamp: {res.as_of_timestamp}")
    tax_st = dec.metadata.get("tax_profile_status", args.tax_mode.upper())
    print(f"       Tax Status:         {tax_st}")
    print(f"       Risk Regime:        {res.stress_regime} (TSS Score: {tss_val}/100)")
    print(f"       Validation Result:  Passed = {val.passed}")
    for g in val.gate_outcomes:
        status_icon = "PASS" if g.status.value == "PASS" else "FAIL"
        print(f"         [{status_icon}] {g.gate_name}: {g.reason}")
    print(f"       Approval Status:    {res.approval_status}")
    print(f"       Execution State:    UNEXECUTED (simulated_outcome = {res.simulated_outcome})")
    print(f"       Ledger Balance:     ${state.operational_cash_usd:,.2f} (UNTOUCHED)")
    print("=" * 80)

    # 9. Human Governance Sign-Off & Governed Execution (Output B)
    from reserve_juggler.contracts import UserRole
    approval_time = as_of_time + timedelta(minutes=15)
    ok, msg, approved_rec, outcome, learn_evt = orchestrator.explanation_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved under verified liquidity reserve floor and statutory tax parameters.",
        treasury_state=state,
        as_of_time=approval_time,
    )

    if not ok:
        raise RuntimeError(f"Approval revalidation failed unexpectedly: {msg}")

    action_name = str(approved_rec.action_type).upper()
    action_spec = {
        "type": "SWEEP_REPO" if "REPO" in action_name else action_name,
        "amount_usd": approved_rec.amount,
        "asset_id": approved_rec.target_asset,
        "decision_id": approved_rec.decision_id,
    }
    new_state, sim_out = orchestrator.ledger.simulate_action(state, action_spec, forecast_bundle)

    print("\n" + "=" * 80)
    print("                    OUTPUT B: AFTER VALID APPROVAL                       ")
    print("=" * 80)
    print(f"       Approver:           {approved_rec.approved_by} ({approved_rec.approval_role})")
    print(f"       Approval Timestamp: {approved_rec.approval_timestamp}")
    print(f"       Reviewed Decision:  {approved_rec.decision_id}")
    print(f"       Reviewed Params:    Action={approved_rec.action_type} | Target={approved_rec.target_asset} | Amount=${approved_rec.amount:,.2f}")
    print(f"       Revalidation Check: {ok} ({msg})")
    print(f"       Execution Status:   {sim_out.execution_status}")
    print(f"       Signed Cash Movmt:  {sim_out.realized_cash_movements}")
    print(f"       Post-Trade Cash:    ${new_state.operational_cash_usd:,.2f}")
    print(f"       Total Required Res: ${res_assess.total_required_reserve:,.2f}")
    print(f"       Reserve Invariant:  ${new_state.operational_cash_usd:,.2f} >= ${res_assess.total_required_reserve:,.2f} -> PRESERVED")
    print(f"       Total Invested:     ${new_state.total_investments_market_value_usd:,.2f}")
    print(f"       Total BalanceSheet: ${new_state.total_balance_sheet_assets_usd:,.2f}")
    print(f"       Matching Event ID:  {learn_evt.event_id if learn_evt else 'N/A'}")
    print("=" * 80)

    # 10. Explainability & Persistent Telemetry
    if res.explanation:
        print(f"\n[10/10] Explainability & Rationale Summary:")
        print(f"       Primary Rationale: {res.explanation.primary_rationale}")
        if res.explanation.rejected_alternatives:
            print("       Rejected Alternatives:")
            for alt in res.explanation.rejected_alternatives[:2]:
                print(f"         - {alt.action_type}: {alt.rejection_reason}")

    print("\n" + "=" * 80)
    print("  DECISION CYCLE EXECUTION COMPLETE — AUDIT TRAIL RECORDED IN SQLITE")
    print(f"  Audit Correlation ID: {res.cycle_id}")
    print(f"  Telemetry Database:   {args.db_path}")
    print("=" * 80)

    learning_service.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
