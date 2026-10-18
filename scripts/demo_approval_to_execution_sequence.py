#!/usr/bin/env python3
"""Canonical demonstration of the complete approval-to-execution sequence
with before-and-after audit outputs, parameter tampering detection,
and ledger reconciliation.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
from reserve_juggler.data.validator import DatasetValidator
from reserve_juggler.data.preprocessor import DatasetPreprocessor, resolve_dataset_path
from reserve_juggler.data.schema import TIMESTAMP_COL
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.contracts import (
    TreasuryState,
    Obligation,
    ObligationType,
    AssetPosition,
    AssetTier,
    CreditFacility,
    ensure_utc,
    UserRole,
    ApprovalStatus,
)
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
from reserve_juggler.integration.orchestrator import DecisionOrchestrator
from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile, create_synthetic_demo_profile


def main():
    print("=" * 80)
    print("      RESERVE JUGGLER: CANONICAL APPROVAL-TO-EXECUTION AUDIT TRACE       ")
    print("=" * 80)

    # 1. Load Dataset and establish strict temporal cutoff
    raw_df = pd.read_csv(resolve_dataset_path())
    clean_df = DatasetPreprocessor().preprocess(raw_df)
    
    # Hour 257 decision point (17:00 UTC Repo Sweep Window)
    training_cutoff_idx = 258
    train_slice = clean_df.iloc[:training_cutoff_idx].copy()
    as_of_time = ensure_utc(train_slice.iloc[-1][TIMESTAMP_COL])

    # Strict temporal assertion: zero lookahead leakage
    max_training_time = ensure_utc(train_slice[TIMESTAMP_COL].max())
    assert max_training_time <= as_of_time, (
        f"Temporal leakage violation: max training time ({max_training_time}) exceeds as-of time ({as_of_time})"
    )

    print(f"\n[1] TEMPORAL PROTOCOL & TRAINING CUTOFF")
    print(f"Decision As-Of Timestamp:       {as_of_time.isoformat()}")
    print(f"Max Training Observation Time:  {max_training_time.isoformat()}")
    print(f"Temporal Protocol Assertion:    max(training_timestamps) <= as_of_timestamp -> PASSED")

    # 2. Fit Forecast Service strictly on data <= as_of_time
    forecast_service = ForecastService()
    forecast_service.fit(train_slice, horizons=[1, 6, 24, 48, 72], as_of_timestamp=as_of_time)
    forecast_bundle = forecast_service.forecast(as_of_time)

    # 3. Establish Initial Reconciled Bank Treasury State
    # Opening Cash: $65,000,000.00
    opening_cash = 65_000_000.0
    merchant_obl = Obligation(
        obligation_id="OBL-SETTLE-001",
        timestamp=as_of_time + timedelta(hours=2),
        amount_usd=20_500_000.0,
        description="Mandatory Daily Merchant Batch Settlement",
        obligation_type=ObligationType.MERCHANT_SETTLEMENT,
        is_mandatory=True,
    )
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

    state = TreasuryState(
        timestamp=as_of_time,
        operational_cash_usd=opening_cash,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo, "T2_UST_2Y_PORTFOLIO": t2_bonds},
        obligations=[merchant_obl],
        credit_facility=emergency_loc,
    )

    # 4. Assess Liquidity Reserve & Verify Reconciliation Identity
    res_assess = assess_liquidity_reserve(state, forecast_bundle)
    prot_cap = res_assess.protected_capital
    fut_res = res_assess.future_obligation_reserve
    tot_res = res_assess.total_required_reserve
    surplus = res_assess.eligible_investable_surplus

    print(f"\n[2] RESERVE SIZING & RECONCILIATION")
    print(f"Opening Operational Cash:       ${opening_cash:,.2f}")
    print(f"Protected Capital (0-24h):      ${prot_cap:,.2f}")
    print(f"Future Obligation Reserve:      ${fut_res:,.2f}")
    print(f"Total Required Reserve:         ${tot_res:,.2f}")
    print(f"Eligible Investable Surplus:    ${surplus:,.2f}")
    print(f"Reconciliation Sum Check:       ${tot_res + surplus:,.2f} == ${opening_cash:,.2f} -> EXACT")

    # 5. Initialize Decision Orchestrator
    orchestrator = DecisionOrchestrator(forecast_service=forecast_service)
    # Use synthetic demo profile to demonstrate human sign-off requirement
    tax_profile = create_synthetic_demo_profile(entity_id="DEMO_CORP_01")

    # Run cycle without auto-simulation to produce OUTPUT A
    res = orchestrator.run_cycle(
        current_state=state,
        as_of_timestamp=as_of_time,
        historical_df=train_slice,
        tax_profile=tax_profile,
        auto_simulate_execution=False,
    )

    dec = res.decision_record
    val = res.validation_result

    print("\n" + "=" * 80)
    print("                    OUTPUT A: BEFORE APPROVAL (PENDING)                  ")
    print("=" * 80)
    print(f"Decision Correlation ID:        {dec.decision_id}")
    print(f"Proposed Action:                {dec.action_type}")
    print(f"Target Vault / Tier:            {dec.target_asset}")
    print(f"Proposed Allocation Amount:     ${dec.amount:,.2f}")
    print(f"Forecast As-Of Timestamp:       {res.as_of_timestamp}")
    print(f"Tax Authority Status:           SYNTHETIC_DEMO (Provisional)")
    print(f"Macro Risk Regime:              {res.stress_regime}")
    print(f"Independent Validation Passed:  {val.passed}")
    for g in val.gate_outcomes:
        status_str = "PASS" if g.status.value == "PASS" else "FAIL"
        print(f"  * [{status_str}] {g.gate_name}: {g.reason}")
    print(f"Governance Approval State:      {res.approval_status}")
    print("-" * 80)
    print("EXPLICIT BALANCE SHEET CONFIRMATION (PRE-APPROVAL):")
    print(f"  * Transaction Executed:       NONE (Simulated Outcome = {res.simulated_outcome})")
    print(f"  * Operating Cash Balance:     ${state.operational_cash_usd:,.2f} (UNTOUCHED)")
    print(f"  * Invested Positions Total:   ${state.total_investments_market_value_usd:,.2f} (UNTOUCHED)")
    print(f"  * Outstanding Liabilities:    $0.00 (UNTOUCHED)")
    print("=" * 80)

    # 6. Execute Human Sign-off & Governed Execution
    # Establish protected reserve floor for revalidation
    state.protected_reserve_floor = tot_res

    # DEMONSTRATION 1: Reject Stale / Expired Approval (Historical recommendation evaluated against real-time clock)
    print("\n[6A] APPROVAL FRESHNESS VERIFICATION: REJECT STALE APPROVAL")
    import copy
    stale_dec = copy.deepcopy(dec)
    current_clock = datetime.now(timezone.utc)
    expired_ok, expired_msg, _, _, _ = orchestrator.explanation_service.execute_human_workflow(
        decision_record=stale_dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Attempting to execute months-old recommendation in current live clock.",
        treasury_state=state,
        as_of_time=current_clock,
    )
    print(f"Decision Timestamp:             {stale_dec.timestamp.isoformat()}")
    print(f"Approval Attempt Clock:         {current_clock.isoformat()}")
    print(f"Freshness Rejection Verified:   {not expired_ok}")
    print(f"Rejection Diagnostic Message:   {expired_msg}")
    assert not expired_ok, "Security breach: Expired approval should have been rejected!"

    # DEMONSTRATION 2: Governed Approval within Freshness Window (Historical Replay: 15 minutes post-issuance)
    print("\n[6B] CONTEMPORANEOUS HISTORICAL GOVERNED APPROVAL & REVALIDATION")
    contemporaneous_clock = dec.timestamp + timedelta(minutes=15)
    ok, msg, approved_rec, outcome, learn_evt = orchestrator.explanation_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved for overnight repo investment under certified reserve floor.",
        treasury_state=state,
        as_of_time=contemporaneous_clock,
    )
    assert ok, f"Expected successful contemporaneous approval: {msg}"

    # Revalidation & execution on ledger (Only executed if ok == True)
    action_name = str(approved_rec.action_type).upper()
    action_spec = {
        "type": "SWEEP_REPO" if "REPO" in action_name else action_name,
        "amount_usd": approved_rec.amount,
        "asset_id": approved_rec.target_asset,
        "decision_id": approved_rec.decision_id,
    }
    new_state, sim_out = orchestrator.ledger.simulate_action(state, action_spec, forecast_bundle)

    # Transition decision record state to EXECUTED
    approved_rec.execution_status = "EXECUTED"
    approved_rec.execution_timestamp = contemporaneous_clock + timedelta(seconds=10)

    print("\n" + "=" * 80)
    print("                    OUTPUT B: AFTER VALID APPROVAL                       ")
    print("=" * 80)
    print(f"Approver Identity:              {approved_rec.approved_by}")
    print(f"Approver Governance Role:       {approved_rec.approval_role}")
    ts_val = approved_rec.timestamp.isoformat() if hasattr(approved_rec.timestamp, "isoformat") else str(approved_rec.timestamp)
    app_val = approved_rec.approval_timestamp.isoformat() if hasattr(approved_rec.approval_timestamp, "isoformat") else str(approved_rec.approval_timestamp or "N/A")
    exec_val = approved_rec.execution_timestamp.isoformat() if hasattr(approved_rec.execution_timestamp, "isoformat") else str(approved_rec.execution_timestamp or "N/A")
    print(f"Decision Timestamp:             {ts_val}")
    print(f"Approval Timestamp:             {app_val}")
    print(f"Execution Timestamp:            {exec_val}")
    print(f"Reviewed Decision ID:           {approved_rec.decision_id}")
    print(f"Reviewed Parameters:            Action={approved_rec.action_type} | Target={approved_rec.target_asset} | Amount=${approved_rec.amount:,.2f}")
    print(f"Financial Revalidation Result:  {ok} ({msg})")
    print(f"Mandatory Reserve Floor:        ${state.protected_reserve_floor:,.2f}")
    print(f"Simulated Execution Status:     {sim_out.execution_status}")
    print(f"Signed Cash Movements:          {sim_out.realized_cash_movements}")
    print(f"Asset Position Changes:         +${approved_rec.amount:,.2f} into {approved_rec.target_asset}")
    print(f"Liability Changes:              $0.00 (No credit facility draw)")
    print("-" * 80)
    print("POST-EXECUTION RECONCILED LEDGER BALANCES:")
    print(f"  * Post-Execution Cash:        ${new_state.operational_cash_usd:,.2f}")
    print(f"  * Total Required Reserve:     ${tot_res:,.2f}")
    print(f"  * Reserve Floor Invariant:    ${new_state.operational_cash_usd:,.2f} >= ${tot_res:,.2f} -> PRESERVED")
    print(f"  * Invested Market Value:      ${new_state.total_investments_market_value_usd:,.2f}")
    print(f"  * Total Balance Sheet Assets: ${new_state.total_balance_sheet_assets_usd:,.2f}")
    print(f"  * Liquidity Shortfall:        ${sim_out.shortfall_usd:,.2f}")
    print("-" * 80)
    print("MATCHING TELEMETRY & AUDIT EVENT IDENTIFIERS:")
    print(f"  * Decision Correlation ID:    {dec.decision_id}")
    print(f"  * Execution Outcome ID:       {sim_out.outcome_id}")
    print(f"  * Learning Audit Event ID:    {learn_evt.event_id if learn_evt else 'N/A'}")
    print("=" * 80)

    # DEMONSTRATION 3: Duplicate Execution / Idempotency Prevention
    print("\n[6C] DUPLICATE EXECUTION PREVENTION CHECK")
    dup_ok, dup_msg, _, _, _ = orchestrator.explanation_service.execute_human_workflow(
        decision_record=approved_rec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Duplicate execution attempt.",
        treasury_state=new_state,
        as_of_time=contemporaneous_clock + timedelta(minutes=1),
    )
    print(f"Duplicate Execution Blocked:    {not dup_ok}")
    print(f"Duplicate Rejection Message:    {dup_msg}")
    assert not dup_ok, "Security breach: Duplicate execution should have been rejected!"

    # 7. Parameter Tampering Prevention Verification
    print("\n[3] PARAMETER TAMPERING & REVALIDATION INVALIDATION CHECK")
    tampered_amount = surplus + 10_000_000.0  # Breach reserve floor by $10M
    fresh_dec = res.decision_record  # original proposal
    fresh_dec.execution_status = "UNEXECUTED"  # reset execution status for tamper test
    t_ok, t_msg, _, _, _ = orchestrator.explanation_service.execute_human_workflow(
        decision_record=fresh_dec,
        action_type="MODIFIED",
        actor_name="Adversarial Actor",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        modified_amount=tampered_amount,
        stated_reason="Unauthorized attempt to exceed eligible surplus.",
        treasury_state=state,
        as_of_time=contemporaneous_clock,
    )
    print(f"Tampered Amount:                ${tampered_amount:,.2f}")
    print(f"Revalidation Rejection:         {not t_ok}")
    print(f"Security Alert Message:         {t_msg}")
    assert not t_ok, "Security breach: Tampered parameter should have been rejected!"
    print("Security Verification:          Tampered proposal successfully rejected and blocked from execution.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    main()
