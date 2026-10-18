#!/usr/bin/env python3
"""Execute a complete end-to-end trace showing Team 1, Team 2, and Team 3 working together."""

from reserve_juggler.integration.facade import ApplicationFacade
from reserve_juggler.integration.contracts import UserRole

def main():
    facade = ApplicationFacade(use_synthetic=True)
    snap = facade.get_dashboard_snapshot()

    print("================================================================================")
    print("       RESERVE JUGGLER: CROSS-TEAM INTEGRATED END-TO-END EXECUTION TRACE        ")
    print("================================================================================")
    print("Snapshot Version:       ", snap.snapshot_version)
    print("Operational Scenario:   ", snap.active_scenario)
    print("Data Source Mode:       ", snap.data_source_mode)
    print(f"Treasury Operating Cash: ${snap.treasury_state.operational_cash_usd:,.2f}")
    print(f"Total Balance Sheet:     ${snap.treasury_state.total_balance_sheet_assets_usd:,.2f}")
    print(f"Total Liquidatable Val:  ${snap.treasury_state.total_liquidatable_value_usd:,.2f}")
    print(f"Committed Obligations:   ${snap.treasury_state.upcoming_obligations_usd:,.2f}")

    print("\n--------------------------------------------------------------------------------")
    print(" [1] TEAM 1: TIME-SERIES DATA & MULTI-HORIZON PROBABILISTIC FORECASTING        ")
    print("--------------------------------------------------------------------------------")
    fb = snap.forecast_bundle
    print("Forecast Timestamp:      ", fb.forecast_timestamp)
    print(f"Horizons Evaluated:       {len(fb.point_forecasts)}")
    for pt in fb.point_forecasts:
        print(f"  * Horizon {pt.horizon_hours:2d}h -> Deposits: ${pt.deposit_point_forecast:12,.2f} | "
              f"Withdrawals: ${pt.withdrawal_point_forecast:12,.2f} | "
              f"P95 Outflow: ${pt.net_outflow_p95:11,.2f} | Reliability: {pt.reliability_tier}")

    print("\n--------------------------------------------------------------------------------")
    print(" [2] TEAM 3: MACRO-PANIC SENTINEL & REAL-TIME RISK INTELLIGENCE                ")
    print("--------------------------------------------------------------------------------")
    print("Sentinel System Status:   ", snap.sentinel_result.system_status)
    print("Aggregate Stress (0-100):", snap.sentinel_result.aggregate_macro_stress_score)
    print(f"Active Signals Tracked:   {len(snap.sentinel_result.active_signals)}")
    for sig in snap.sentinel_result.active_signals:
        print(f"  * [{sig.severity}] {sig.signal_id} ({sig.category}): Score = {sig.sentiment_score:+.2f}")

    print("\n--------------------------------------------------------------------------------")
    print(" [3] TEAM 2: CONVEX OPTIMIZER, STATUTORY TAX ENGINE & ASSET ALLOCATION          ")
    print("--------------------------------------------------------------------------------")
    dec = snap.latest_decision
    print("Decision Correlation ID: ", dec.decision_id)
    print("Action Recommendation:   ", dec.selected_action)
    print("Target Vault / Tier:     ", dec.target_asset)
    print(f"Optimal Allocation:      ${dec.amount:,.2f}")
    print(f"Protected Hot Reserve:   ${float(dec.reserve) if isinstance(dec.reserve, (int, float)) else 0.0:,.2f}")
    print("Treasury Safety Score:   ", f"{dec.treasury_safety_score}/100")
    print("Governance Approval:     ", dec.approval_status)

    print("\n--------------------------------------------------------------------------------")
    print(" [4] TEAM 3: EXPLAINABILITY RATIONALE & INDEPENDENT SAFETY GATES               ")
    print("--------------------------------------------------------------------------------")
    exp = snap.decision_explanation
    print("Primary Audit Rationale: ", exp.primary_rationale)
    print("All Hard Gates Passed:   ", exp.all_hard_gates_passed)
    for gate in exp.hard_gates:
        print(f"  * Gate: {gate.gate_name:<36} | Passed: {str(gate.passed):<5} | Reason: {gate.reason}")
    if exp.rejected_alternatives:
        print("Explicitly Rejected Alternative Actions:")
        for alt in exp.rejected_alternatives:
            print(f"  * {alt.action_type:<14} -> Rejected: {alt.rejection_reason}")

    print("\n--------------------------------------------------------------------------------")
    print(" [5] TEAM 3: HUMAN GOVERNANCE SIGN-OFF & SIMULATED EXECUTION                    ")
    print("--------------------------------------------------------------------------------")
    ok, msg = facade.submit_human_action(
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved for execution under validated liquidity thresholds."
    )
    print("Sign-Off Accepted:       ", ok)
    print("System Response Message: ", msg)

    print("\n--------------------------------------------------------------------------------")
    print(" [6] TEAM 3: PERSISTENT SELF-LEARNING TELEMETRY & MULTI-ARMED BANDIT METRICS   ")
    print("--------------------------------------------------------------------------------")
    perf = facade.learning_service.evaluate_system_performance()
    print("Total Recorded Telemetry Events:", perf.total_learning_events)
    print("Overall System Health:         ", perf.overall_system_health)
    print("Active Production Versions:    ", perf.active_versions)
    print("Matched Decision/Outcome Pairs: ", len(facade.learning_service.matcher.match_all()))
    print("Decision Performance Summary:   ", f"Total Decisions = {perf.decision_metrics.total_decisions} | Gate Pass Rate = {perf.decision_metrics.hard_gate_pass_rate:.1%}")
    print("================================================================================")

if __name__ == "__main__":
    main()
