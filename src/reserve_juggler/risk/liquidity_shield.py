"""
Adaptive Liquidity Shield and Hard Safety Gates.

Implements dynamic reserve sizing, capital protection firewalls, and hard safety gates.
Protects customer payouts unconditionally before optimizing investment yields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from reserve_juggler.decision.protocols import (
        AssetPosition,
        ForecastBundle,
        Obligation,
        RiskSignal,
        TreasuryState,
    )
from reserve_juggler.risk.models import (
    HardGateEvaluation,
    ReserveAssessment,
)


def assess_liquidity_reserve(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    risk_signals: Optional[Sequence[RiskSignal]] = None,
    operational_safety_buffer: float = 0.10,  # 10% operational margin on obligations
    payout_window_hours: float = 24.0,
    stress_multiplier: float = 1.0,
) -> ReserveAssessment:
    """Assess dynamic liquidity reserves and eligible investable surplus.

    Distinguishes:
    - Protected Capital: cash and buffer strictly needed to meet immediate payout obligations.
    - Future-Obligation Reserve: additional capital required for future forecast deficits
      before longer-term assets or repos mature and settle.
    - Eligible Investable Surplus: remaining unencumbered funds eligible for deployment.

    Prevents double-counting overlapping horizons or reserves.
    """
    now = state.timestamp
    available_cash = max(0.0, float(state.available_cash))
    risk_signals = risk_signals or []

    # Apply risk signal stress multiplier if high severity risk signals are present
    signal_stress = 1.0
    for sig in risk_signals:
        if sig.severity.upper() == "CRITICAL":
            signal_stress = max(signal_stress, 1.30)
        elif sig.severity.upper() == "HIGH":
            signal_stress = max(signal_stress, 1.15)
    effective_stress = stress_multiplier * signal_stress

    # 1. Immediate Payout Obligations (within payout_window_hours, typically 24h)
    immediate_obligations = 0.0
    future_obligations_beyond = 0.0
    for ob in state.future_obligations:
        hours_diff = (ob.due_time - now).total_seconds() / 3600.0
        amt = float(ob.amount)
        if hours_diff <= payout_window_hours:
            immediate_obligations += amt
        else:
            future_obligations_beyond += amt

    # Operational safety margin on immediate obligations
    op_buffer = immediate_obligations * operational_safety_buffer

    # 2. Cumulative Forecast Deficit Analysis
    # We analyze cumulative net outflow = max(0, withdrawal_p90 - deposit_p10)
    # Sorted by forecast horizon
    sorted_points = sorted(
        forecast_bundle.point_forecasts or [], key=lambda p: float(p.forecast_horizon)
    )

    immediate_forecast_deficit = 0.0
    max_cumulative_future_deficit = 0.0
    cumulative_forecast_outflow = 0.0

    # Collect repo maturities by horizon
    repo_maturities_by_hour: list[tuple[float, float]] = []
    total_maturing_repo_liquidity = 0.0

    for pos in state.asset_positions:
        if pos.instrument_type.lower() == "repo" or pos.tier == 1:
            if pos.maturity:
                hours_to_mat = max(0.0, (pos.maturity - now).total_seconds() / 3600.0)
                # Repo proceeds net of fees upon maturity
                net_repo_val = float(pos.market_value) * (1.0 - float(pos.exit_fees))
                repo_maturities_by_hour.append((hours_to_mat, net_repo_val))
                total_maturing_repo_liquidity += net_repo_val

    for fp in sorted_points:
        horizon = float(fp.forecast_horizon)
        # Conservative net outflow rate
        pt_deficit = max(0.0, float(fp.withdrawal_p90) - float(fp.deposit_p10)) * effective_stress
        cumulative_forecast_outflow += pt_deficit

        if horizon <= payout_window_hours:
            immediate_forecast_deficit += pt_deficit
        else:
            # Future horizon: check how much repo liquidity has matured by this horizon
            matured_by_h = sum(
                val for h_mat, val in repo_maturities_by_hour if h_mat <= horizon
            )
            net_gap = max(0.0, cumulative_forecast_outflow - matured_by_h)
            if net_gap > max_cumulative_future_deficit:
                max_cumulative_future_deficit = net_gap

    # Protected Capital covers immediate obligations + buffer + immediate forecast deficit
    protected_capital = immediate_obligations + op_buffer + immediate_forecast_deficit

    # Future-Obligation Reserve covers future deficits not satisfied by maturing assets or protected capital
    # Avoid double counting: subtracting what's already protected
    future_obligation_reserve = max(0.0, max_cumulative_future_deficit - immediate_forecast_deficit)

    # Total required reserves
    total_required_reserve = protected_capital + future_obligation_reserve

    # Eligible Investable Surplus: remaining unencumbered cash
    eligible_surplus = max(0.0, available_cash - total_required_reserve)

    # Identify uncovered stress scenarios explicitly
    uncovered_scenarios: list[str] = []
    total_liquidatable_resources = available_cash + float(state.credit_availability)
    for pos in state.asset_positions:
        stressed_val = float(pos.market_value) * (1.0 - float(pos.haircut) - float(pos.exit_fees))
        total_liquidatable_resources += max(0.0, stressed_val)

    worst_case_stress_demand = (immediate_obligations + future_obligations_beyond) * 1.5 + (
        cumulative_forecast_outflow * 2.0
    )

    if total_liquidatable_resources < worst_case_stress_demand:
        deficit_amt = worst_case_stress_demand - total_liquidatable_resources
        uncovered_scenarios.append(
            f"Severe compound shock: Combined 50% obligation spike and 2x P90 withdrawal would exceed total resources by ${deficit_amt:,.2f}"
        )

    if float(state.credit_availability) <= 0.0 and available_cash < immediate_obligations:
        uncovered_scenarios.append("Credit facility exhausted during immediate cash deficit.")

    return ReserveAssessment(
        protected_capital=protected_capital,
        future_obligation_reserve=future_obligation_reserve,
        eligible_investable_surplus=eligible_surplus,
        immediate_obligations=immediate_obligations,
        operational_safety_buffer=op_buffer,
        conservative_net_deficit=immediate_forecast_deficit + future_obligation_reserve,
        maturing_repo_liquidity=total_maturing_repo_liquidity,
        uncovered_stress_scenarios=uncovered_scenarios,
    )


def evaluate_hard_safety_gates(
    state: TreasuryState,
    reserve_assessment: ReserveAssessment,
    proposed_investment_amount: float = 0.0,
    proposed_lockup_hours: float = 0.0,
    proposed_bridge_draw: float = 0.0,
    target_tier: int = 1,
) -> HardGateEvaluation:
    """Evaluate non-negotiable safety gates independently of the Treasury Safety Score.

    Hard gates reject any proposed allocation or action that compromises customer payouts,
    breaches liquidity buffers, or exceeds credit authorization, regardless of expected yield.
    """
    now = state.timestamp
    available_cash = max(0.0, float(state.available_cash))
    credit_avail = max(0.0, float(state.credit_availability))
    immediate_obligations = reserve_assessment.immediate_obligations

    gates: dict[str, bool] = {}
    violated_gates: list[str] = []
    reason_codes: list[str] = []

    # Hard Gate 1: Immediate Payout Coverage Gate
    # Total available immediate funding (cash + credit) must be >= immediate mandatory payouts
    payout_coverage_passed = (available_cash + credit_avail) >= immediate_obligations
    gates["immediate_payout_coverage"] = payout_coverage_passed
    if not payout_coverage_passed:
        violated_gates.append("immediate_payout_coverage")
        reason_codes.append("BREACH_IMMEDIATE_PAYOUT_COVERAGE: Immediate funding insufficient for payouts.")

    # Hard Gate 2: Minimum Protected Reserve Gate
    # Any new investment allocation must NOT deplete cash below protected capital
    if proposed_investment_amount > 0.0:
        remaining_cash_after_investment = available_cash - proposed_investment_amount
        reserve_gate_passed = remaining_cash_after_investment >= (
            reserve_assessment.protected_capital + reserve_assessment.future_obligation_reserve
        )
        gates["protected_reserve_intact"] = reserve_gate_passed
        if not reserve_gate_passed:
            violated_gates.append("protected_reserve_intact")
            reason_codes.append(
                f"BREACH_PROTECTED_RESERVE: Investment of ${proposed_investment_amount:,.2f} "
                f"exceeds eligible surplus of ${reserve_assessment.eligible_investable_surplus:,.2f}."
            )
    else:
        gates["protected_reserve_intact"] = True

    # Hard Gate 3: Repo / Asset Lockup Timing Gate
    # A lockup must not trap cash across an obligation due date unless surplus cash covers it
    if proposed_lockup_hours > 0.0 and proposed_investment_amount > 0.0:
        # Check if obligations due before lockup ends exceed remaining uncommitted cash
        obligations_during_lockup = 0.0
        for ob in state.future_obligations:
            h = (ob.due_time - now).total_seconds() / 3600.0
            if 0.0 <= h <= proposed_lockup_hours:
                obligations_during_lockup += float(ob.amount)

        free_cash = available_cash - proposed_investment_amount
        timing_safe = free_cash >= obligations_during_lockup
        gates["lockup_timing_feasibility"] = timing_safe
        if not timing_safe:
            violated_gates.append("lockup_timing_feasibility")
            reason_codes.append(
                f"BREACH_LOCKUP_TIMING: Asset lockup of {proposed_lockup_hours:.1f}h conflicts with "
                f"obligations of ${obligations_during_lockup:,.2f} due during lockup period."
            )
    else:
        gates["lockup_timing_feasibility"] = True

    # Hard Gate 4: Authorized Credit Availability Gate
    # Proposed credit bridge draw must not exceed currently available authorized limit
    if proposed_bridge_draw > 0.0:
        credit_safe = proposed_bridge_draw <= credit_avail
        gates["credit_availability_limit"] = credit_safe
        if not credit_safe:
            violated_gates.append("credit_availability_limit")
            reason_codes.append(
                f"BREACH_CREDIT_LIMIT: Requested draw of ${proposed_bridge_draw:,.2f} "
                f"exceeds authorized facility capacity of ${credit_avail:,.2f}."
            )
    else:
        gates["credit_availability_limit"] = True

    all_passed = len(violated_gates) == 0

    return HardGateEvaluation(
        passed=all_passed,
        gates=gates,
        violated_gates=violated_gates,
        reason_codes=reason_codes,
        uncovered_scenarios=list(reserve_assessment.uncovered_stress_scenarios),
    )
