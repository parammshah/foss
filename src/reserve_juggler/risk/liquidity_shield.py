"""
Adaptive Liquidity Shield and Hard Safety Gates.

Implements dynamic reserve sizing, capital protection firewalls, and action-specific hard safety gates.
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
from reserve_juggler.contracts import ensure_utc


def assess_liquidity_reserve(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    risk_signals: Optional[Sequence[RiskSignal]] = None,
    operational_safety_buffer: float = 0.10,  # 10% operational margin on obligations
    payout_window_hours: float = 24.0,
    stress_multiplier: float = 1.0,
    forecast_is_cumulative: bool = False,
    forecast_horizon_hours: float = 72.0,
    use_calibrated_net_quantile: bool = False,
    policy_is_certified: bool = False,
) -> ReserveAssessment:
    """Assess dynamic liquidity reserves and eligible investable surplus.

    Distinguishes:
    - Protected Capital: cash and buffer strictly needed to meet immediate payout obligations (0-24h).
    - Future-Obligation Reserve: additional capital required for future forecast deficits
      and scheduled obligations (24-72h) before longer-term assets or repos mature and settle.
    - Eligible Investable Surplus: remaining unencumbered funds eligible for deployment.

    Enforces invariants:
    - Increasing scheduled obligations cannot decrease required reserves.
    - Normal repo maturity does not incur an early-exit penalty.
    - Horizon semantics prevent double-counting.
    """
    now = ensure_utc(state.timestamp)
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

    # 1. Scheduled Obligations by Horizon
    immediate_obligations = 0.0
    future_obligations_beyond = 0.0
    obligations_by_hour: list[tuple[float, float]] = []

    for ob in state.future_obligations:
        hours_diff = (ob.due_time - now).total_seconds() / 3600.0
        amt = float(ob.amount)
        if hours_diff <= forecast_horizon_hours:
            obligations_by_hour.append((hours_diff, amt))
            if hours_diff <= payout_window_hours:
                immediate_obligations += amt
            else:
                future_obligations_beyond += amt

    # Operational safety margin on immediate obligations
    op_buffer = immediate_obligations * operational_safety_buffer

    # 2. Maturing Repo Cash Availability (Normal maturity does NOT incur early exit fees!)
    repo_maturities_by_hour: list[tuple[float, float]] = []
    total_maturing_repo_liquidity = 0.0

    for pos in state.asset_positions:
        if pos.instrument_type.lower() == "repo" or pos.tier == 1:
            if pos.maturity:
                hours_to_mat = max(0.0, (pos.maturity - now).total_seconds() / 3600.0)
                # Normal maturity proceeds: full nominal value without early redemption penalty
                full_repo_val = float(pos.market_value)
                repo_maturities_by_hour.append((hours_to_mat, full_repo_val))
                total_maturing_repo_liquidity += full_repo_val

    # 3. Forecast Deficit Analysis
    # Filter constituent forecast points strictly within the declared evaluation horizon
    sorted_points = sorted(
        [
            p
            for p in (forecast_bundle.point_forecasts or [])
            if float(p.forecast_horizon) <= forecast_horizon_hours
        ],
        key=lambda p: float(p.forecast_horizon),
    )

    immediate_forecast_deficit = 0.0
    max_future_net_gap = 0.0
    running_cumulative_outflow = 0.0

    for fp in sorted_points:
        horizon = float(fp.forecast_horizon)
        rel_status = getattr(fp, "calibration_reliability_status", "EMPIRICALLY_VALIDATED")
        is_reliable = getattr(fp, "calibration_is_reliable", True)

        if not is_reliable or rel_status == "DEGRADED":
            # Degraded calibration triggers documented conservative liquidity policy:
            # Enforce conservative marginal proxy bound with additional 15% stress buffer
            base_proxy = max(0.0, float(getattr(fp, "net_outflow_marginal_proxy", 0.0) or (float(fp.withdrawal_p90) - float(fp.deposit_p10))))
            pt_deficit = max(base_proxy * 1.15, float(getattr(fp, "net_outflow_p95", 0.0) or 0.0)) * effective_stress
        elif use_calibrated_net_quantile and hasattr(fp, "net_outflow_p95") and fp.net_outflow_p95 is not None:
            # Calibrated quantile (or conditionally calibrated fallback with 1.55x diurnal buffer at h=6)
            pt_deficit = max(0.0, float(fp.net_outflow_p95)) * effective_stress
        else:
            # Conservative marginal proxy heuristic bound: max(0, W_P90 - D_P10)
            pt_deficit = max(0.0, float(fp.withdrawal_p90) - float(fp.deposit_p10)) * effective_stress

        is_cum = forecast_is_cumulative or getattr(forecast_bundle, "is_cumulative", False) or getattr(fp, "is_cumulative", False)
        has_calibrated_cum = (
            use_calibrated_net_quantile
            and getattr(fp, "cumulative_net_outflow_p95", None) is not None
            and float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0) > 0.0
        )

        if is_cum:
            # Point value already represents cumulative total through this horizon
            cum_outflow_at_h = pt_deficit
        elif has_calibrated_cum:
            # Calibrated multi-hour cumulative quantile from uncertainty engine
            cum_outflow_at_h = max(0.0, float(fp.cumulative_net_outflow_p95)) * effective_stress
        else:
            # Interval accumulation for discrete hourly flows
            running_cumulative_outflow += pt_deficit
            cum_outflow_at_h = running_cumulative_outflow

        # Cumulative scheduled obligations due by this horizon
        cum_obligations_at_h = sum(
            amt for h_due, amt in obligations_by_hour if h_due <= horizon
        )

        # Total repo liquidity available by this horizon
        matured_repo_at_h = sum(
            val for h_mat, val in repo_maturities_by_hour if h_mat <= horizon
        )

        total_demand_at_h = cum_outflow_at_h + cum_obligations_at_h

        if horizon <= payout_window_hours:
            if is_cum or has_calibrated_cum:
                immediate_forecast_deficit = max(immediate_forecast_deficit, cum_outflow_at_h)
            else:
                immediate_forecast_deficit = cum_outflow_at_h
        else:
            # Future horizon: check unmet net gap taking into account maturing repo
            net_gap = max(0.0, total_demand_at_h - matured_repo_at_h)
            if net_gap > max_future_net_gap:
                max_future_net_gap = net_gap

    # Also check scheduled obligations at every material payment deadline within evaluation horizon
    obligation_deadlines = [h_due for h_due, _ in obligations_by_hour if h_due > payout_window_hours and h_due <= forecast_horizon_hours]
    future_check_horizons = sorted(set([h for h in [30.0, 48.0, 72.0] if h <= forecast_horizon_hours] + obligation_deadlines))

    for horizon in future_check_horizons:
        cum_obligations_at_h = sum(
            amt for h_due, amt in obligations_by_hour if h_due <= horizon
        )
        matured_repo_at_h = sum(
            val for h_mat, val in repo_maturities_by_hour if h_mat <= horizon
        )
        net_gap = max(0.0, cum_obligations_at_h - matured_repo_at_h)
        if net_gap > max_future_net_gap:
            max_future_net_gap = net_gap

    # Protected Capital covers immediate obligations (0-24h) + buffer + immediate forecast deficit
    protected_capital = immediate_obligations + op_buffer + immediate_forecast_deficit

    # Future-Obligation Reserve covers future obligations and forecast deficits (24-72h)
    # net of maturing repo liquidity and avoiding double counting what's protected
    immediate_covered = immediate_obligations + immediate_forecast_deficit
    future_obligation_reserve = max(0.0, max_future_net_gap - immediate_covered)

    # Total required reserves
    total_required_reserve = protected_capital + future_obligation_reserve

    # Eligible Investable Surplus: remaining unencumbered cash strictly beyond required reserves
    eligible_surplus = max(0.0, available_cash - total_required_reserve)

    # Identify uncovered stress scenarios explicitly
    uncovered_scenarios: list[str] = []
    total_liquidatable_resources = available_cash + float(state.credit_availability)
    for pos in state.asset_positions:
        pos_ef = float(pos.exit_fees) if pos.exit_fees is not None else 0.005
        pos_hc = float(pos.haircut) if pos.haircut is not None else 0.0
        stressed_val = float(pos.market_value) * (1.0 - pos_hc - pos_ef)
        total_liquidatable_resources += max(0.0, stressed_val)

    worst_case_stress_demand = (immediate_obligations + future_obligations_beyond) * 1.5 + (
        (running_cumulative_outflow or immediate_forecast_deficit) * 2.0
    )

    if total_liquidatable_resources < worst_case_stress_demand:
        deficit_amt = worst_case_stress_demand - total_liquidatable_resources
        uncovered_scenarios.append(
            f"Severe compound shock: Combined 50% obligation spike and 2x P90 withdrawal would exceed total resources by ${deficit_amt:,.2f}"
        )

    if float(state.credit_availability) <= 0.0 and available_cash < immediate_obligations:
        uncovered_scenarios.append("Credit facility exhausted during immediate cash deficit.")

    # Determine policy reliability status from constituent forecast points
    has_degraded = False
    has_fallback = False
    for fp in sorted_points:
        p_stat = getattr(fp, "point_reliability_status", getattr(fp, "calibration_reliability_status", "EMPIRICALLY_VALIDATED"))
        c_stat = getattr(fp, "cumulative_reliability_status", p_stat)
        if p_stat == "DEGRADED" or c_stat == "DEGRADED" or not getattr(fp, "calibration_is_reliable", True):
            has_degraded = True
        elif p_stat == "CONDITIONALLY_CALIBRATED_FALLBACK" or c_stat == "CONDITIONALLY_CALIBRATED_FALLBACK":
            has_fallback = True

    if has_degraded or not sorted_points:
        policy_rel_status = "DEGRADED"
    elif has_fallback or any(float(fp.forecast_horizon) > 1.0 for fp in sorted_points):
        # Multi-hour targets contain uncalibrated cumulative proxies or heuristic volatility buffers
        policy_rel_status = "CONDITIONALLY_CALIBRATED_FALLBACK"
    elif not policy_is_certified:
        # Policy-level evidence is insufficient to certify >= 95% population coverage under operational liquidity dynamics.
        # Although 1h constituent input is calibrated, finite holdout (e.g. N=100, Wilson [93.0%, 99.4%]) does not certify
        # population policy coverage under serial dependence. Retain conditionally calibrated fallback.
        policy_rel_status = "CONDITIONALLY_CALIBRATED_FALLBACK"
    else:
        # Strictly 1h single-period horizon where point and cumulative forecasts are direct unbuffered conformal quantiles
        # AND explicit policy-level statistical certification has been verified.
        policy_rel_status = "EMPIRICALLY_VALIDATED"

    return ReserveAssessment(
        protected_capital=protected_capital,
        future_obligation_reserve=future_obligation_reserve,
        eligible_investable_surplus=eligible_surplus,
        immediate_obligations=immediate_obligations,
        operational_safety_buffer=op_buffer,
        conservative_net_deficit=immediate_forecast_deficit + future_obligation_reserve,
        maturing_repo_liquidity=total_maturing_repo_liquidity,
        uncovered_stress_scenarios=uncovered_scenarios,
        policy_reliability_status=policy_rel_status,
    )



def evaluate_hard_safety_gates(
    state: TreasuryState,
    reserve_assessment: ReserveAssessment,
    proposed_investment_amount: float = 0.0,
    proposed_lockup_hours: float = 0.0,
    proposed_bridge_draw: float = 0.0,
    target_tier: int = 1,
    payment_deadline_hours: Optional[float] = None,
    availability_delay_hours: Optional[float] = None,
    repayment_feasibility: bool = True,
    is_partial_funding: bool = False,
    residual_unfunded_amount: float = 0.0,
    action_type: Optional[str] = None,
    resulting_tier_exposure_pct: Optional[float] = None,
    tier_portfolio_cap_pct: Optional[float] = None,
    allow_emergency_credit_draw: bool = False,
) -> HardGateEvaluation:
    """Evaluate non-negotiable safety gates independently of the Treasury Safety Score.

    Hard gates reject any proposed allocation or action that compromises customer payouts,
    breaches liquidity buffers, exceeds credit authorization, or proposes an un-repayable loan.
    """
    now = ensure_utc(state.timestamp)
    available_cash = max(0.0, float(state.available_cash))
    credit_avail = max(0.0, float(state.credit_availability))
    immediate_obligations = reserve_assessment.immediate_obligations

    gates: dict[str, bool] = {}
    violated_gates: list[str] = []
    reason_codes: list[str] = []

    # Hard Gate 1: Immediate Payout Coverage Gate
    # For funding actions, evaluate coverage considering the action's net proceeds by deadline.
    is_funding_action = (
        action_type in (
            "BREAK_REPO_EARLY",
            "SELL_BOND",
            "SELL_EQUITY",
            "DRAW_BRIDGE_CREDIT",
            "WAIT_REPO_MATURITY",
        )
        or proposed_bridge_draw > 0.0
    )
    if is_funding_action:
        deadline_met = (
            availability_delay_hours is None
            or payment_deadline_hours is None
            or availability_delay_hours <= payment_deadline_hours
        )
        if payment_deadline_hours is not None:
            payout_coverage_passed = (residual_unfunded_amount == 0.0) and deadline_met
        else:
            total_funds = available_cash + proposed_bridge_draw
            payout_coverage_passed = (total_funds >= immediate_obligations) and (residual_unfunded_amount == 0.0)
    else:
        # Non-funding action (e.g. KEEP_CASH, BUY_BOND, BUY_EQUITY, UNRESOLVED_FUNDING_SHORTFALL):
        # Do NOT treat undrawn credit line as cash!
        # Payout coverage requires actual available cash to cover immediate obligations.
        payout_coverage_passed = available_cash >= immediate_obligations

    gates["immediate_payout_coverage"] = payout_coverage_passed
    if not payout_coverage_passed:
        violated_gates.append("immediate_payout_coverage")
        reason_codes.append(
            f"BREACH_IMMEDIATE_PAYOUT_COVERAGE: Immediate funding insufficient for payouts "
            f"(${immediate_obligations:,.2f} required, residual gap: ${residual_unfunded_amount:,.2f})."
        )

    # Hard Gate 2: Minimum Protected Reserve Gate (for new surplus allocations)
    if proposed_investment_amount > 0.0 and proposed_bridge_draw == 0.0:
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

    # Hard Gate 3: Lockup Timing Gate
    if proposed_lockup_hours > 0.0 and proposed_investment_amount > 0.0:
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

    # Hard Gate 5: Funding Deadline Feasibility Gate (for emergency actions)
    if payment_deadline_hours is not None and availability_delay_hours is not None:
        deadline_met = availability_delay_hours <= payment_deadline_hours
        gates["funding_deadline_satisfied"] = deadline_met
        if not deadline_met:
            violated_gates.append("funding_deadline_satisfied")
            reason_codes.append(
                f"BREACH_FUNDING_DEADLINE: Proceeds arrive in {availability_delay_hours:.1f}h, "
                f"which is after payment deadline of {payment_deadline_hours:.1f}h."
            )
    else:
        gates["funding_deadline_satisfied"] = True

    # Hard Gate 6: Valid Bridge Repayment Source Gate
    if proposed_bridge_draw > 0.0 or action_type == "DRAW_BRIDGE_CREDIT":
        gate_passed = repayment_feasibility or allow_emergency_credit_draw
        gates["valid_repayment_source"] = gate_passed
        if not gate_passed:
            violated_gates.append("valid_repayment_source")
            reason_codes.append(
                "BREACH_REPAYMENT_SOURCE: Bridge facility draw lacks a verified future cash source for repayment."
            )
    else:
        gates["valid_repayment_source"] = True

    # Hard Gate 7: Full Funding vs Residual Gap Gate
    if is_partial_funding or residual_unfunded_amount > 0.0:
        # Action only partially funds the required payout
        gates["full_payout_funding"] = False
        violated_gates.append("full_payout_funding")
        reason_codes.append(
            f"BREACH_PARTIAL_FUNDING: Action leaves residual unfunded shortfall of ${residual_unfunded_amount:,.2f}."
        )
    else:
        gates["full_payout_funding"] = True

    # Hard Gate 8: Portfolio Exposure Cap Gate
    if resulting_tier_exposure_pct is not None and tier_portfolio_cap_pct is not None:
        cap_passed = resulting_tier_exposure_pct <= (tier_portfolio_cap_pct + 1e-4)
        gates["portfolio_exposure_limit"] = cap_passed
        if not cap_passed:
            violated_gates.append("portfolio_exposure_limit")
            reason_codes.append(
                f"BREACH_PORTFOLIO_EXPOSURE: Resulting Tier {target_tier} exposure "
                f"({resulting_tier_exposure_pct*100:.1f}%) exceeds portfolio limit ({tier_portfolio_cap_pct*100:.1f}%)."
            )
    else:
        gates["portfolio_exposure_limit"] = True

    all_passed = len(violated_gates) == 0

    return HardGateEvaluation(
        passed=all_passed,
        gates=gates,
        violated_gates=violated_gates,
        reason_codes=reason_codes,
        uncovered_scenarios=list(reserve_assessment.uncovered_stress_scenarios),
    )
