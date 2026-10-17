"""
Operations Research Treasury Optimizer and Time-Aware Next Best Action Engine.

Evaluates feasible actions, enforces hard constraints and safety gates,
accounts for financing APR, settlement delays, haircuts, taxes, and exposure caps,
and selects the cost-minimizing or after-tax net-value maximizing treasury action.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from reserve_juggler.decision.protocols import (
        AssetPosition,
        ForecastBundle,
        Obligation,
        RiskSignal,
        TreasuryState,
    )
from reserve_juggler.optimization.models import (
    ActionType,
    OptimizationResult,
    TierExposureLimits,
    TreasuryAction,
    get_tier_limits_for_tss,
)
from reserve_juggler.risk.capital_firewall import (
    calculate_holding_time_gate,
    classify_hot_core_capital,
)
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.models import ReserveAssessment, SafetyScoreResult
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value
from reserve_juggler.tax.tax_profile import IncomeType, TaxProfile

# Hours per year standard day-count convention
HOURS_PER_YEAR = 8760.0


def calculate_bridge_financing_cost(
    principal: float,
    apr: float,
    duration_hours: float,
    annual_convention: float = HOURS_PER_YEAR,
) -> float:
    """Calculate bridge credit financing interest cost.

    Benchmark Example:
    $10,000,000 principal at 12% APR (0.12) for 2 hours:
    Cost = 10,000,000 * 0.12 * (2 / 8760) = $273.9726...
    """
    if principal <= 0.0 or apr <= 0.0 or duration_hours <= 0.0:
        return 0.0
    return principal * apr * (duration_hours / annual_convention)


def optimize_treasury_actions(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    risk_signals: Optional[Sequence[RiskSignal]] = None,
    tax_profile: Optional[TaxProfile] = None,
    safety_score: Optional[SafetyScoreResult] = None,
    reserve_assessment: Optional[ReserveAssessment] = None,
    planning_horizon_hours: float = 24.0,
    repo_overnight_rate_apy: float = 0.043,  # Challenge indicative rate: ~4.3% APY
    repo_sweep_lockup_hours: float = 15.0,  # 17:00 entry to 08:00 maturity = 15h
    equity_slippage_haircut: float = 0.15,  # 15.0% liquidation slippage from dataset
) -> OptimizationResult:
    """Operations Research optimization engine determining the Next Best Action.

    Distinct Operating Modes:
    1. Emergency Funding Mode (immediate_cash_shortfall > 0):
       - Minimizes total all-in cost among plans that fully meet mandatory payouts by deadline.
       - Evaluates actual net proceeds, time until usable, residual gaps, and verified repayment sources.
       - Returns UNRESOLVED_FUNDING_SHORTFALL if no combination of actions can meet the deadline.
    2. Surplus Allocation Mode (immediate_cash_shortfall <= 0):
       - Maximizes expected after-tax net benefit minus quantified dollar risk costs.
       - Evaluates KEEP_CASH, PLACE_REPO, BUY_BOND, BUY_EQUITY.
       - Strictly enforces TSS-based tier caps and holding-time gatekeepers.
    """
    now = state.timestamp
    available_cash = max(0.0, float(state.available_cash))
    credit_avail = max(0.0, float(state.credit_availability))
    credit_apr = float(getattr(state, "credit_apr", 0.12))

    if safety_score is None:
        safety_score = calculate_treasury_safety_score(
            state, forecast_bundle, risk_signals=risk_signals
        )
    if reserve_assessment is None:
        reserve_assessment = assess_liquidity_reserve(
            state, forecast_bundle, risk_signals=risk_signals
        )

    tier_caps = get_tier_limits_for_tss(safety_score.composite_score)
    eligible_surplus = reserve_assessment.eligible_investable_surplus

    # Check for existing exposure breaches
    tier_breaches: list[str] = []
    remediation_notes: list[str] = []
    total_assets = available_cash + sum(float(p.market_value) for p in state.asset_positions)

    if total_assets > 0.0:
        tier2_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 2)
        tier3_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 3)

        tier2_ratio = tier2_val / total_assets
        tier3_ratio = tier3_val / total_assets

        if tier2_ratio > tier_caps.tier2_max_pct:
            tier_breaches.append(
                f"Tier 2 exposure ({tier2_ratio*100:.1f}%) exceeds TSS new-investment cap ({tier_caps.tier2_max_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 2 allocations; do NOT force fire-sale of existing bonds."
            )
        if tier3_ratio > tier_caps.tier3_max_pct:
            tier_breaches.append(
                f"Tier 3 exposure ({tier3_ratio*100:.1f}%) exceeds TSS new-investment cap ({tier_caps.tier3_max_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 3 allocations; rebalance orderly without distressed selling."
            )

    candidates: list[TreasuryAction] = []
    immediate_obligations = reserve_assessment.immediate_obligations
    cash_shortfall = max(0.0, immediate_obligations - available_cash)

    # =========================================================================
    # MODE 1: EMERGENCY FUNDING MODE (Cash Shortfall > 0)
    # =========================================================================
    if cash_shortfall > 0.0:
        optimization_mode = "EMERGENCY_FUNDING"

        # Determine earliest obligation deadline
        earliest_due_hours = 0.0
        earliest_found = False
        for ob in state.future_obligations:
            diff = (ob.due_time - now).total_seconds() / 3600.0
            if diff >= 0.0:
                if not earliest_found or diff < earliest_due_hours:
                    earliest_due_hours = diff
                    earliest_found = True

        repo_positions = [
            p
            for p in state.asset_positions
            if p.instrument_type.lower() == "repo" or p.tier == 1
        ]

        # -------------------------------------------------------------
        # 1. Candidate: DRAW_BRIDGE_CREDIT
        # -------------------------------------------------------------
        # Bridge draw can be evaluated up to authorized capacity
        bridge_draw_amount = min(cash_shortfall, credit_avail)

        if bridge_draw_amount > 0.0:
            # Check future cash sources for repayment
            # Repayment requires a verified future cash inflow (e.g. maturing repo)
            repayment_source_desc = ""
            repayment_time_hours: Optional[float] = None
            expected_repayment_proceeds = 0.0
            repayment_possible = False

            # Check if any repo matures and generates cash after payout but within a reasonable horizon
            for rp in repo_positions:
                if rp.maturity:
                    mat_hours = max(0.0, (rp.maturity - now).total_seconds() / 3600.0)
                    # Normal maturity proceeds (no early exit fee!)
                    proceeds = float(rp.market_value)

                    # Check competing obligations due before or at repo maturity
                    competing_obs = sum(
                        float(ob.amount)
                        for ob in state.future_obligations
                        if 0.0 <= (ob.due_time - now).total_seconds() / 3600.0 <= mat_hours
                    )
                    # Subtract immediate payout funded by this bridge
                    competing_beyond_immediate = max(0.0, competing_obs - immediate_obligations)
                    net_repayment_available = max(0.0, proceeds - competing_beyond_immediate)

                    if net_repayment_available >= bridge_draw_amount:
                        repayment_possible = True
                        repayment_time_hours = mat_hours
                        expected_repayment_proceeds = net_repayment_available
                        repayment_source_desc = f"Maturing Repo '{rp.asset_id}' at T+{mat_hours:.1f}h"
                        break

            # If no repo matures, cash from state or credit capacity cannot be used as proof of repayment!
            bridge_duration = repayment_time_hours if (repayment_possible and repayment_time_hours) else 2.0
            bridge_interest = calculate_bridge_financing_cost(
                bridge_draw_amount, credit_apr, bridge_duration
            )

            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                financing_costs=bridge_interest,
                tax_profile=tax_profile,
            )

            funded_amount = bridge_draw_amount
            residual_gap = max(0.0, cash_shortfall - funded_amount)
            is_partial = residual_gap > 0.0

            # Evaluate hard gate for this specific bridge draw
            gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                proposed_bridge_draw=bridge_draw_amount,
                payment_deadline_hours=earliest_due_hours,
                availability_delay_hours=0.0,  # Immediate draw
                repayment_feasibility=repayment_possible,
                is_partial_funding=is_partial,
                residual_unfunded_amount=residual_gap,
                action_type="DRAW_BRIDGE_CREDIT",
            )

            is_feasible = (
                gate_eval.passed
                and not is_partial
                and repayment_possible
            )

            credit_left = max(0.0, credit_avail - bridge_draw_amount)
            total_all_in_cost = bridge_interest - tax_res.tax_shield

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.DRAW_BRIDGE_CREDIT,
                    amount=bridge_draw_amount,
                    expected_net_value=-total_all_in_cost,
                    total_estimated_cost=bridge_interest,
                    risk_penalty=0.0,
                    is_feasible=is_feasible,
                    hard_gate_passed=gate_eval.passed,
                    cost_breakdown={
                        "financing_cost": bridge_interest,
                        "tax_shield": tax_res.tax_shield,
                        "penalty_cost": 0.0,
                    },
                    timing_hours=0.0,
                    repayment_feasibility=repayment_possible,
                    reason_code=(
                        f"DRAW_BRIDGE_CREDIT: Draw ${bridge_draw_amount:,.2f} at {credit_apr*100:.1f}% APR "
                        f"for {bridge_duration:.1f}h (interest ${bridge_interest:,.2f}). Repayment via {repayment_source_desc}."
                        if is_feasible
                        else f"REJECTED_BRIDGE: {'; '.join(gate_eval.reason_codes)}"
                    ),
                    required_payout_amount=immediate_obligations,
                    existing_cash_applied=available_cash,
                    gross_proceeds=bridge_draw_amount,
                    net_proceeds=bridge_draw_amount,
                    transaction_cost=0.0,
                    availability_delay_hours=0.0,
                    payment_deadline_hours=earliest_due_hours,
                    funded_by_deadline=funded_amount,
                    residual_unfunded_amount=residual_gap,
                    is_partial_funding=is_partial,
                    secondary_action_needed=is_partial,
                    principal_borrowed=bridge_draw_amount,
                    apr=credit_apr,
                    interest_cost=bridge_interest,
                    expected_repayment_time_hours=repayment_time_hours,
                    expected_repayment_proceeds=expected_repayment_proceeds,
                    repayment_source_desc=repayment_source_desc,
                    residual_debt_after_repayment=0.0 if repayment_possible else bridge_draw_amount,
                    credit_availability_after_draw=credit_left,
                    metadata={"credit_apr": credit_apr, "duration_hours": bridge_duration},
                )
            )

        # -------------------------------------------------------------
        # 2. Candidate: BREAK_REPO_EARLY
        # -------------------------------------------------------------
        for rp in repo_positions:
            nominal_val = float(rp.market_value)
            exit_fee_rate = float(rp.exit_fees) if float(rp.exit_fees) > 0.0 else 0.005
            settlement_delay = float(rp.settlement_delay_hours)

            # Net proceeds from liquidating repo: nominal * (1 - exit_fee_rate)
            # To net cash_shortfall, needed nominal is cash_shortfall / (1 - exit_fee_rate)
            needed_nominal = cash_shortfall / (1.0 - exit_fee_rate)
            break_nominal = min(nominal_val, needed_nominal)

            penalty = break_nominal * exit_fee_rate
            net_proceeds = break_nominal - penalty

            # If repo position covers needed nominal, funded amount is cash_shortfall
            if break_nominal >= needed_nominal - 1e-4:
                funded_amount = cash_shortfall
                residual_gap = 0.0
                action_amount = cash_shortfall
            else:
                funded_amount = net_proceeds
                residual_gap = max(0.0, cash_shortfall - funded_amount)
                action_amount = break_nominal

            is_partial = residual_gap > 0.01

            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                transaction_fees=penalty,
                tax_profile=tax_profile,
            )

            # Check timing against payment deadline
            timing_safe = settlement_delay <= earliest_due_hours

            gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                payment_deadline_hours=earliest_due_hours,
                availability_delay_hours=settlement_delay,
                is_partial_funding=is_partial,
                residual_unfunded_amount=residual_gap,
                action_type="BREAK_REPO_EARLY",
            )

            is_feasible = (
                gate_eval.passed
                and not is_partial
                and timing_safe
            )

            all_in_penalty_cost = penalty - tax_res.tax_shield

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.BREAK_REPO_EARLY,
                    asset_id=rp.asset_id,
                    amount=action_amount,
                    expected_net_value=-all_in_penalty_cost,
                    total_estimated_cost=penalty,
                    risk_penalty=0.0,  # Transparent monetary cost
                    is_feasible=is_feasible,
                    hard_gate_passed=gate_eval.passed,
                    cost_breakdown={
                        "penalty_cost": penalty,
                        "tax_shield": tax_res.tax_shield,
                        "financing_cost": 0.0,
                    },
                    timing_hours=settlement_delay,
                    repayment_feasibility=True,
                    reason_code=(
                        f"BREAK_REPO_EARLY: Liquidate repo '{rp.asset_id}' to generate ${net_proceeds:,.2f} net proceeds "
                        f"at {exit_fee_rate*100:.2f}% exit penalty (${penalty:,.2f})."
                        if is_feasible
                        else (
                            f"REJECTED_REPO_BREAK: Net proceeds ${net_proceeds:,.2f} leave residual gap ${residual_gap:,.2f}."
                            if is_partial
                            else f"REJECTED_REPO_BREAK: {'; '.join(gate_eval.reason_codes)}"
                        )
                    ),
                    required_payout_amount=immediate_obligations,
                    existing_cash_applied=available_cash,
                    gross_proceeds=break_nominal,
                    net_proceeds=net_proceeds,
                    transaction_cost=penalty,
                    availability_delay_hours=settlement_delay,
                    payment_deadline_hours=earliest_due_hours,
                    funded_by_deadline=funded_amount if timing_safe else 0.0,
                    residual_unfunded_amount=residual_gap,
                    is_partial_funding=is_partial,
                    secondary_action_needed=is_partial,
                    metadata={"exit_fee_rate": exit_fee_rate},
                )
            )

        # -------------------------------------------------------------
        # 3. Candidate: WAIT_REPO_MATURITY
        # -------------------------------------------------------------
        for rp in repo_positions:
            hours_to_mat = 0.0
            if rp.maturity:
                hours_to_mat = max(0.0, (rp.maturity - now).total_seconds() / 3600.0)

            # Check if waiting would cause a default on obligations due before repo matures
            violates_payout = hours_to_mat > earliest_due_hours and earliest_due_hours >= 0.0
            is_feasible = not violates_payout

            gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                payment_deadline_hours=earliest_due_hours,
                availability_delay_hours=hours_to_mat,
                action_type="WAIT_REPO_MATURITY",
            )

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.WAIT_REPO_MATURITY,
                    asset_id=rp.asset_id,
                    amount=float(rp.market_value),
                    expected_net_value=0.0 if is_feasible else -1e9,
                    total_estimated_cost=0.0 if is_feasible else 1e9,
                    is_feasible=is_feasible,
                    hard_gate_passed=is_feasible and gate_eval.passed,
                    timing_hours=hours_to_mat,
                    reason_code=(
                        f"WAIT_REPO_MATURITY: Repo matures in {hours_to_mat:.1f}h prior to obligation deadline."
                        if is_feasible
                        else f"REJECTED_WAIT: Repo matures in {hours_to_mat:.1f}h, which is AFTER payment deadline ({earliest_due_hours:.1f}h). Delaying payout is prohibited!"
                    ),
                    required_payout_amount=immediate_obligations,
                    existing_cash_applied=available_cash,
                    gross_proceeds=float(rp.market_value),
                    net_proceeds=float(rp.market_value),
                    availability_delay_hours=hours_to_mat,
                    payment_deadline_hours=earliest_due_hours,
                    funded_by_deadline=float(rp.market_value) if is_feasible else 0.0,
                    residual_unfunded_amount=0.0 if is_feasible else cash_shortfall,
                )
            )

        # -------------------------------------------------------------
        # 4. Candidate: SELL_BOND
        # -------------------------------------------------------------
        for bp in state.asset_positions:
            if bp.tier == 2:
                settlement_delay = float(bp.settlement_delay_hours)
                haircut_rate = float(bp.haircut)
                nominal_val = float(bp.market_value)
                net_proceeds = nominal_val * (1.0 - haircut_rate)
                haircut_loss = nominal_val * haircut_rate

                timing_safe = settlement_delay <= earliest_due_hours
                residual_gap = max(0.0, cash_shortfall - net_proceeds)
                is_partial = residual_gap > 0.0

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    payment_deadline_hours=earliest_due_hours,
                    availability_delay_hours=settlement_delay,
                    is_partial_funding=is_partial,
                    residual_unfunded_amount=residual_gap,
                    action_type="SELL_BOND",
                )

                is_feasible = gate_eval.passed and timing_safe and not is_partial

                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.SELL_BOND,
                        asset_id=bp.asset_id,
                        amount=nominal_val,
                        expected_net_value=-haircut_loss,
                        total_estimated_cost=haircut_loss,
                        is_feasible=is_feasible,
                        hard_gate_passed=gate_eval.passed,
                        cost_breakdown={"haircut_loss": haircut_loss},
                        timing_hours=settlement_delay,
                        reason_code=(
                            f"SELL_BOND: Liquidate sovereign bond position '{bp.asset_id}' to fund payout."
                            if is_feasible
                            else (
                                f"REJECTED_BOND_SETTLEMENT: Bond settlement ({settlement_delay:.1f}h) exceeds deadline ({earliest_due_hours:.1f}h)."
                                if not timing_safe
                                else f"REJECTED_BOND_PROCEEDS: Net proceeds ${net_proceeds:,.2f} leave shortfall ${residual_gap:,.2f}."
                            )
                        ),
                        required_payout_amount=immediate_obligations,
                        existing_cash_applied=available_cash,
                        gross_proceeds=nominal_val,
                        net_proceeds=net_proceeds,
                        transaction_cost=haircut_loss,
                        availability_delay_hours=settlement_delay,
                        payment_deadline_hours=earliest_due_hours,
                        funded_by_deadline=net_proceeds if timing_safe else 0.0,
                        residual_unfunded_amount=residual_gap,
                        is_partial_funding=is_partial,
                    )
                )

        # -------------------------------------------------------------
        # 5. Candidate: SELL_EQUITY
        # -------------------------------------------------------------
        for ep in state.asset_positions:
            if ep.tier == 3:
                settlement_delay = float(ep.settlement_delay_hours)
                # Specified emergency liquidation slippage: 15%
                slippage_rate = equity_slippage_haircut
                nominal_val = float(ep.market_value)
                slippage_loss = nominal_val * slippage_rate
                net_proceeds = nominal_val * (1.0 - slippage_rate)

                timing_safe = settlement_delay <= earliest_due_hours
                residual_gap = max(0.0, cash_shortfall - net_proceeds)
                is_partial = residual_gap > 0.0

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    payment_deadline_hours=earliest_due_hours,
                    availability_delay_hours=settlement_delay,
                    is_partial_funding=is_partial,
                    residual_unfunded_amount=residual_gap,
                    action_type="SELL_EQUITY",
                )

                is_feasible = gate_eval.passed and timing_safe and not is_partial

                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.SELL_EQUITY,
                        asset_id=ep.asset_id,
                        amount=nominal_val,
                        expected_net_value=-slippage_loss,
                        total_estimated_cost=slippage_loss,
                        is_feasible=is_feasible,
                        hard_gate_passed=gate_eval.passed,
                        cost_breakdown={"slippage_loss": slippage_loss},
                        timing_hours=settlement_delay,
                        reason_code=(
                            f"SELL_EQUITY: Liquidate equity '{ep.asset_id}' to fund payout."
                            if is_feasible
                            else (
                                f"REJECTED_EQUITY_SETTLEMENT: Equity settlement ({settlement_delay:.1f}h) exceeds deadline ({earliest_due_hours:.1f}h)."
                                if not timing_safe
                                else f"REJECTED_EQUITY_PROCEEDS: Net proceeds ${net_proceeds:,.2f} leave shortfall ${residual_gap:,.2f}."
                            )
                        ),
                        required_payout_amount=immediate_obligations,
                        existing_cash_applied=available_cash,
                        gross_proceeds=nominal_val,
                        net_proceeds=net_proceeds,
                        transaction_cost=slippage_loss,
                        availability_delay_hours=settlement_delay,
                        payment_deadline_hours=earliest_due_hours,
                        funded_by_deadline=net_proceeds if timing_safe else 0.0,
                        residual_unfunded_amount=residual_gap,
                        is_partial_funding=is_partial,
                    )
                )

        # -------------------------------------------------------------
        # EMERGENCY FUNDING SELECTION: MINIMIZE ALL-IN COST & SHORTFALL
        # -------------------------------------------------------------
        # 1. Candidates that fully fund the shortfall by deadline
        full_funding_candidates = [
            c for c in candidates
            if c.is_feasible and c.hard_gate_passed and c.residual_unfunded_amount == 0.0
        ]

        if full_funding_candidates:
            # Pick lowest total_estimated_cost
            selected_action = min(full_funding_candidates, key=lambda a: a.total_estimated_cost)
            objective_value = -selected_action.total_estimated_cost
            unresolved_shortfall = 0.0
            is_unresolved = False
        else:
            # 2. If no full funding candidate exists, find best partial funding candidate
            # that can release usable cash by the deadline
            partial_candidates = [
                c for c in candidates
                if c.funded_by_deadline > 0.0 and c.availability_delay_hours <= earliest_due_hours
            ]

            if partial_candidates:
                # Select the candidate that maximizes cash funded by deadline (minimizing residual gap)
                # and among ties, minimizes total estimated cost
                selected_action = max(
                    partial_candidates,
                    key=lambda a: (a.funded_by_deadline, -a.total_estimated_cost),
                )
                objective_value = -selected_action.total_estimated_cost
                unresolved_shortfall = selected_action.residual_unfunded_amount
                is_unresolved = True
            else:
                # 3. REGRESSION TEST C: EXPLICIT UNRESOLVED FUNDING SHORTFALL
                # No authorized action can meet the deadline
                selected_action = TreasuryAction(
                    action_type=ActionType.UNRESOLVED_FUNDING_SHORTFALL,
                    amount=cash_shortfall,
                    expected_net_value=-cash_shortfall,
                    total_estimated_cost=cash_shortfall,
                    is_feasible=False,
                    hard_gate_passed=False,
                    reason_code=(
                        f"CRITICAL_SHORTFALL: Unresolved funding shortfall of ${cash_shortfall:,.2f}. "
                        f"No authorized action can provide funds by the payment deadline of {earliest_due_hours:.1f}h."
                    ),
                    required_payout_amount=immediate_obligations,
                    existing_cash_applied=available_cash,
                    payment_deadline_hours=earliest_due_hours,
                    residual_unfunded_amount=cash_shortfall,
                    is_partial_funding=True,
                )
                objective_value = -cash_shortfall
                unresolved_shortfall = cash_shortfall
                is_unresolved = True

    # =========================================================================
    # MODE 2: SURPLUS ALLOCATION MODE (Cash Shortfall == 0)
    # =========================================================================
    else:
        optimization_mode = "SURPLUS_ALLOCATION"
        unresolved_shortfall = 0.0
        is_unresolved = False

        # 1. Base Action: KEEP_CASH
        candidates.append(
            TreasuryAction(
                action_type=ActionType.KEEP_CASH,
                amount=available_cash,
                expected_net_value=0.0,
                total_estimated_cost=0.0,
                risk_penalty=0.0,
                is_feasible=True,
                hard_gate_passed=True,
                timing_hours=0.0,
                reason_code="KEEP_CASH: Retain operational liquidity in primary treasury account.",
            )
        )

        if eligible_surplus > 0.0:
            # 2. Candidate: PLACE_REPO (Tier 1)
            repo_apr = repo_overnight_rate_apy
            lockup_h = repo_sweep_lockup_hours
            repo_gross = eligible_surplus * repo_apr * (lockup_h / HOURS_PER_YEAR)
            repo_tax_res = evaluate_tax_adjusted_value(
                gross_revenue=repo_gross,
                holding_days=lockup_h / 24.0,
                instrument_id="OVERNIGHT_REPO",
                tax_profile=tax_profile,
                explicit_income_type=IncomeType.INTEREST,
            )

            gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                proposed_investment_amount=eligible_surplus,
                proposed_lockup_hours=lockup_h,
                target_tier=1,
            )

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.PLACE_REPO,
                    amount=eligible_surplus,
                    expected_net_value=repo_tax_res.after_tax_net_value,
                    total_estimated_cost=0.0,
                    risk_penalty=0.0,  # Sovereign/repo near zero risk
                    is_feasible=gate_eval.passed,
                    hard_gate_passed=gate_eval.passed,
                    cost_breakdown={"tax_liability": repo_tax_res.tax_liability},
                    timing_hours=lockup_h,
                    reason_code=(
                        f"PLACE_REPO: Allocate ${eligible_surplus:,.2f} surplus into overnight repo at {repo_apr*100:.2f}% APY "
                        f"({lockup_h:.0f}h sweep window)."
                        if gate_eval.passed
                        else f"REJECTED_REPO_LOCKUP: {'; '.join(gate_eval.reason_codes)}"
                    ),
                    metadata={"annual_yield": repo_apr, "lockup_hours": lockup_h},
                )
            )

            # 3. Candidate: BUY_BOND (Tier 2)
            tier2_cap_amount = eligible_surplus * tier_caps.tier2_max_pct
            if tier2_cap_amount > 0.0:
                bond_yield = 0.048
                bond_haircut = 0.02
                expected_holding_days = 180.0
                bond_duration = 2.0

                class MockBond:
                    asset_id = "SOVEREIGN_2Y"
                    tier = 2
                    instrument_type = "bond"
                    haircut = bond_haircut
                    annual_yield = bond_yield

                holding_eval = calculate_holding_time_gate(
                    MockBond(), expected_holding_days=expected_holding_days
                )

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    proposed_investment_amount=tier2_cap_amount,
                    target_tier=2,
                )

                is_feasible = gate_eval.passed and holding_eval.passed

                bond_gross = tier2_cap_amount * bond_yield * (expected_holding_days / 365.0)
                bond_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=bond_gross,
                    holding_days=expected_holding_days,
                    instrument_id="SOVEREIGN_2Y",
                    tax_profile=tax_profile,
                    explicit_income_type=IncomeType.INTEREST,
                )

                # Quantified monetary duration risk cost: principal * duration * 50bps rate shock
                quantified_duration_risk_cost = tier2_cap_amount * bond_duration * 0.005

                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.BUY_BOND,
                        asset_id="SOVEREIGN_2Y",
                        amount=tier2_cap_amount,
                        expected_net_value=bond_tax_res.after_tax_net_value,
                        total_estimated_cost=0.0,
                        risk_penalty=quantified_duration_risk_cost,
                        is_feasible=is_feasible,
                        hard_gate_passed=is_feasible,
                        cost_breakdown={
                            "tax_liability": bond_tax_res.tax_liability,
                            "quantified_duration_risk": quantified_duration_risk_cost,
                        },
                        timing_hours=expected_holding_days * 24.0,
                        reason_code=(
                            f"BUY_BOND: Allocate ${tier2_cap_amount:,.2f} Core Capital into 2Y sovereign bonds "
                            f"within TSS Tier 2 cap ({tier_caps.tier2_max_pct*100:.0f}%)."
                            if is_feasible
                            else (
                                f"REJECTED_HOLDING_GATE: {holding_eval.notes}"
                                if not holding_eval.passed
                                else f"REJECTED_BOND_GATE: {'; '.join(gate_eval.reason_codes)}"
                            )
                        ),
                        metadata={"tier": 2, "holding_eval": holding_eval.to_dict()},
                    )
                )

            # 4. Candidate: BUY_EQUITY (Tier 3)
            tier3_cap_amount = eligible_surplus * tier_caps.tier3_max_pct
            if tier3_cap_amount > 0.0:
                equity_expected_return = 0.08
                equity_volatility = 0.18
                holding_days = 365.0

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    proposed_investment_amount=tier3_cap_amount,
                    target_tier=3,
                )

                equity_gross = tier3_cap_amount * equity_expected_return
                eq_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=equity_gross,
                    holding_days=holding_days,
                    instrument_id="EQUITY_INDEX",
                    tax_profile=tax_profile,
                    explicit_income_type=IncomeType.DIVIDEND,
                )

                # Quantified monetary volatility downside risk: principal * (volatility * sqrt(1/12))
                quantified_equity_risk_cost = tier3_cap_amount * (equity_volatility * 0.288)

                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.BUY_EQUITY,
                        asset_id="EQUITY_INDEX",
                        amount=tier3_cap_amount,
                        expected_net_value=eq_tax_res.after_tax_net_value,
                        total_estimated_cost=0.0,
                        risk_penalty=quantified_equity_risk_cost,
                        is_feasible=gate_eval.passed,
                        hard_gate_passed=gate_eval.passed,
                        cost_breakdown={
                            "tax_liability": eq_tax_res.tax_liability,
                            "quantified_volatility_risk": quantified_equity_risk_cost,
                        },
                        timing_hours=holding_days * 24.0,
                        reason_code=(
                            f"BUY_EQUITY: Allocate ${tier3_cap_amount:,.2f} Core Capital into growth equities "
                            f"within TSS Tier 3 cap ({tier_caps.tier3_max_pct*100:.0f}%)."
                            if gate_eval.passed
                            else f"REJECTED_EQUITY_GATE: {'; '.join(gate_eval.reason_codes)}"
                        ),
                        metadata={"tier": 3, "expected_return": equity_expected_return},
                    )
                )

        # -------------------------------------------------------------
        # SURPLUS ALLOCATION SELECTION: MAXIMIZE RISK-ADJUSTED VALUE
        # -------------------------------------------------------------
        feasible_alloc_candidates = [
            c for c in candidates if c.is_feasible and c.hard_gate_passed
        ]

        if not feasible_alloc_candidates:
            selected_action = candidates[0]  # KEEP_CASH
        else:
            def surplus_objective(act: TreasuryAction) -> float:
                return act.expected_net_value - act.total_estimated_cost - act.risk_penalty

            selected_action = max(feasible_alloc_candidates, key=surplus_objective)

        objective_value = (
            selected_action.expected_net_value
            - selected_action.total_estimated_cost
            - selected_action.risk_penalty
        )

    return OptimizationResult(
        selected_action=selected_action,
        candidate_actions=candidates,
        objective_value=objective_value,
        eligible_surplus=eligible_surplus,
        tier_caps_applied=tier_caps,
        tier_breaches_detected=tier_breaches,
        remediation_notes=remediation_notes,
        optimization_mode=optimization_mode,
        unresolved_shortfall=unresolved_shortfall,
        is_unresolved_shortfall=is_unresolved,
    )
