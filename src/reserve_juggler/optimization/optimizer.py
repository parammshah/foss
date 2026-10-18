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
    HumanOverride,
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
    repo_sweep_window_open: Optional[bool] = None,
    human_override: Optional[HumanOverride] = None,
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
    target_entity = getattr(state, "entity_id", None)
    target_jurisdiction = getattr(state, "jurisdiction", None)

    # Determine repo sweep window status
    if repo_sweep_window_open is not None:
        is_window_open = bool(repo_sweep_window_open)
    elif hasattr(state, "repo_sweep_window_open") and state.repo_sweep_window_open is not None:
        is_window_open = bool(state.repo_sweep_window_open)
    else:
        # Challenge illustrative rule: repo sweep entry window around 17:00 UTC
        is_window_open = (now.hour == 17)

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
    tier2_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 2)
    tier3_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 3)
    tier2_ratio = (tier2_val / total_assets) if total_assets > 0.0 else 0.0
    tier3_ratio = (tier3_val / total_assets) if total_assets > 0.0 else 0.0

    if total_assets > 0.0:
        if tier2_ratio > tier_caps.tier2_portfolio_cap_pct:
            tier_breaches.append(
                f"Tier 2 exposure ({tier2_ratio*100:.1f}%) exceeds hard portfolio cap ({tier_caps.tier2_portfolio_cap_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 2 allocations; do NOT force fire-sale of existing bonds."
            )
        elif tier2_ratio > tier_caps.tier2_max_pct:
            tier_breaches.append(
                f"Tier 2 exposure ({tier2_ratio*100:.1f}%) exceeds TSS new-investment cap ({tier_caps.tier2_max_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 2 allocations; do NOT force fire-sale of existing bonds."
            )

        if tier3_ratio > tier_caps.tier3_portfolio_cap_pct:
            tier_breaches.append(
                f"Tier 3 exposure ({tier3_ratio*100:.1f}%) exceeds hard portfolio cap ({tier_caps.tier3_portfolio_cap_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 3 allocations; rebalance orderly without distressed selling."
            )
        elif tier3_ratio > tier_caps.tier3_max_pct:
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
            repayment_source_desc = "NONE: No maturing repo or proceeds consumed by intervening obligations"
            repayment_time_hours: Optional[float] = None
            expected_repayment_proceeds = 0.0
            repayment_possible = False
            residual_debt = bridge_draw_amount
            intervening_obs_total = 0.0

            # Time-indexed cash flow evaluation for repayment:
            # Post-immediate payout cash
            cash_after_immediate_payout = max(0.0, available_cash + bridge_draw_amount - immediate_obligations)

            for rp in repo_positions:
                if rp.maturity:
                    mat_hours = max(0.0, (rp.maturity - now).total_seconds() / 3600.0)
                    repo_proceeds = float(rp.market_value)

                    # Intervening obligations due strictly after earliest_due_hours up to repo maturity
                    intervening_obs = sum(
                        float(ob.amount)
                        for ob in state.future_obligations
                        if earliest_due_hours < (ob.due_time - now).total_seconds() / 3600.0 <= mat_hours
                    )

                    # Cash available at repo maturity after paying intervening obligations
                    cash_available_at_mat = cash_after_immediate_payout + repo_proceeds - intervening_obs

                    # Bridge financing cost to repo maturity
                    int_cost_at_mat = calculate_bridge_financing_cost(bridge_draw_amount, credit_apr, mat_hours)
                    total_debt = bridge_draw_amount + int_cost_at_mat

                    # Total repo maturity proceeds (nominal plus normal accrued maturity yield)
                    repo_yield = repo_proceeds * float(getattr(rp, "annual_yield", 0.0)) * (mat_hours / HOURS_PER_YEAR)
                    total_repo_inflow = repo_proceeds + repo_yield
                    cash_available_at_mat = cash_after_immediate_payout + total_repo_inflow - intervening_obs

                    # Feasible if available cash covers the debt (or at least principal with negligible difference)
                    if cash_available_at_mat >= bridge_draw_amount - 1e-4:
                        repayment_possible = True
                        repayment_time_hours = mat_hours
                        expected_repayment_proceeds = total_repo_inflow
                        residual_debt = 0.0
                        intervening_obs_total = intervening_obs
                        repayment_source_desc = (
                            f"Maturing Repo '{rp.asset_id}' at T+{mat_hours:.1f}h "
                            f"(usable proceeds ${total_repo_inflow:,.2f} cover principal ${bridge_draw_amount:,.2f} "
                            f"after ${intervening_obs:,.2f} intervening obligations)"
                        )
                        break
                    else:
                        debt_gap = max(0.0, total_debt - max(0.0, cash_available_at_mat))
                        if debt_gap < residual_debt:
                            residual_debt = debt_gap
                            intervening_obs_total = intervening_obs
                            repayment_time_hours = mat_hours
                            expected_repayment_proceeds = total_repo_inflow
                            repayment_source_desc = (
                                f"UNVERIFIED: Maturing Repo '{rp.asset_id}' at T+{mat_hours:.1f}h leaves "
                                f"residual debt gap of ${debt_gap:,.2f} after intervening obligations (${intervening_obs:,.2f})"
                            )

            bridge_duration = repayment_time_hours if (repayment_time_hours is not None and repayment_time_hours > 0.0) else 2.0
            bridge_interest = calculate_bridge_financing_cost(
                bridge_draw_amount, credit_apr, bridge_duration
            )

            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                financing_costs=bridge_interest,
                tax_profile=tax_profile,
                target_entity_id=target_entity,
                target_jurisdiction=target_jurisdiction,
            )

            funded_amount = bridge_draw_amount
            residual_gap = max(0.0, cash_shortfall - funded_amount)
            is_partial = residual_gap > 0.01

            cand_bridge_id = f"BRIDGE_DRAW_{bridge_draw_amount:.0f}_{bridge_duration:.0f}H"
            override_matches = bool(
                human_override
                and (
                    human_override.candidate_id == cand_bridge_id
                    or human_override.candidate_id == ActionType.DRAW_BRIDGE_CREDIT.value
                    or human_override.candidate_id.startswith("BRIDGE_DRAW")
                )
            )

            # Hard safety gate evaluation
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
                allow_emergency_credit_draw=override_matches,
            )

            if override_matches:
                is_feasible = gate_eval.passed and not is_partial
            else:
                is_feasible = (
                    gate_eval.passed
                    and not is_partial
                    and repayment_possible
                )

            payout_funded = (funded_amount >= cash_shortfall - 1e-4) and (earliest_due_hours >= 0.0)
            repayment_plan_feasible = repayment_possible
            funding_plan_resolved = (payout_funded and (repayment_plan_feasible or override_matches))

            credit_left = max(0.0, credit_avail - bridge_draw_amount)
            total_all_in_cost = bridge_interest - tax_res.tax_shield

            candidates.append(
                TreasuryAction(
                    candidate_id=cand_bridge_id,
                    action_type=ActionType.DRAW_BRIDGE_CREDIT,
                    amount=bridge_draw_amount,
                    expected_net_value=-total_all_in_cost,
                    total_estimated_cost=bridge_interest,
                    risk_penalty=0.0,
                    is_feasible=is_feasible,
                    hard_gate_passed=gate_eval.passed,
                    payout_funded=payout_funded,
                    funding_plan_resolved=funding_plan_resolved,
                    repayment_plan_feasible=repayment_plan_feasible,
                    cost_breakdown={
                        "financing_cost": bridge_interest,
                        "tax_shield": tax_res.tax_shield,
                        "penalty_cost": 0.0,
                    },
                    timing_hours=0.0,
                    repayment_feasibility=repayment_possible,
                    reason_code=(
                        (
                            f"DRAW_BRIDGE_CREDIT: Draw ${bridge_draw_amount:,.2f} at {credit_apr*100:.1f}% APR "
                            f"for {bridge_duration:.1f}h (interest ${bridge_interest:,.2f}) authorized by human override "
                            f"({human_override.authorized_by}: {human_override.override_reason}). Repayment via {repayment_source_desc}."
                            if override_matches
                            else f"DRAW_BRIDGE_CREDIT: Draw ${bridge_draw_amount:,.2f} at {credit_apr*100:.1f}% APR "
                            f"for {bridge_duration:.1f}h (interest ${bridge_interest:,.2f}). Repayment via {repayment_source_desc}."
                        )
                        if is_feasible
                        else (
                            f"REJECTED_BRIDGE_REPAYMENT: Repayment funds consumed by intervening obligations "
                            f"or insufficient repo proceeds (residual debt ${residual_debt:,.2f})."
                            if not repayment_possible
                            else f"REJECTED_BRIDGE: {'; '.join(gate_eval.reason_codes)}"
                        )
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
                    payouts_due_before_repayment=intervening_obs_total,
                    residual_debt_after_repayment=residual_debt if not repayment_possible else 0.0,
                    credit_availability_after_draw=credit_left,
                    ranking_is_provisional=tax_res.ranking_is_provisional,
                    metadata={
                        "credit_apr": credit_apr,
                        "duration_hours": bridge_duration,
                        "can_cover_immediate": (bridge_draw_amount >= cash_shortfall),
                        "repayment_uncertain": not repayment_possible,
                        "human_override": human_override.to_dict() if override_matches else None,
                    },
                )
            )

        # -------------------------------------------------------------
        # 2. Candidate: BREAK_REPO_EARLY
        # -------------------------------------------------------------
        for rp in repo_positions:
            nominal_val = float(rp.market_value)
            if rp.exit_fees is not None:
                exit_fee_rate = max(0.0, float(rp.exit_fees))
            else:
                exit_fee_rate = 0.005  # Missing fallback default
            settlement_delay = float(rp.settlement_delay_hours)

            needed_nominal = cash_shortfall / (1.0 - exit_fee_rate) if exit_fee_rate < 1.0 else cash_shortfall
            break_nominal = min(nominal_val, needed_nominal)

            penalty = break_nominal * exit_fee_rate
            net_proceeds = break_nominal - penalty

            funded_amount = min(cash_shortfall, net_proceeds)
            residual_gap = max(0.0, cash_shortfall - funded_amount)
            is_partial = residual_gap > 0.01
            action_amount = break_nominal

            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                transaction_fees=penalty,
                tax_profile=tax_profile,
                target_entity_id=target_entity,
                target_jurisdiction=target_jurisdiction,
            )

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

            break_cand_id = f"BREAK_REPO_{rp.asset_id}_{action_amount:.0f}"
            break_payout_funded = (funded_amount >= cash_shortfall - 1e-4) and timing_safe
            candidates.append(
                TreasuryAction(
                    candidate_id=break_cand_id,
                    action_type=ActionType.BREAK_REPO_EARLY,
                    asset_id=rp.asset_id,
                    amount=action_amount,
                    expected_net_value=-all_in_penalty_cost,
                    total_estimated_cost=penalty,
                    risk_penalty=0.0,
                    is_feasible=is_feasible,
                    hard_gate_passed=gate_eval.passed,
                    payout_funded=break_payout_funded,
                    funding_plan_resolved=break_payout_funded and is_feasible,
                    repayment_plan_feasible=True,
                    cost_breakdown={
                        "penalty_cost": penalty,
                        "tax_shield": tax_res.tax_shield,
                        "financing_cost": 0.0,
                    },
                    timing_hours=settlement_delay,
                    repayment_feasibility=True,
                    reason_code=(
                        f"BREAK_REPO_EARLY: Liquidate gross ${break_nominal:,.2f} of repo '{rp.asset_id}' to generate ${net_proceeds:,.2f} net proceeds "
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
                    ranking_is_provisional=tax_res.ranking_is_provisional,
                    metadata={"exit_fee_rate": exit_fee_rate},
                )
            )

        # -------------------------------------------------------------
        # 3. Candidate: WAIT_REPO_MATURITY
        # -------------------------------------------------------------
        for rp in repo_positions:
            hours_to_mat = max(0.0, (rp.maturity - now).total_seconds() / 3600.0) if rp.maturity else 0.0
            repo_proceeds = float(rp.market_value)

            # Timing check: must mature at or before earliest payout deadline
            timing_safe = (hours_to_mat <= earliest_due_hours) if earliest_due_hours >= 0.0 else True

            # Competing obligations due before repo matures (consuming available cash before proceeds arrive)
            competing_before_mat = sum(
                float(ob.amount)
                for ob in state.future_obligations
                if 0.0 <= (ob.due_time - now).total_seconds() / 3600.0 < hours_to_mat
            )

            # Net usable repo proceeds after any competing obligations not covered by opening cash
            cash_deficit_before_mat = max(0.0, competing_before_mat - available_cash)
            usable_proceeds = max(0.0, repo_proceeds - cash_deficit_before_mat)

            funded_amount = min(cash_shortfall, usable_proceeds) if timing_safe else 0.0
            residual_gap = max(0.0, cash_shortfall - funded_amount)
            is_partial = residual_gap > 0.01

            gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                payment_deadline_hours=earliest_due_hours,
                availability_delay_hours=hours_to_mat,
                is_partial_funding=is_partial,
                residual_unfunded_amount=residual_gap,
                action_type="WAIT_REPO_MATURITY",
            )

            is_feasible = (
                timing_safe
                and not is_partial
                and gate_eval.passed
            )

            wait_cand_id = f"WAIT_REPO_{rp.asset_id}"
            wait_payout_funded = (funded_amount >= cash_shortfall - 1e-4) and timing_safe

            candidates.append(
                TreasuryAction(
                    candidate_id=wait_cand_id,
                    action_type=ActionType.WAIT_REPO_MATURITY,
                    asset_id=rp.asset_id,
                    amount=repo_proceeds,
                    expected_net_value=0.0 if is_feasible else -residual_gap,
                    total_estimated_cost=0.0 if is_feasible else residual_gap,
                    risk_penalty=0.0,
                    is_feasible=is_feasible,
                    hard_gate_passed=gate_eval.passed,
                    payout_funded=wait_payout_funded,
                    funding_plan_resolved=wait_payout_funded and is_feasible,
                    repayment_plan_feasible=True,
                    timing_hours=hours_to_mat,
                    reason_code=(
                        f"WAIT_REPO_MATURITY: Maturing repo '{rp.asset_id}' releases ${repo_proceeds:,.2f} at T+{hours_to_mat:.1f}h "
                        f"prior to payment deadline ({earliest_due_hours:.1f}h), fully funding the payout."
                        if is_feasible
                        else (
                            f"REJECTED_WAIT: Repo matures in {hours_to_mat:.1f}h, which is AFTER payment deadline ({earliest_due_hours:.1f}h). Delaying payout is prohibited!"
                            if not timing_safe
                            else f"REJECTED_WAIT_PROCEEDS: Repo proceeds ${repo_proceeds:,.2f} leave residual gap ${residual_gap:,.2f}. Additional funding required."
                        )
                    ),
                    required_payout_amount=immediate_obligations,
                    existing_cash_applied=available_cash,
                    gross_proceeds=repo_proceeds,
                    net_proceeds=repo_proceeds,
                    availability_delay_hours=hours_to_mat,
                    payment_deadline_hours=earliest_due_hours,
                    funded_by_deadline=funded_amount,
                    residual_unfunded_amount=residual_gap,
                    is_partial_funding=is_partial,
                    secondary_action_needed=is_partial,
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

                needed_nominal = cash_shortfall / (1.0 - haircut_rate) if haircut_rate < 1.0 else cash_shortfall
                sell_nominal = min(nominal_val, needed_nominal)
                haircut_loss = sell_nominal * haircut_rate
                net_proceeds = sell_nominal - haircut_loss

                timing_safe = settlement_delay <= earliest_due_hours
                funded_amount = min(cash_shortfall, net_proceeds)
                residual_gap = max(0.0, cash_shortfall - funded_amount)
                is_partial = residual_gap > 0.01

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
                bond_cand_id = f"SELL_BOND_{bp.asset_id}_{sell_nominal:.0f}"
                bond_payout_funded = (funded_amount >= cash_shortfall - 1e-4) and timing_safe

                candidates.append(
                    TreasuryAction(
                        candidate_id=bond_cand_id,
                        action_type=ActionType.SELL_BOND,
                        asset_id=bp.asset_id,
                        amount=sell_nominal,
                        expected_net_value=-haircut_loss,
                        total_estimated_cost=haircut_loss,
                        risk_penalty=0.0,
                        is_feasible=is_feasible,
                        hard_gate_passed=gate_eval.passed,
                        payout_funded=bond_payout_funded,
                        funding_plan_resolved=bond_payout_funded and is_feasible,
                        repayment_plan_feasible=True,
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
                        gross_proceeds=sell_nominal,
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
                slippage_rate = equity_slippage_haircut
                nominal_val = float(ep.market_value)

                needed_nominal = cash_shortfall / (1.0 - slippage_rate) if slippage_rate < 1.0 else cash_shortfall
                sell_nominal = min(nominal_val, needed_nominal)
                slippage_loss = sell_nominal * slippage_rate
                net_proceeds = sell_nominal - slippage_loss

                timing_safe = settlement_delay <= earliest_due_hours
                funded_amount = min(cash_shortfall, net_proceeds)
                residual_gap = max(0.0, cash_shortfall - funded_amount)
                is_partial = residual_gap > 0.01

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
                eq_cand_id = f"SELL_EQUITY_{ep.asset_id}_{sell_nominal:.0f}"
                eq_payout_funded = (funded_amount >= cash_shortfall - 1e-4) and timing_safe

                candidates.append(
                    TreasuryAction(
                        candidate_id=eq_cand_id,
                        action_type=ActionType.SELL_EQUITY,
                        asset_id=ep.asset_id,
                        amount=sell_nominal,
                        expected_net_value=-slippage_loss,
                        total_estimated_cost=slippage_loss,
                        risk_penalty=0.0,
                        is_feasible=is_feasible,
                        hard_gate_passed=gate_eval.passed,
                        payout_funded=eq_payout_funded,
                        funding_plan_resolved=eq_payout_funded and is_feasible,
                        repayment_plan_feasible=True,
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
                        gross_proceeds=sell_nominal,
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
        # 0. Check for explicit human override
        override_candidate = None
        if human_override:
            for c in candidates:
                if (
                    c.candidate_id == human_override.candidate_id
                    or c.action_type.value == human_override.candidate_id
                    or (human_override.candidate_id.startswith("BRIDGE_DRAW") and c.action_type == ActionType.DRAW_BRIDGE_CREDIT)
                ):
                    override_candidate = c
                    break

        if override_candidate is not None:
            selected_action = override_candidate
            objective_value = -selected_action.total_estimated_cost
            unresolved_shortfall = selected_action.residual_unfunded_amount
            is_unresolved = not selected_action.funding_plan_resolved
        else:
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
                is_unresolved = not selected_action.funding_plan_resolved
            else:
                # 2. If no full funding candidate exists, find best partial funding candidate
                # that can release usable cash by the deadline
                partial_candidates = [
                    c for c in candidates
                    if c.funded_by_deadline > 0.0 and c.availability_delay_hours <= earliest_due_hours
                ]

                if partial_candidates:
                    selected_action = max(
                        partial_candidates,
                        key=lambda a: (a.funded_by_deadline, -a.total_estimated_cost),
                    )
                    objective_value = -selected_action.total_estimated_cost
                    unresolved_shortfall = selected_action.residual_unfunded_amount
                    is_unresolved = True
                else:
                    # 3. REGRESSION TEST C: EXPLICIT UNRESOLVED FUNDING SHORTFALL
                    selected_action = TreasuryAction(
                        candidate_id=f"UNRESOLVED_SHORTFALL_{cash_shortfall:.0f}",
                        action_type=ActionType.UNRESOLVED_FUNDING_SHORTFALL,
                        amount=cash_shortfall,
                        expected_net_value=-cash_shortfall,
                        total_estimated_cost=cash_shortfall,
                        is_feasible=False,
                        hard_gate_passed=False,
                        payout_funded=False,
                        funding_plan_resolved=False,
                        repayment_plan_feasible=False,
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
        keep_cash_cand_id = f"KEEP_CASH_{now.strftime('%Y%m%d_%H%M')}"
        candidates.append(
            TreasuryAction(
                candidate_id=keep_cash_cand_id,
                action_type=ActionType.KEEP_CASH,
                amount=available_cash,
                expected_net_value=0.0,
                total_estimated_cost=0.0,
                risk_penalty=0.0,
                is_feasible=True,
                hard_gate_passed=True,
                payout_funded=True,
                funding_plan_resolved=True,
                repayment_plan_feasible=True,
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
                target_entity_id=target_entity,
                target_jurisdiction=target_jurisdiction,
            )

            if not is_window_open:
                repo_feasible = False
                gate_passed = False
                repo_reason = (
                    f"REJECTED_WINDOW_CLOSED: Repo sweep entry window is closed at {now.strftime('%H:%M UTC')}. "
                    f"Entry window opens around 17:00 UTC."
                )
            else:
                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    proposed_investment_amount=eligible_surplus,
                    proposed_lockup_hours=lockup_h,
                    target_tier=1,
                    action_type="PLACE_REPO",
                )
                repo_feasible = gate_eval.passed
                gate_passed = gate_eval.passed
                repo_reason = (
                    f"PLACE_REPO: Allocate ${eligible_surplus:,.2f} surplus into overnight repo at {repo_apr*100:.2f}% APY "
                    f"({lockup_h:.0f}h sweep window)."
                    if gate_eval.passed
                    else f"REJECTED_REPO_LOCKUP: {'; '.join(gate_eval.reason_codes)}"
                )

            place_repo_cand_id = f"PLACE_REPO_{eligible_surplus:.0f}_{lockup_h:.0f}H"
            candidates.append(
                TreasuryAction(
                    candidate_id=place_repo_cand_id,
                    action_type=ActionType.PLACE_REPO,
                    asset_id="T1_REPO_PORTFOLIO",
                    amount=eligible_surplus,
                    expected_net_value=repo_tax_res.after_tax_net_value,
                    total_estimated_cost=0.0,
                    risk_penalty=0.0,  # Sovereign/repo near zero risk
                    is_feasible=repo_feasible,
                    hard_gate_passed=gate_passed,
                    payout_funded=True,
                    funding_plan_resolved=True,
                    repayment_plan_feasible=True,
                    cost_breakdown={"tax_liability": repo_tax_res.tax_liability},
                    timing_hours=lockup_h,
                    reason_code=repo_reason,
                    ranking_is_provisional=repo_tax_res.ranking_is_provisional,
                    metadata={"annual_yield": repo_apr, "lockup_hours": lockup_h, "window_open": is_window_open},
                )
            )

            # Check Core Capital eligibility from Capital Firewall
            cap_classification = classify_hot_core_capital(state, forecast_bundle, reserve_assessment)
            core_capital_available = cap_classification.investable_core_capital > 0.0

            # 3. Candidate: BUY_BOND (Tier 2)
            tier2_cap_amount = eligible_surplus * tier_caps.tier2_max_pct
            bond_cand_id = f"BUY_BOND_{tier2_cap_amount:.0f}"

            if tier2_ratio >= tier_caps.tier2_portfolio_cap_pct:
                bond_feasible = False
                bond_gate_passed = False
                bond_reason = (
                    f"REJECTED_PORTFOLIO_EXPOSURE_BREACH: Existing Tier 2 exposure (${tier2_val:,.2f}, {tier2_ratio*100:.1f}%) "
                    f"exceeds hard portfolio cap ({tier_caps.tier2_portfolio_cap_pct*100:.1f}%). New Tier 2 purchases are frozen."
                )
                bond_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=0.0,
                    tax_profile=tax_profile,
                )
                candidates.append(
                    TreasuryAction(
                        candidate_id=bond_cand_id,
                        action_type=ActionType.BUY_BOND,
                        asset_id="SOVEREIGN_2Y",
                        amount=tier2_cap_amount,
                        expected_net_value=0.0,
                        total_estimated_cost=0.0,
                        risk_penalty=0.0,
                        is_feasible=False,
                        hard_gate_passed=False,
                        payout_funded=True,
                        funding_plan_resolved=False,
                        repayment_plan_feasible=True,
                        timing_hours=180.0 * 24.0,
                        reason_code=bond_reason,
                        ranking_is_provisional=bond_tax_res.ranking_is_provisional,
                        metadata={"tier": 2, "portfolio_exposure_pct": tier2_ratio},
                    )
                )
            elif not core_capital_available:
                candidates.append(
                    TreasuryAction(
                        candidate_id=bond_cand_id,
                        action_type=ActionType.BUY_BOND,
                        asset_id="SOVEREIGN_2Y",
                        amount=tier2_cap_amount,
                        expected_net_value=0.0,
                        total_estimated_cost=0.0,
                        risk_penalty=0.0,
                        is_feasible=False,
                        hard_gate_passed=False,
                        payout_funded=True,
                        funding_plan_resolved=False,
                        repayment_plan_feasible=True,
                        timing_hours=180.0 * 24.0,
                        reason_code=f"REJECTED_CORE_CAPITAL: {cap_classification.rationale}",
                        ranking_is_provisional=(tax_profile is None),
                        metadata={"tier": 2, "core_capital_available": False},
                    )
                )
            elif tier2_cap_amount > 0.0:
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

                # Cap amount to headroom under portfolio cap
                headroom_tier2 = max(0.0, total_assets * tier_caps.tier2_portfolio_cap_pct - tier2_val)
                actual_bond_purchase = min(tier2_cap_amount, headroom_tier2)
                resulting_ratio2 = (tier2_val + actual_bond_purchase) / total_assets if total_assets > 0 else 0.0

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    proposed_investment_amount=actual_bond_purchase,
                    proposed_lockup_hours=expected_holding_days * 24.0,
                    target_tier=2,
                    resulting_tier_exposure_pct=resulting_ratio2,
                    tier_portfolio_cap_pct=tier_caps.tier2_portfolio_cap_pct,
                    action_type="BUY_BOND",
                )

                is_feasible = gate_eval.passed and holding_eval.passed and actual_bond_purchase > 0.0

                bond_gross = actual_bond_purchase * bond_yield * (expected_holding_days / 365.0)
                bond_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=bond_gross,
                    holding_days=expected_holding_days,
                    instrument_id="SOVEREIGN_2Y",
                    tax_profile=tax_profile,
                    explicit_income_type=IncomeType.INTEREST,
                    target_entity_id=target_entity,
                    target_jurisdiction=target_jurisdiction,
                )

                # Quantified monetary duration risk cost: principal * duration * 50bps rate shock
                quantified_duration_risk_cost = actual_bond_purchase * bond_duration * 0.005

                candidates.append(
                    TreasuryAction(
                        candidate_id=f"BUY_BOND_{actual_bond_purchase:.0f}",
                        action_type=ActionType.BUY_BOND,
                        asset_id="SOVEREIGN_2Y",
                        amount=actual_bond_purchase,
                        expected_net_value=bond_tax_res.after_tax_net_value,
                        total_estimated_cost=0.0,
                        risk_penalty=quantified_duration_risk_cost,
                        is_feasible=is_feasible,
                        hard_gate_passed=is_feasible,
                        payout_funded=True,
                        funding_plan_resolved=is_feasible,
                        repayment_plan_feasible=True,
                        cost_breakdown={
                            "tax_liability": bond_tax_res.tax_liability,
                            "quantified_duration_risk": quantified_duration_risk_cost,
                        },
                        timing_hours=expected_holding_days * 24.0,
                        reason_code=(
                            f"BUY_BOND: Allocate ${actual_bond_purchase:,.2f} Core Capital into 2Y sovereign bonds "
                            f"within TSS Tier 2 cap ({tier_caps.tier2_max_pct*100:.0f}% surplus, {tier_caps.tier2_portfolio_cap_pct*100:.0f}% portfolio)."
                            if is_feasible
                            else (
                                f"REJECTED_HOLDING_GATE: {holding_eval.notes}"
                                if not holding_eval.passed
                                else f"REJECTED_BOND_GATE: {'; '.join(gate_eval.reason_codes)}"
                            )
                        ),
                        ranking_is_provisional=bond_tax_res.ranking_is_provisional,
                        metadata={"tier": 2, "holding_eval": holding_eval.to_dict(), "resulting_exposure": resulting_ratio2},
                    )
                )

            # 4. Candidate: BUY_EQUITY (Tier 3)
            tier3_cap_amount = eligible_surplus * tier_caps.tier3_max_pct
            eq_cand_id = f"BUY_EQUITY_{tier3_cap_amount:.0f}"

            if not core_capital_available:
                candidates.append(
                    TreasuryAction(
                        candidate_id=eq_cand_id,
                        action_type=ActionType.BUY_EQUITY,
                        asset_id="EQUITY_INDEX",
                        amount=tier3_cap_amount,
                        expected_net_value=0.0,
                        total_estimated_cost=0.0,
                        risk_penalty=0.0,
                        is_feasible=False,
                        hard_gate_passed=False,
                        payout_funded=True,
                        funding_plan_resolved=False,
                        repayment_plan_feasible=True,
                        timing_hours=365.0 * 24.0,
                        reason_code=f"REJECTED_CORE_CAPITAL: {cap_classification.rationale}",
                        metadata={"tier": 3, "core_capital_available": False},
                    )
                )
            elif tier3_ratio >= tier_caps.tier3_portfolio_cap_pct:
                eq_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=0.0,
                    tax_profile=tax_profile,
                )
                candidates.append(
                    TreasuryAction(
                        candidate_id=eq_cand_id,
                        action_type=ActionType.BUY_EQUITY,
                        asset_id="EQUITY_INDEX",
                        amount=tier3_cap_amount,
                        expected_net_value=0.0,
                        total_estimated_cost=0.0,
                        risk_penalty=0.0,
                        is_feasible=False,
                        hard_gate_passed=False,
                        payout_funded=True,
                        funding_plan_resolved=False,
                        repayment_plan_feasible=True,
                        timing_hours=365.0 * 24.0,
                        reason_code=(
                            f"REJECTED_PORTFOLIO_EXPOSURE_BREACH: Existing Tier 3 exposure (${tier3_val:,.2f}, {tier3_ratio*100:.1f}%) "
                            f"exceeds hard portfolio cap ({tier_caps.tier3_portfolio_cap_pct*100:.1f}%). New Tier 3 purchases are frozen."
                        ),
                        ranking_is_provisional=eq_tax_res.ranking_is_provisional,
                        metadata={"tier": 3, "portfolio_exposure_pct": tier3_ratio},
                    )
                )
            elif tier3_cap_amount > 0.0:
                equity_expected_return = 0.08
                equity_volatility = 0.18
                holding_days = 365.0

                headroom_tier3 = max(0.0, total_assets * tier_caps.tier3_portfolio_cap_pct - tier3_val)
                actual_eq_purchase = min(tier3_cap_amount, headroom_tier3)
                resulting_ratio3 = (tier3_val + actual_eq_purchase) / total_assets if total_assets > 0 else 0.0

                gate_eval = evaluate_hard_safety_gates(
                    state=state,
                    reserve_assessment=reserve_assessment,
                    proposed_investment_amount=actual_eq_purchase,
                    proposed_lockup_hours=holding_days * 24.0,
                    target_tier=3,
                    resulting_tier_exposure_pct=resulting_ratio3,
                    tier_portfolio_cap_pct=tier_caps.tier3_portfolio_cap_pct,
                    action_type="BUY_EQUITY",
                )

                equity_gross = actual_eq_purchase * equity_expected_return
                eq_tax_res = evaluate_tax_adjusted_value(
                    gross_revenue=equity_gross,
                    holding_days=holding_days,
                    instrument_id="EQUITY_INDEX",
                    tax_profile=tax_profile,
                    explicit_income_type=IncomeType.DIVIDEND,
                    target_entity_id=target_entity,
                    target_jurisdiction=target_jurisdiction,
                )

                # Quantified monetary volatility downside risk: principal * (volatility * sqrt(1/12))
                quantified_equity_risk_cost = actual_eq_purchase * (equity_volatility * 0.288)

                is_feasible = gate_eval.passed and actual_eq_purchase > 0.0

                candidates.append(
                    TreasuryAction(
                        candidate_id=f"BUY_EQUITY_{actual_eq_purchase:.0f}",
                        action_type=ActionType.BUY_EQUITY,
                        asset_id="EQUITY_INDEX",
                        amount=actual_eq_purchase,
                        expected_net_value=eq_tax_res.after_tax_net_value,
                        total_estimated_cost=0.0,
                        risk_penalty=quantified_equity_risk_cost,
                        is_feasible=is_feasible,
                        hard_gate_passed=is_feasible,
                        payout_funded=True,
                        funding_plan_resolved=is_feasible,
                        repayment_plan_feasible=True,
                        cost_breakdown={
                            "tax_liability": eq_tax_res.tax_liability,
                            "quantified_volatility_risk": quantified_equity_risk_cost,
                        },
                        timing_hours=holding_days * 24.0,
                        reason_code=(
                            f"BUY_EQUITY: Allocate ${actual_eq_purchase:,.2f} Core Capital into growth equities "
                            f"within TSS Tier 3 cap ({tier_caps.tier3_max_pct*100:.0f}% surplus, {tier_caps.tier3_portfolio_cap_pct*100:.0f}% portfolio)."
                            if is_feasible
                            else f"REJECTED_EQUITY_GATE: {'; '.join(gate_eval.reason_codes)}"
                        ),
                        ranking_is_provisional=eq_tax_res.ranking_is_provisional,
                        metadata={"tier": 3, "expected_return": equity_expected_return, "resulting_exposure": resulting_ratio3},
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

    is_provisional_tax = bool(
        tax_profile is None
        or getattr(getattr(tax_profile, "jurisdiction_rules", None), "is_synthetic", False)
        or any(c.ranking_is_provisional for c in candidates)
    )
    if is_provisional_tax:
        selected_action.ranking_is_provisional = True
        remediation_notes.append(
            "PROVISIONAL_TAX_ASSUMPTIONS: Missing, unverified, or synthetic tax profile utilized. "
            "Optimization ranking is provisional (gross yield or demo rules); human approval required prior to trade execution."
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
        human_override=human_override,
    )
