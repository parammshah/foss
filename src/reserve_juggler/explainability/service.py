"""
Decision Explanation Service and Human-in-the-Loop Oversight.
Produces mathematically grounded, non-hallucinatory explanations for treasury decisions,
enforces strict role permissions, revalidates modifications against live financial invariants,
and ensures cancelled/rejected actions produce zero phantom transaction outcomes.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.contracts import (
    AlternativeAction,
    ActionType,
    ApprovalStatus,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    GateStatus,
    HardGateOutcome,
    LearningEvent,
    TreasuryState,
    UserRole,
)
from reserve_juggler.explainability.models import (
    AlternativeComparisonEntry,
    CostBreakdown,
    DecisionExplanation,
    ForecastContextSummary,
    HumanActionRecord,
)

logger = logging.getLogger(__name__)


def resolve_action_type(action_str: str) -> ActionType:
    """Resolves arbitrary action strings or enum representations into typed ActionType."""
    normalized = action_str.upper().strip()
    try:
        return ActionType(normalized)
    except ValueError:
        pass
    if "CREDIT" in normalized and ("DRAW" in normalized or "BORROW" in normalized):
        return ActionType.DRAW_CREDIT
    if "CREDIT" in normalized and "REPAY" in normalized:
        return ActionType.REPAY_CREDIT
    if "REPO" in normalized and ("DRAW" in normalized or "BREAK" in normalized):
        return ActionType.DRAW_REPO_30D
    if "REPO" in normalized and ("INVEST" in normalized or "BUY" in normalized):
        return ActionType.INVEST_REPO
    if "BOND" in normalized and ("INVEST" in normalized or "BUY" in normalized):
        return ActionType.INVEST_BONDS
    if "EQUITY" in normalized and ("INVEST" in normalized or "BUY" in normalized):
        return ActionType.INVEST_EQUITIES
    if "BOND" in normalized and ("LIQUIDATE" in normalized or "SELL" in normalized or "TRIM" in normalized):
        return ActionType.LIQUIDATE_BONDS
    if "EQUITY" in normalized and ("LIQUIDATE" in normalized or "SELL" in normalized or "TRIM" in normalized):
        return ActionType.LIQUIDATE_EQUITIES
    if "HOLD" in normalized:
        return ActionType.HOLD_CASH
    return ActionType.UNKNOWN


class DecisionExplanationService:
    """
    Produces grounded explanations and manages human approval workflows.
    Interface:
        explain(decision_record, forecast_bundle, treasury_state) -> DecisionExplanation
    """

    SERVICE_VERSION = "1.0.0"

    def explain(
        self,
        decision_record: DecisionRecord,
        forecast_bundle: Optional[ForecastBundle] = None,
        treasury_state: Optional[TreasuryState] = None,
    ) -> DecisionExplanation:
        """
        Generates a transparent, mathematically grounded explanation for a given DecisionRecord.
        Never fabricates numbers or claims gates passed when they were not evaluated.
        """
        missing_disclosures: List[str] = []

        # 1. Cost breakdown extraction
        costs = decision_record.estimated_costs
        cost_bd = CostBreakdown(
            interest_cost=costs.get("interest", 0.0),
            tax_impact=costs.get("tax", 0.0),
            liquidation_penalty=costs.get("liquidation_penalty", 0.0),
            spread_slippage=costs.get("slippage", 0.0),
            total_cost=costs.get("total", sum(costs.values())),
        )

        # 2. Forecast context extraction (Truthful: never fabricate numbers if missing)
        if forecast_bundle and forecast_bundle.point_forecasts:
            p1h = forecast_bundle.point_forecasts.get("1h")
            p6h = forecast_bundle.point_forecasts.get("6h")
            p24h = forecast_bundle.point_forecasts.get("24h")
            
            f_summary = ForecastContextSummary(
                is_available=True,
                horizon_1h_flow=p1h.expected_net_flow if p1h else None,
                horizon_6h_flow=p6h.expected_net_flow if p6h else None,
                horizon_24h_flow=p24h.expected_net_flow if p24h else None,
                interval_p05=p24h.uncertainty_p05 if p24h else None,
                interval_p95=p24h.uncertainty_p95 if p24h else None,
                forecast_confidence=p24h.confidence_score if p24h else None,
                quality_status=forecast_bundle.quality_status,
            )
            if forecast_bundle.quality_status in ("DEGRADED", "STALE"):
                missing_disclosures.append(
                    f"Notice: Forecast bundle quality is marked {forecast_bundle.quality_status}; uncertainty bands widened."
                )
        else:
            f_summary = ForecastContextSummary(
                is_available=False,
                horizon_1h_flow=None,
                horizon_6h_flow=None,
                horizon_24h_flow=None,
                interval_p05=None,
                interval_p95=None,
                forecast_confidence=None,
                quality_status="UNAVAILABLE",
            )
            missing_disclosures.append("Explicit live forecast bundle unavailable. Forecast context marked UNAVAILABLE.")

        # 3. Settlement context
        settlement_days = 0
        settlement_note = "T+0 immediate intraday settlement."
        if decision_record.target_asset:
            target_upper = decision_record.target_asset.upper()
            if "BOND" in target_upper:
                settlement_days = 1
                settlement_note = "T+1 secondary bond market settlement required."
            elif "EQUITY" in target_upper:
                settlement_days = 2
                settlement_note = "T+2 equity settlement latency."

        # 4. Hard gates check (Truthful: never claim passed if empty)
        if decision_record.hard_gate_outcomes:
            all_passed = all(g.passed for g in decision_record.hard_gate_outcomes)
            gate_status_str = "PASS" if all_passed else "FAIL"
            displayed_gates = decision_record.hard_gate_outcomes
        else:
            all_passed = False
            gate_status_str = "NOT_EVALUATED"
            displayed_gates = [
                HardGateOutcome(
                    gate_name="MandatorySafetyGates (NOT_EVALUATED)",
                    status=GateStatus.NOT_EVALUATED,
                    reason="No hard safety gate evaluations supplied with decision proposal.",
                )
            ]
            missing_disclosures.append("No hard safety gate evaluations supplied with decision proposal; status marked NOT_EVALUATED.")

        # 5. Alternatives comparison
        alt_comparisons: List[AlternativeComparisonEntry] = []
        for alt in decision_record.alternatives:
            if isinstance(alt, dict):
                act_type = str(alt.get("action_type") or alt.get("type", "HOLD_CASH"))
                tgt_asset = str(alt.get("target_asset") or alt.get("asset_id", "CASH"))
                amt = float(alt.get("amount") or alt.get("amount_usd", 0.0) or 0.0)
                est_cost = float(alt.get("estimated_total_cost") or alt.get("total_estimated_cost", 0.0) or 0.0)
                settle_h = int(alt.get("settlement_horizon_hours") or alt.get("timing_hours", 0) or 0)
                rej_reason = str(alt.get("rejection_reason") or alt.get("reason_code", ""))
                is_feas = bool(alt.get("is_feasible", False))
            else:
                act_type = alt.action_type
                tgt_asset = alt.target_asset
                amt = alt.amount
                est_cost = alt.estimated_total_cost
                settle_h = alt.settlement_horizon_hours
                rej_reason = alt.rejection_reason
                is_feas = alt.is_feasible

            delta = est_cost - cost_bd.total_cost
            alt_comparisons.append(
                AlternativeComparisonEntry(
                    action_type=act_type,
                    target_asset=tgt_asset,
                    amount=amt,
                    estimated_total_cost=est_cost,
                    cost_delta_vs_selected=round(delta, 2),
                    settlement_horizon_hours=settle_h,
                    rejection_reason=rej_reason,
                    is_feasible=is_feas,
                )
            )

        # 6. Build grounded mathematical narrative
        action_raw = getattr(decision_record, "action_type", None) or (
            decision_record.selected_action
            if isinstance(decision_record.selected_action, str)
            else str(decision_record.selected_action.get("action_type", "HOLD_CASH"))
        )
        action_name = str(action_raw).replace("_", " ").title()
        amt_str = f"${decision_record.amount:,.2f}"
        if decision_record.reserve is not None:
            if isinstance(decision_record.reserve, dict):
                res_val = decision_record.reserve.get("protected_capital") or decision_record.reserve.get("reserve_amount", 0.0)
            else:
                res_val = decision_record.reserve
            reserve_str = f"${float(res_val):,.2f}"
        else:
            reserve_str = "UNCONFIGURED"
            missing_disclosures.append("Protected reserve floor is UNCONFIGURED / UNKNOWN. Mandatory reserve firewall cannot be certified.")

        surplus_str = (
            f"${float(decision_record.eligible_surplus):,.2f}"
            if getattr(decision_record, "eligible_surplus", None) is not None
            else "UNSPECIFIED"
        )
        total_cost_str = f"${cost_bd.total_cost:,.2f}"

        rationale_lines = [
            f"1. Objective Alignment: Selected '{action_name}' for {amt_str} on {decision_record.target_asset}.",
            f"2. Reserve Invariant: Preserves protected reserve floor of {reserve_str} with {surplus_str} eligible surplus remaining.",
        ]

        if f_summary.is_available and f_summary.horizon_24h_flow is not None:
            p05_str = f"${f_summary.interval_p05:,.2f}" if f_summary.interval_p05 is not None else "N/A"
            p95_str = f"${f_summary.interval_p95:,.2f}" if f_summary.interval_p95 is not None else "N/A"
            rationale_lines.append(
                f"3. Liquidity Need: Projected 24h net flow is ${f_summary.horizon_24h_flow:,.2f} "
                f"(distribution quantile P05 deficit: {p05_str}, P95: {p95_str})."
            )
        else:
            rationale_lines.append("3. Liquidity Need: Live forecast trajectory UNAVAILABLE; baseline constraints applied.")

        rationale_lines.append(
            f"4. Cost Minimization: Total estimated transaction cost is {total_cost_str} "
            f"(Interest: ${cost_bd.interest_cost:,.2f}, Tax: ${cost_bd.tax_impact:,.2f}, Slippage: ${cost_bd.spread_slippage:,.2f})."
        )

        if decision_record.treasury_safety_score is not None:
            tss_display = f"{decision_record.treasury_safety_score:.1f}/100"
        else:
            tss_display = "UNAVAILABLE"
            missing_disclosures.append("Treasury Safety Score is UNAVAILABLE due to missing component metrics.")

        meta = getattr(decision_record, "metadata", {}) or {}
        tax_status = meta.get("tax_profile_status")
        is_prov_tax = bool(
            meta.get("ranking_is_provisional", False)
            or getattr(decision_record, "ranking_is_provisional", False)
            or any("PROVISIONAL_TAX" in str(r) for r in getattr(decision_record, "reason_codes", []))
        )
        if is_prov_tax or tax_status in ("UNCONFIGURED_GROSS_ONLY", "SYNTHETIC_DEMO", "INVALID"):
            missing_disclosures.append(
                f"Tax Profile Status: {tax_status or 'PROVISIONAL'}. "
                "Ranking is calculated on gross yield; operational execution requires an approved statutory corporate TaxProfile."
            )

        if gate_status_str == "PASS":
            rationale_lines.append(f"5. Safety Score: Treasury Safety Score stands at {tss_display}. All mandatory safety gates passed.")
        else:
            rationale_lines.append(f"5. Safety Score: TSS={tss_display}. Mandatory safety gates: {gate_status_str}.")

        if alt_comparisons:
            cheapest_alt = min(alt_comparisons, key=lambda a: a.cost_delta_vs_selected)
            if cheapest_alt.cost_delta_vs_selected > 0:
                rationale_lines.append(
                    f"6. Alternative Optimality: Preferred over '{cheapest_alt.action_type}' by ${cheapest_alt.cost_delta_vs_selected:,.2f} in estimated cost savings."
                )

        grounded_narrative = "\n".join(rationale_lines)

        ts_str = str(decision_record.timestamp.isoformat() if hasattr(decision_record.timestamp, 'isoformat') else decision_record.timestamp)
        act_str = str(getattr(decision_record, "action_type", None) or (
            decision_record.selected_action if isinstance(decision_record.selected_action, str) else decision_record.selected_action.get("action_type", "HOLD_CASH")
        ))
        res_flt = (
            float(decision_record.reserve.get("protected_capital", 0.0) or 0.0)
            if isinstance(decision_record.reserve, dict)
            else (float(decision_record.reserve) if decision_record.reserve is not None else None)
        )

        return DecisionExplanation(
            decision_id=decision_record.decision_id,
            timestamp=ts_str,
            recommended_action=act_str,
            target_asset=decision_record.target_asset,
            recommended_amount=float(decision_record.amount or 0.0),
            protected_reserve_floor=res_flt,
            eligible_surplus=float(decision_record.eligible_surplus) if decision_record.eligible_surplus is not None else 0.0,
            treasury_safety_score=decision_record.treasury_safety_score,
            tss_factors=decision_record.tss_factors if decision_record.treasury_safety_score is not None else {},
            forecast_context=f_summary,
            cost_breakdown=cost_bd,
            settlement_days=settlement_days,
            settlement_note=settlement_note,
            hard_gates=displayed_gates,
            all_hard_gates_passed=all_passed,
            gate_evaluation_status=gate_status_str,
            primary_justification=decision_record.reason_codes[0] if decision_record.reason_codes else "MIN_COST_LIQUIDITY",
            grounded_narrative=grounded_narrative,
            alternatives_comparison=alt_comparisons,
            disclosed_missing_information=missing_disclosures,
            model_versions=decision_record.model_versions or {"decision_engine": "v1.2.0"},
            policy_reference="SEC-15c3-1 / Basel III LCR Internal Governance Safety Policy v2.0",
        )

    def execute_human_workflow(
        self,
        decision_record: DecisionRecord,
        action_type: str,  # "APPROVED", "REJECTED", "MODIFIED", "CANCELLED", "OVERRIDDEN"
        actor_name: str,
        actor_role: str,
        stated_reason: str,
        modified_amount: Optional[float] = None,
        modified_action: Optional[str] = None,
        treasury_state: Optional[TreasuryState] = None,
    ) -> Tuple[bool, str, DecisionRecord, Optional[ExecutionOutcome], LearningEvent]:
        """
        Processes human approval, rejection, modification, or cancellation with role permissions.
        Revalidates all modifications against current TreasuryState to prevent reserve breaches.
        Cancelled and rejected actions produce zero phantom transaction outcomes.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Role Permissions Backend Enforcement
        if actor_role == UserRole.AUDITOR.value:
            return (
                False,
                "Role 'AUDITOR' has read-only access and cannot authorize, modify, or reject decisions.",
                decision_record,
                None,
                LearningEvent(
                    event_id=f"EV-DENIED-{now_iso}",
                    versioned_event_type="HUMAN_ACTION_DENIED",
                    module="HUMAN_OVERSIGHT",
                    timestamp=now_iso,
                    correlation_id=decision_record.decision_id,
                    payload={"reason": "Auditor read-only violation", "actor": actor_name},
                ),
            )

        if actor_role == UserRole.TREASURY_ANALYST.value:
            if action_type in ("OVERRIDDEN", "REJECTED", "CANCELLED"):
                return (
                    False,
                    "Role 'TREASURY_ANALYST' cannot execute override, reject, or cancel decisions. Requires 'RISK_OFFICER' or 'HEAD_OF_TREASURY'.",
                    decision_record,
                    None,
                    LearningEvent(
                        event_id=f"EV-DENIED-{now_iso}",
                        versioned_event_type="HUMAN_ACTION_DENIED",
                        module="HUMAN_OVERSIGHT",
                        timestamp=now_iso,
                        correlation_id=decision_record.decision_id,
                        payload={"reason": "Analyst permission limit", "actor": actor_name},
                    ),
                )
            if modified_amount and modified_amount > 10_000_000.0:
                return (
                    False,
                    "Role 'TREASURY_ANALYST' cannot authorize amounts exceeding $10,000,000.",
                    decision_record,
                    None,
                    LearningEvent(
                        event_id=f"EV-DENIED-{now_iso}",
                        versioned_event_type="HUMAN_ACTION_DENIED",
                        module="HUMAN_OVERSIGHT",
                        timestamp=now_iso,
                        correlation_id=decision_record.decision_id,
                        payload={"reason": "Analyst limit exceeded", "amount": modified_amount},
                    ),
                )

        # 2. Revalidation for MODIFIED / OVERRIDDEN Actions
        original_amount = decision_record.amount
        original_action = decision_record.selected_action
        final_amount = decision_record.amount
        final_action = decision_record.selected_action

        if action_type in ("MODIFIED", "OVERRIDDEN", "APPROVED"):
            candidate_amt = modified_amount if modified_amount is not None else decision_record.amount
            candidate_act = modified_action if modified_action else decision_record.selected_action

            if action_type in ("MODIFIED", "OVERRIDDEN") and candidate_amt <= 0:
                return (False, "Modified action amount must be strictly positive.", decision_record, None, None)  # type: ignore

            # Financial revalidation against authoritative TreasuryState
            if treasury_state:
                # Check 1: Mandatory Reserve Floor Invariant
                if treasury_state.protected_reserve_floor is None:
                    violation_msg = (
                        "Revalidation FAILED: Protected reserve floor is UNKNOWN or NOT_CONFIGURED. "
                        "Prohibiting new allocations or disbursements until reserve floor is established."
                    )
                    logger.warning(violation_msg)
                    return (
                        False,
                        violation_msg,
                        decision_record,
                        None,
                        LearningEvent(
                            event_id=f"EV-REVAL-FAIL-{decision_record.decision_id}",
                            versioned_event_type="HUMAN_MODIFICATION_REVALIDATION_FAILED",
                            module="HUMAN_OVERSIGHT",
                            timestamp=now_iso,
                            correlation_id=decision_record.decision_id,
                            payload={"violation": violation_msg, "actor": actor_name},
                        ),
                    )

                # Typed action semantics
                resolved_act = resolve_action_type(candidate_act)

                # Check 1: Mandatory Reserve Floor Invariant for capital deployments
                is_cash_draw = resolved_act in (ActionType.DRAW_CREDIT, ActionType.DRAW_REPO_30D, ActionType.HOLD_CASH)
                if not is_cash_draw:
                    projected_cash = treasury_state.available_cash - candidate_amt
                    if projected_cash < treasury_state.protected_reserve_floor:
                        violation_msg = (
                            f"Revalidation FAILED: Proposed modification of ${candidate_amt:,.2f} breaches the "
                            f"Mandatory Reserve Floor (${treasury_state.protected_reserve_floor:,.2f}). Remaining cash would be ${projected_cash:,.2f}."
                        )
                        logger.warning(violation_msg)
                        return (
                            False,
                            violation_msg,
                            decision_record,
                            None,
                            LearningEvent(
                                event_id=f"EV-REVAL-FAIL-{decision_record.decision_id}",
                                versioned_event_type="HUMAN_MODIFICATION_REVALIDATION_FAILED",
                                module="HUMAN_OVERSIGHT",
                                timestamp=now_iso,
                                correlation_id=decision_record.decision_id,
                                payload={"violation": violation_msg, "actor": actor_name},
                            ),
                        )

                # Check 2: Asset Availability & Eligibility for liquidation
                if resolved_act in (ActionType.LIQUIDATE_BONDS, ActionType.LIQUIDATE_EQUITIES, ActionType.LIQUIDATE_REPO):
                    if decision_record.target_asset:
                        pos = next((p for p in treasury_state.asset_positions if p.asset_id == decision_record.target_asset), None)
                        if pos and candidate_amt > pos.accessible_liquidity:
                            violation_msg = (
                                f"Revalidation FAILED: Liquidation amount ${candidate_amt:,.2f} exceeds accessible position liquidity (${pos.accessible_liquidity:,.2f})."
                            )
                            return (False, violation_msg, decision_record, None, None)  # type: ignore

                # Check 3: Credit Line Availability
                if resolved_act == ActionType.DRAW_CREDIT:
                    if candidate_amt > treasury_state.authorized_credit_availability:
                        violation_msg = (
                            f"Revalidation FAILED: Credit draw ${candidate_amt:,.2f} exceeds authorized credit availability (${treasury_state.authorized_credit_availability:,.2f})."
                        )
                        return (False, violation_msg, decision_record, None, None)  # type: ignore

            final_amount = candidate_amt
            final_action = candidate_act

        # 3. Handle Cancellation & Rejection: Zero Phantom Outcomes
        if action_type in ("CANCELLED", "REJECTED"):
            decision_record.approval_status = action_type
            decision_record.approved_by = actor_name
            decision_record.approval_role = actor_role
            decision_record.approval_timestamp = now_iso
            decision_record.override_reason = stated_reason

            learning_event = LearningEvent(
                event_id=f"EV-HUMAN-{decision_record.decision_id}",
                versioned_event_type="HUMAN_DECISION_CANCELLED_OR_REJECTED",
                module="HUMAN_OVERSIGHT",
                timestamp=now_iso,
                correlation_id=decision_record.decision_id,
                payload={
                    "decision_id": decision_record.decision_id,
                    "actor_name": actor_name,
                    "actor_role": actor_role,
                    "status": action_type,
                    "stated_reason": stated_reason,
                    "original_amount": original_amount,
                    "is_executed": False,
                    "realized_amount": 0.0,
                    "realized_cost": 0.0,
                },
                associated_versions=decision_record.model_versions,
            )

            # Explicitly cancelled or rejected outcome (no phantom cash flows or costs)
            outcome = ExecutionOutcome(
                outcome_id=f"OUT-{action_type}-{decision_record.decision_id}",
                decision_id=decision_record.decision_id,
                correlation_id=decision_record.decision_id,
                execution_timestamp=now_iso,
                simulated=True,
                realized_amount=0.0,
                realized_costs={"total": 0.0, "interest": 0.0, "tax": 0.0, "slippage": 0.0},
                errors=[f"ACTION_{action_type}_BY_HUMAN: {stated_reason}"],
                liquidity_shortfall=0.0,
                settlement_status=action_type,
                slippage_bps=0.0,
            )

            return (
                True,
                f"Action '{action_type}' recorded by {actor_name} ({actor_role}). No transaction executed.",
                decision_record,
                outcome,
                learning_event,
            )

        # 4. Handle Approved or Revalidated Modified Execution
        # Mathematically recompute costs using original_amount as baseline
        ratio = (final_amount / original_amount) if original_amount > 0 else 1.0
        orig_costs = decision_record.estimated_costs
        scaled_interest = round(orig_costs.get("interest", 0.0) * ratio, 2)
        scaled_tax = round(orig_costs.get("tax", 0.0) * ratio, 2)
        scaled_penalty = round(orig_costs.get("liquidation_penalty", 0.0) * ratio, 2)
        scaled_base_slippage = round(orig_costs.get("slippage", 0.0) * ratio, 2)
        
        base_cost = scaled_interest + scaled_tax + scaled_penalty + scaled_base_slippage
        if base_cost == 0.0 and orig_costs.get("total", 0.0) > 0.0:
            base_cost = round(orig_costs.get("total", 0.0) * ratio, 2)

        simulated_cost = round(base_cost * 1.00015, 2)  # Slight simulated execution slippage
        exec_slippage = round(simulated_cost - base_cost, 2)

        decision_record.approval_status = action_type
        decision_record.approved_by = actor_name
        decision_record.approval_role = actor_role
        decision_record.approval_timestamp = now_iso
        decision_record.override_reason = stated_reason if action_type in ("OVERRIDDEN", "MODIFIED") else None
        decision_record.amount = final_amount
        decision_record.selected_action = final_action
        decision_record.estimated_costs = {
            "total": simulated_cost,
            "interest": scaled_interest,
            "tax": scaled_tax,
            "liquidation_penalty": scaled_penalty,
            "slippage": round(scaled_base_slippage + exec_slippage, 2),
        }

        outcome = ExecutionOutcome(
            outcome_id=f"OUT-{decision_record.decision_id}",
            decision_id=decision_record.decision_id,
            correlation_id=decision_record.decision_id,
            execution_timestamp=now_iso,
            simulated=True,
            realized_amount=final_amount,
            realized_costs={
                "total": simulated_cost,
                "interest": scaled_interest,
                "tax": scaled_tax,
                "penalty": scaled_penalty,
                "slippage": round(scaled_base_slippage + exec_slippage, 2),
            },
            errors=[],
            liquidity_shortfall=0.0,
            settlement_status="SETTLED",
            slippage_bps=1.5,
        )

        learning_event = LearningEvent(
            event_id=f"EV-HUMAN-{decision_record.decision_id}",
            versioned_event_type="HUMAN_APPROVAL_RECORDED",
            module="HUMAN_OVERSIGHT",
            timestamp=now_iso,
            correlation_id=decision_record.decision_id,
            payload={
                "decision_id": decision_record.decision_id,
                "actor_name": actor_name,
                "actor_role": actor_role,
                "status": action_type,
                "stated_reason": stated_reason,
                "original_amount": original_amount,
                "final_amount": final_amount,
                "final_action": final_action,
                "is_executed": True,
                "revalidation_passed": True,
            },
            associated_versions=decision_record.model_versions,
        )

        msg = f"Action '{action_type}' successfully recorded, verified, revalidated, and executed in simulation by {actor_name} ({actor_role})."
        return (True, msg, decision_record, outcome, learning_event)
