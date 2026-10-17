"""
Treasury Decision Service orchestration module.

Integrates Safety Scoring, Liquidity Shield, Capital Firewall, Tax Engine,
and Operations Research Optimizer into the canonical decide(...) service.
Publishes audit and learning events for Team 3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

from reserve_juggler.decision.events import (
    EventPublisher,
    InMemoryEventPublisher,
    StandardLearningEvent,
)
from reserve_juggler.decision.models import StandardDecisionRecord
from reserve_juggler.decision.protocols import (
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    LearningEvent,
    RiskSignal,
    TreasuryState,
)
from reserve_juggler.optimization.models import ActionType
from reserve_juggler.optimization.optimizer import optimize_treasury_actions
from reserve_juggler.risk.capital_firewall import classify_hot_core_capital
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_profile import TaxProfile


class TreasuryDecisionService:
    """Enterprise decision engine orchestrating treasury safety and capital optimization."""

    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        policy_version: str = "2.0.0",
        model_version: str = "1.4.0",
    ) -> None:
        self.publisher = event_publisher or InMemoryEventPublisher()
        self.policy_version = policy_version
        self.model_version = model_version

    def _publish_event(self, event: StandardLearningEvent) -> None:
        """Publish event supporting both EventPublisher protocol and raw callables."""
        if hasattr(self.publisher, "publish") and callable(getattr(self.publisher, "publish")):
            self.publisher.publish(event)
        elif callable(self.publisher):
            self.publisher(event)

    def decide(
        self,
        state: TreasuryState,
        forecast_bundle: ForecastBundle,
        risk_signals: Optional[Sequence[RiskSignal]] = None,
        tax_profile: Optional[TaxProfile] = None,
        repo_overnight_rate_apy: float = 0.043,
        repo_sweep_lockup_hours: float = 15.0,
    ) -> DecisionRecord:
        """Evaluate treasury condition and produce the Next Best Action decision record.

        Steps:
        1. Calculate Treasury Safety Score (TSS) composite and component breakdown.
        2. Assess dynamic liquidity reserves (Protected Capital, Future-Obligation Reserve, Eligible Surplus).
        3. Classify Hot Money vs Core Capital.
        4. Optimize feasible action alternatives across financing APR, settlement delays, taxes, and caps.
        5. Evaluate action-specific hard safety gates on the chosen action proposal.
        6. Emit structured audit/learning events for Team 3.
        7. Return canonical JSON-serializable DecisionRecord.
        """
        decision_id = f"dec-{uuid4().hex[:12]}"
        now = state.timestamp
        risk_signals = list(risk_signals or [])

        # 1. Safety Score
        safety_score = calculate_treasury_safety_score(
            state, forecast_bundle, risk_signals=risk_signals
        )

        # 2. Dynamic Liquidity Reserve
        reserve_assessment = assess_liquidity_reserve(
            state, forecast_bundle, risk_signals=risk_signals
        )

        # 3. Hot / Core Capital Partitioning
        capital_classification = classify_hot_core_capital(
            state, forecast_bundle, reserve_assessment
        )

        # 4. Operations Research Optimization
        opt_result = optimize_treasury_actions(
            state=state,
            forecast_bundle=forecast_bundle,
            risk_signals=risk_signals,
            tax_profile=tax_profile,
            safety_score=safety_score,
            reserve_assessment=reserve_assessment,
            repo_overnight_rate_apy=repo_overnight_rate_apy,
            repo_sweep_lockup_hours=repo_sweep_lockup_hours,
        )

        selected = opt_result.selected_action
        alternatives = [a.to_dict() for a in opt_result.candidate_actions if a != selected]

        # 5. Evaluate Action-Specific Hard Safety Gates on the SELECTED Proposal
        # Ensures DecisionRecord reports the actual gate results of the proposal, not a dummy base evaluation
        if selected.action_type == ActionType.DRAW_BRIDGE_CREDIT:
            selected_gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                proposed_bridge_draw=selected.principal_borrowed,
                payment_deadline_hours=selected.payment_deadline_hours,
                availability_delay_hours=selected.availability_delay_hours,
                repayment_feasibility=selected.repayment_feasibility,
                is_partial_funding=selected.is_partial_funding,
                residual_unfunded_amount=selected.residual_unfunded_amount,
                action_type=selected.action_type.value,
            )
        elif selected.action_type in (ActionType.PLACE_REPO, ActionType.BUY_BOND, ActionType.BUY_EQUITY):
            selected_gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                proposed_investment_amount=selected.amount,
                proposed_lockup_hours=selected.timing_hours,
                target_tier=2 if selected.action_type == ActionType.BUY_BOND else (3 if selected.action_type == ActionType.BUY_EQUITY else 1),
                action_type=selected.action_type.value,
            )
        elif selected.action_type in (ActionType.BREAK_REPO_EARLY, ActionType.SELL_BOND, ActionType.SELL_EQUITY):
            selected_gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                payment_deadline_hours=selected.payment_deadline_hours,
                availability_delay_hours=selected.availability_delay_hours,
                is_partial_funding=selected.is_partial_funding,
                residual_unfunded_amount=selected.residual_unfunded_amount,
                action_type=selected.action_type.value,
            )
        else:
            # KEEP_CASH or UNRESOLVED_FUNDING_SHORTFALL
            selected_gate_eval = evaluate_hard_safety_gates(
                state=state,
                reserve_assessment=reserve_assessment,
                is_partial_funding=selected.is_partial_funding,
                residual_unfunded_amount=selected.residual_unfunded_amount,
                action_type=selected.action_type.value,
            )

        # Combine reason codes
        reason_codes: list[str] = [selected.reason_code]
        if selected_gate_eval.reason_codes:
            for r in selected_gate_eval.reason_codes:
                if r not in reason_codes:
                    reason_codes.append(r)
        if opt_result.remediation_notes:
            reason_codes.extend(opt_result.remediation_notes)

        # Build DecisionRecord
        record = StandardDecisionRecord(
            decision_id=decision_id,
            timestamp=now,
            selected_action=selected.to_dict(),
            alternatives=alternatives,
            score=safety_score.to_dict(),
            reserve=reserve_assessment.to_dict(),
            eligible_surplus=reserve_assessment.eligible_investable_surplus,
            cost_breakdown=selected.cost_breakdown,
            hard_gate_results=selected_gate_eval.to_dict(),
            reason_codes=reason_codes,
            metadata={
                "capital_classification": capital_classification.to_dict(),
                "tier_caps_applied": opt_result.tier_caps_applied.to_dict(),
                "tier_breaches_detected": opt_result.tier_breaches_detected,
                "objective_value": round(opt_result.objective_value, 4),
                "optimization_mode": opt_result.optimization_mode,
                "is_unresolved_shortfall": opt_result.is_unresolved_shortfall,
                "unresolved_shortfall": round(opt_result.unresolved_shortfall, 2),
            },
        )

        # 6. Publish Learning Events for Team 3
        # Check if any candidate action was rejected
        for cand in opt_result.candidate_actions:
            if not cand.is_feasible or not cand.hard_gate_passed:
                rej_event = StandardLearningEvent(
                    versioned_event_type="ACTION_REJECTED_V1",
                    source_module="reserve_juggler.decision",
                    correlation_id=decision_id,
                    structured_data={
                        "candidate_action": cand.to_dict(),
                        "reason": cand.reason_code,
                        "residual_gap": cand.residual_unfunded_amount,
                    },
                    model_policy_versions={
                        "decision_policy": self.policy_version,
                        "risk_model": self.model_version,
                    },
                    timestamp=now,
                )
                self._publish_event(rej_event)

        # If unresolved shortfall, publish escalation event
        if opt_result.is_unresolved_shortfall:
            shortfall_event = StandardLearningEvent(
                versioned_event_type="UNRESOLVED_FUNDING_SHORTFALL_V1",
                source_module="reserve_juggler.decision",
                correlation_id=decision_id,
                structured_data={
                    "shortfall_amount": opt_result.unresolved_shortfall,
                    "immediate_obligations": reserve_assessment.immediate_obligations,
                    "available_cash": state.available_cash,
                    "credit_available": state.credit_availability,
                },
                model_policy_versions={
                    "decision_policy": self.policy_version,
                    "risk_model": self.model_version,
                },
                timestamp=now,
            )
            self._publish_event(shortfall_event)

        # Proposal Event
        prop_event = StandardLearningEvent(
            versioned_event_type="DECISION_PROPOSED_V1",
            source_module="reserve_juggler.decision",
            correlation_id=decision_id,
            structured_data=record.to_dict(),
            model_policy_versions={
                "decision_policy": self.policy_version,
                "risk_model": self.model_version,
            },
            timestamp=now,
        )
        self._publish_event(prop_event)

        return record

    def record_execution_outcome(
        self, decision_id: str, outcome: ExecutionOutcome
    ) -> LearningEvent:
        """Record post-execution feedback and broadcast outcome event for Team 3."""
        event = StandardLearningEvent(
            versioned_event_type="EXECUTION_OUTCOME_RECORDED_V1",
            source_module="reserve_juggler.decision",
            correlation_id=decision_id,
            structured_data={
                "realized_cash_movements": outcome.realized_cash_movements,
                "costs": outcome.costs,
                "penalties": outcome.penalties,
                "losses": outcome.losses,
                "liquidity_shortfalls": outcome.liquidity_shortfalls,
                "metadata": dict(outcome.metadata or {}),
            },
            model_policy_versions={
                "decision_policy": self.policy_version,
                "risk_model": self.model_version,
            },
            timestamp=datetime.now(timezone.utc),
        )
        self._publish_event(event)
        return event
