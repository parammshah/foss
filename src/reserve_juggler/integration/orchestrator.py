"""
Canonical End-to-End Treasury Decision Orchestrator for Reserve Juggler.

Coordinates the complete 10-step financial decision lifecycle across:
- Team 1: Data Ingestion, Ledger Accounting, Multi-Horizon Forecasting
- Team 2: Safety Scoring, Liquidity Shield, Tax Engine, Operations Research Optimization
- Team 3: Macro-Panic Sentinel, Explainability, System-Wide Self-Learning & Governance

Enforces mandatory non-negotiable financial safety constraints independently
before any proposed execution can be approved or simulated.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd

from reserve_juggler.contracts import (
    ActionType,
    ApprovalStatus,
    AssetPosition,
    AssetTier,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    GateStatus,
    HardGateOutcome,
    LearningEvent,
    Obligation,
    QualityStatus,
    RiskSignal,
    ServiceStatus,
    TreasuryState,
    UserRole,
    ensure_utc,
    format_iso_utc,
)
from reserve_juggler.data.validator import DatasetValidator, ValidationReport
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.explainability.models import DecisionExplanation
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.ledger.ledger import TreasuryLedger
from reserve_juggler.learning.service import LearningService
from reserve_juggler.tax.tax_profile import TaxProfile

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of independent safety constraint validation."""
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    gate_outcomes: List[HardGateOutcome] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_human_approval: bool = False
    shortfall_amount_usd: float = 0.0


class IndependentConstraintValidator:
    """
    Independent financial gatekeeper between optimizer output and execution.
    Evaluates customer obligation protection, holding period economics,
    stress mode restrictions, and credit facility limits.
    """

    def __init__(self, reserve_buffer_ratio: float = 1.05) -> None:
        self.reserve_buffer_ratio = reserve_buffer_ratio

    def validate_proposal(
        self,
        state: TreasuryState,
        proposal: Any,
        forecast_bundle: Optional[ForecastBundle] = None,
        risk_signals: Sequence[RiskSignal] = (),
        stress_regime: str = "NORMAL",
    ) -> ValidationResult:
        reasons: List[str] = []
        gates: List[HardGateOutcome] = []
        warnings: List[str] = []
        requires_human = False
        shortfall = 0.0

        # Gate 1: Customer Obligations & Immediate Payout Protection
        committed_obligations = (
            sum(ob.amount_usd for ob in state.obligations if ob.is_mandatory and not ob.is_settled)
            if state.obligations
            else max(0.0, getattr(state, "upcoming_obligations_usd", 0.0))
        )
        available_cash = max(0.0, state.operational_cash_usd)

        # Discretionary investments cannot consume cash committed to obligations
        sel_act = getattr(proposal, "selected_action", "HOLD_CASH")
        if isinstance(sel_act, dict):
            proposed_action = str(sel_act.get("action_type", "HOLD_CASH"))
        else:
            proposed_action = str(sel_act)
        proposed_amount = float(getattr(proposal, "amount", 0.0) or 0.0)

        is_investment = any(
            k in proposed_action.upper()
            for k in ["BUY", "INVEST", "SWEEP", "PLACE_REPO"]
        )

        if available_cash < committed_obligations:
            shortfall_amt = committed_obligations - available_cash
            shortfall = max(shortfall, shortfall_amt)
            reasons.append(
                f"Immediate liquidity shortfall: available cash (${available_cash:,.2f}) cannot cover committed obligations (${committed_obligations:,.2f}). Deficit: ${shortfall_amt:,.2f}."
            )
            gates.append(
                HardGateOutcome(
                    gate_name="CustomerObligationsProtectionGate",
                    status=GateStatus.FAIL,
                    threshold_value=committed_obligations,
                    observed_value=available_cash,
                    reason=f"Immediate liquidity shortfall of ${shortfall_amt:,.2f}.",
                )
            )
        elif is_investment:
            cash_after_investment = available_cash - proposed_amount
            if cash_after_investment < committed_obligations:
                shortfall_amt = committed_obligations - cash_after_investment
                reasons.append(
                    f"Discretionary investment ${proposed_amount:,.2f} breaches committed customer obligations: "
                    f"Cash remaining (${cash_after_investment:,.2f}) < Mandatory obligations (${committed_obligations:,.2f})."
                )
                shortfall = max(shortfall, shortfall_amt)
                gates.append(
                    HardGateOutcome(
                        gate_name="CustomerObligationsProtectionGate",
                        status=GateStatus.FAIL,
                        threshold_value=committed_obligations,
                        observed_value=cash_after_investment,
                        reason="Discretionary investment consumes committed customer payouts.",
                    )
                )
            else:
                gates.append(
                    HardGateOutcome(
                        gate_name="CustomerObligationsProtectionGate",
                        status=GateStatus.PASS,
                        threshold_value=committed_obligations,
                        observed_value=cash_after_investment,
                        reason="Customer obligations safely covered by post-investment cash.",
                    )
                )
        # Gate 1B: Total Required Reserve Gate (Protected Capital + Future Obligation Reserve)
        reserve_info = getattr(proposal, "reserve", {})
        if isinstance(reserve_info, dict):
            prot_cap = float(reserve_info.get("protected_capital", 0.0) or 0.0)
            fut_res = float(reserve_info.get("future_obligation_reserve", 0.0) or 0.0)
            total_req_reserve = float(reserve_info.get("total_required_reserve") or (prot_cap + fut_res))
        elif isinstance(reserve_info, (int, float)):
            prot_cap = float(reserve_info)
            fut_res = 0.0
            total_req_reserve = prot_cap
        else:
            prot_cap = committed_obligations
            fut_res = 0.0
            total_req_reserve = committed_obligations

        if is_investment:
            cash_after_investment = available_cash - proposed_amount
            if total_req_reserve > 0 and cash_after_investment < total_req_reserve - 1e-4:
                res_deficit = total_req_reserve - cash_after_investment
                shortfall = max(shortfall, res_deficit)
                reasons.append(
                    f"Discretionary investment ${proposed_amount:,.2f} breaches total required reserve floor "
                    f"(${total_req_reserve:,.2f} = protected ${prot_cap:,.2f} + future reserve ${fut_res:,.2f}): "
                    f"Remaining operating cash (${cash_after_investment:,.2f}) leaves shortfall of ${res_deficit:,.2f}."
                )
                gates.append(
                    HardGateOutcome(
                        gate_name="TotalRequiredReserveGate",
                        status=GateStatus.FAIL,
                        threshold_value=total_req_reserve,
                        observed_value=cash_after_investment,
                        reason=f"Discretionary investment breaches total required reserve floor by ${res_deficit:,.2f}.",
                    )
                )
            else:
                gates.append(
                    HardGateOutcome(
                        gate_name="TotalRequiredReserveGate",
                        status=GateStatus.PASS,
                        threshold_value=total_req_reserve,
                        observed_value=cash_after_investment,
                        reason="Total required reserve safely preserved in cleared operating cash post-investment.",
                    )
                )
        else:
            gates.append(
                HardGateOutcome(
                    gate_name="TotalRequiredReserveGate",
                    status=GateStatus.PASS,
                    threshold_value=total_req_reserve,
                    observed_value=available_cash,
                    reason="Operating cash preserves total required reserve.",
                )
            )

        # Gate 2: Liquidity Shortfall Immediate Escalation
        total_immediate_liquidity = state.operational_cash_usd
        if committed_obligations > total_immediate_liquidity:
            deficit = committed_obligations - total_immediate_liquidity
            if proposed_action in ("HOLD_CASH", "HOLD") and deficit > 1.0:
                shortfall = max(shortfall, deficit)
                reasons.append(
                    f"Immediate liquidity shortfall: Available cash (${total_immediate_liquidity:,.2f}) "
                    f"cannot cover obligations (${committed_obligations:,.2f}). Deficit = ${deficit:,.2f}."
                )
                gates.append(
                    HardGateOutcome(
                        gate_name="ImmediateLiquidityDeficitGate",
                        status=GateStatus.FAIL,
                        threshold_value=committed_obligations,
                        observed_value=total_immediate_liquidity,
                        reason="Available liquidity insufficient to meet immediate obligations without funding action.",
                    )
                )

        # Gate 3: Holding Time Gatekeeper & Break-even economics
        if "BOND" in str(proposed_action).upper() and proposed_amount > 0:
            daily_yield = 0.048 / 365.0
            total_friction = 0.0005 + 0.02
            breakeven_days = total_friction / daily_yield if daily_yield > 0 else 365.0
            if breakeven_days > 180.0:
                warnings.append(f"Long break-even holding period for bond purchase: {breakeven_days:.1f} days.")
            gates.append(
                HardGateOutcome(
                    gate_name="HoldingTimeGatekeeper",
                    status=GateStatus.PASS,
                    threshold_value=breakeven_days,
                    observed_value=180.0,
                    reason=f"Break-even holding horizon calculated at {breakeven_days:.1f} days.",
                )
            )

        # Gate 4: Stress Mode Restrictions
        if stress_regime in ("ELEVATED_WITHDRAWAL_RISK", "SEVERE_LIQUIDITY_STRESS"):
            requires_human = True
            if is_investment and proposed_amount > 0:
                reasons.append(
                    f"Discretionary asset purchase prohibited during active stress regime ({stress_regime})."
                )
                gates.append(
                    HardGateOutcome(
                        gate_name="StressRegimeCapitalLockGate",
                        status=GateStatus.FAIL,
                        threshold_value=0.0,
                        observed_value=proposed_amount,
                        reason=f"Discretionary investments frozen under {stress_regime}.",
                    )
                )

        # Gate 5: Credit Line Committed Capacity
        if any(k in str(proposed_action).upper() for k in ["DRAW_CREDIT", "DRAW_BRIDGE_CREDIT", "BORROW_CREDIT"]):
            avail_credit = state.available_credit_usd
            if proposed_amount > avail_credit:
                reasons.append(
                    f"Requested credit draw (${proposed_amount:,.2f}) exceeds committed facility limit (${avail_credit:,.2f})."
                )
                gates.append(
                    HardGateOutcome(
                        gate_name="CommittedCreditFacilityGate",
                        status=GateStatus.FAIL,
                        threshold_value=avail_credit,
                        observed_value=proposed_amount,
                        reason="Draw exceeds committed line limit.",
                    )
                )
            else:
                requires_human = True

        # Gate 6: Tax Safety and Regulatory Authority Gate
        meta = getattr(proposal, "metadata", {}) or {}
        tax_status = meta.get("tax_profile_status")
        is_provisional_tax = bool(
            meta.get("ranking_is_provisional", False)
            or getattr(proposal, "ranking_is_provisional", False)
            or any("PROVISIONAL_TAX" in str(r) for r in getattr(proposal, "reason_codes", []))
        )
        if is_provisional_tax or tax_status in ("UNCONFIGURED_GROSS_ONLY", "SYNTHETIC_DEMO", "INVALID"):
            warnings.append(
                f"Provisional Tax Assumptions: Tax profile status is '{tax_status or 'PROVISIONAL'}'. "
                "Ranking is calculated on gross yield; human officer approval is required before execution."
            )
            requires_human = True
            gates.append(
                HardGateOutcome(
                    gate_name="TaxSafetyAuthorityGate",
                    status=GateStatus.PASS,
                    threshold_value=1.0,
                    observed_value=0.0,
                    reason=f"Provisional tax profile ({tax_status or 'PROVISIONAL'}): requires human approval prior to commitment.",
                )
            )
        else:
            gates.append(
                HardGateOutcome(
                    gate_name="TaxSafetyAuthorityGate",
                    status=GateStatus.PASS,
                    threshold_value=1.0,
                    observed_value=1.0,
                    reason="Verified statutory corporate tax profile.",
                )
            )

        passed = len(reasons) == 0
        return ValidationResult(
            passed=passed,
            rejection_reasons=reasons,
            gate_outcomes=gates,
            warnings=warnings,
            requires_human_approval=requires_human or not passed,
            shortfall_amount_usd=shortfall,
        )


@dataclass
class OrchestratedDecisionResult:
    """Comprehensive, auditable outcome of an integrated decision cycle."""
    cycle_id: str
    as_of_timestamp: str
    data_quality_report: Optional[Dict[str, Any]]
    stress_regime: str
    risk_signals: List[RiskSignal]
    forecast_bundle: ForecastBundle
    treasury_state: TreasuryState
    decision_record: DecisionRecord
    validation_result: ValidationResult
    explanation: Optional[DecisionExplanation]
    simulated_outcome: Optional[ExecutionOutcome]
    audit_event: LearningEvent
    approval_status: str
    workflow_state: str = "RECOMMENDATION_GENERATED"
    decision_timestamp: Optional[str] = None
    forecast_as_of_timestamp: Optional[str] = None
    execution_timestamp: Optional[str] = None
    model_versions: Dict[str, str] = field(default_factory=dict)


class DecisionOrchestrator:
    """
    Master coordination service executing the 10-step Reserve Juggler lifecycle.
    Wires Data Validation, Forecasting, Sentinel, Optimization, Independent Validation,
    Approval Policies, and Learning Feedback into one coherent workflow.
    """

    ORCHESTRATOR_VERSION = "2.0.0"

    def __init__(
        self,
        validator: Optional[DatasetValidator] = None,
        forecast_service: Optional[ForecastService] = None,
        sentinel: Optional[MacroPanicSentinel] = None,
        decision_service: Optional[TreasuryDecisionService] = None,
        explanation_service: Optional[DecisionExplanationService] = None,
        learning_service: Optional[LearningService] = None,
        ledger: Optional[TreasuryLedger] = None,
        independent_validator: Optional[IndependentConstraintValidator] = None,
    ) -> None:
        self.validator = validator or DatasetValidator()
        self.forecast_service = forecast_service or ForecastService()
        self.sentinel = sentinel or MacroPanicSentinel(use_synthetic_mode=False)
        self.decision_service = decision_service or TreasuryDecisionService()
        self.explanation_service = explanation_service or DecisionExplanationService()
        self.learning_service = learning_service or LearningService()
        self.ledger = ledger or TreasuryLedger()
        self.independent_validator = independent_validator or IndependentConstraintValidator()

    def run_cycle(
        self,
        current_state: TreasuryState,
        as_of_timestamp: datetime,
        historical_df: Optional[Any] = None,
        external_raw_events: Optional[List[Any]] = None,
        tax_profile: Optional[TaxProfile] = None,
        auto_simulate_execution: bool = True,
    ) -> OrchestratedDecisionResult:
        """
        Executes the 10-step canonical decision cycle.
        """
        cycle_id = f"CYC-{as_of_timestamp.strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
        now_utc = ensure_utc(as_of_timestamp)
        now_iso = format_iso_utc(now_utc)

        # Step 1: Validate input data and account state
        dq_report: Optional[Dict[str, Any]] = None
        data_quality_warnings: List[str] = []
        if historical_df is not None:
            # Enforce zero lookahead leakage on training data
            from reserve_juggler.data.schema import TIMESTAMP_COL
            if TIMESTAMP_COL in historical_df.columns:
                hist_ts = pd.to_datetime(historical_df[TIMESTAMP_COL], utc=True)
                max_training_ts = hist_ts.max()
                assert max_training_ts <= now_utc, (
                    f"Temporal leakage detected: max(training_timestamps) ({max_training_ts}) "
                    f"> as_of_timestamp ({now_utc})"
                )

            val_rep = self.validator.validate(historical_df)
            errs = [i for i in val_rep.issues if i.severity == "ERROR"]
            warns = [i for i in val_rep.issues if i.severity == "WARNING"]
            dq_report = {
                "is_valid": val_rep.is_valid,
                "error_count": len(errs),
                "warning_count": len(warns),
                "row_count": val_rep.total_rows,
            }
            data_quality_warnings = [w.message for w in warns]

        # Step 2: Generate multi-horizon forecasts and uncertainty estimates
        if getattr(self.forecast_service, "suites", None) and self.forecast_service._historical_df is not None:
            forecast_bundle = self.forecast_service.forecast(now_utc)
        else:
            from reserve_juggler.integration.fixtures import create_mock_forecast_bundle
            forecast_bundle = create_mock_forecast_bundle(now_iso, scenario="NORMAL")

        # Step 3: Collect available external risk signals & assess freshness
        risk_signals: List[RiskSignal] = []
        if external_raw_events:
            for raw_evt in external_raw_events:
                sig = self.sentinel.process_event(raw_evt)
                if sig and not sig.is_duplicate:
                    risk_signals.append(sig)

        # Step 4: Determine the current liquidity and stress regime
        sentinel_eval = self.sentinel.evaluate(as_of_timestamp=now_iso, treasury_state=current_state)
        max_severity = "LOW"
        for s in risk_signals:
            if s.severity == "CRITICAL":
                max_severity = "CRITICAL"
            elif s.severity == "HIGH" and max_severity != "CRITICAL":
                max_severity = "HIGH"

        if max_severity == "CRITICAL" or sentinel_eval.aggregate_macro_stress_score > 75.0:
            stress_regime = "SEVERE_LIQUIDITY_STRESS"
        elif max_severity == "HIGH" or sentinel_eval.aggregate_macro_stress_score > 40.0:
            stress_regime = "ELEVATED_WITHDRAWAL_RISK"
        elif forecast_bundle.is_fallback or len(data_quality_warnings) > 3:
            stress_regime = "STALE_DEGRADED"
        else:
            stress_regime = "NORMAL"

        # Step 5: Pass forecasts, balances, risk signals, and tax profile to optimizer
        decision_record = self.decision_service.decide(
            state=current_state,
            forecast_bundle=forecast_bundle,
            risk_signals=risk_signals,
            tax_profile=tax_profile,
        )

        # Step 6: Validate proposed allocation independently against hard constraints
        val_result = self.independent_validator.validate_proposal(
            state=current_state,
            proposal=decision_record,
            forecast_bundle=forecast_bundle,
            risk_signals=risk_signals,
            stress_regime=stress_regime,
        )

        # Step 7: Apply approval and execution policies
        dec_ts_str = (
            decision_record.timestamp.isoformat()
            if hasattr(decision_record.timestamp, "isoformat")
            else str(decision_record.timestamp)
        )
        setattr(decision_record, "forecast_as_of_timestamp", now_iso)

        if not val_result.passed:
            approval_status = ApprovalStatus.REJECTED.value
            workflow_state = "SAFETY_VALIDATION_FAILED"
            decision_record.approval_status = approval_status
            decision_record.reason_codes.extend(val_result.rejection_reasons)
        elif val_result.requires_human_approval:
            approval_status = ApprovalStatus.PENDING_APPROVAL.value
            workflow_state = "HUMAN_APPROVAL_PENDING"
            decision_record.approval_status = approval_status
        else:
            if auto_simulate_execution:
                approval_status = ApprovalStatus.APPROVED.value
                workflow_state = "EXECUTION_ELIGIBLE"
            else:
                approval_status = ApprovalStatus.PENDING_APPROVAL.value
                workflow_state = "EXECUTION_ELIGIBLE"
            decision_record.approval_status = approval_status

        # Step 8: Return a unified, explainable decision
        explanation: Optional[DecisionExplanation] = None
        try:
            explanation = self.explanation_service.explain(
                decision_record=decision_record,
                forecast_bundle=forecast_bundle,
                treasury_state=current_state,
            )
        except Exception as e:
            logger.warning("Explanation generation failed: %s", e)

        # Step 9: Simulate financial execution outcome if approved
        simulated_outcome: Optional[ExecutionOutcome] = None
        exec_ts: Optional[str] = None
        if auto_simulate_execution and approval_status == ApprovalStatus.APPROVED.value:
            try:
                action_name = str(getattr(decision_record, "action_type", decision_record.selected_action)).upper()
                action_spec = {
                    "type": action_name,
                    "amount_usd": float(getattr(decision_record, "amount", 0.0) or 0.0),
                    "asset_id": str(getattr(decision_record, "target_asset", "CASH")),
                    "decision_id": decision_record.decision_id,
                }
                new_state, sim_out = self.ledger.simulate_action(current_state, action_spec, forecast_bundle)
                simulated_outcome = sim_out
                exec_ts = now_iso
                if sim_out.execution_status == "SETTLED" or sim_out.settlement_status == "SETTLED":
                    approval_status = ApprovalStatus.EXECUTED.value
                    decision_record.approval_status = approval_status
                    decision_record.execution_timestamp = now_iso
                    workflow_state = "SIMULATED_EXECUTION_COMPLETED"
                else:
                    workflow_state = "SIMULATED_EXECUTION_REJECTED"
            except Exception as e:
                logger.warning("Simulation execution failed: %s", e)
                workflow_state = "SIMULATED_EXECUTION_REJECTED"

        # Step 10: Record decision, audit trail, and telemetry into system-wide self-learning
        audit_event = LearningEvent(
            event_id=f"EV-CYCLE-{cycle_id}",
            versioned_event_type="ORCHESTRATED_DECISION_RECORDED",
            module="ORCHESTRATOR",
            timestamp=now_iso,
            correlation_id=cycle_id,
            decision_id=decision_record.decision_id,
            payload={
                "cycle_id": cycle_id,
                "stress_regime": stress_regime,
                "selected_action": decision_record.selected_action,
                "target_asset": decision_record.target_asset,
                "amount": decision_record.amount,
                "validation_passed": val_result.passed,
                "rejection_reasons": val_result.rejection_reasons,
                "approval_status": approval_status,
                "workflow_state": workflow_state,
                "shortfall_usd": val_result.shortfall_amount_usd,
            },
            associated_versions={
                "orchestrator_version": self.ORCHESTRATOR_VERSION,
                "decision_engine_version": getattr(self.decision_service, "policy_version", "2.0.0"),
            },
        )
        try:
            self.learning_service.store.record(audit_event, ignore_duplicates=True)
        except Exception as e:
            logger.warning("Failed to persist learning audit event: %s", e)

        return OrchestratedDecisionResult(
            cycle_id=cycle_id,
            as_of_timestamp=now_iso,
            data_quality_report=dq_report,
            stress_regime=stress_regime,
            risk_signals=risk_signals,
            forecast_bundle=forecast_bundle,
            treasury_state=current_state,
            decision_record=decision_record,
            validation_result=val_result,
            explanation=explanation,
            simulated_outcome=simulated_outcome,
            audit_event=audit_event,
            approval_status=approval_status,
            workflow_state=workflow_state,
            decision_timestamp=dec_ts_str,
            forecast_as_of_timestamp=now_iso,
            execution_timestamp=exec_ts,
            model_versions={
                "orchestrator": self.ORCHESTRATOR_VERSION,
                "forecasting": getattr(self.forecast_service, "model_version", "1.0.0"),
                "decision_policy": getattr(self.decision_service, "policy_version", "2.0.0"),
            },
        )

    def dispatch_live_execution(
        self,
        decision_record: DecisionRecord,
        treasury_state: TreasuryState,
        actor_name: str,
        actor_role: str,
        stated_reason: str = "Authorized live dispatch",
    ) -> Tuple[bool, str, Optional[ExecutionOutcome]]:
        """
        Dispatches a human-approved decision to live production financial infrastructure.
        Fails closed if live execution is attempted with provisional/synthetic tax profiles,
        unauthorized actor roles, or unconfigured banking gateways.
        """
        success, msg, _, outcome, _ = self.explanation_service.execute_human_workflow(
            decision_record=decision_record,
            action_type="APPROVED",
            actor_name=actor_name,
            actor_role=actor_role,
            stated_reason=stated_reason,
            treasury_state=treasury_state,
            is_live_execution=True,
        )
        return (success, msg, outcome)
