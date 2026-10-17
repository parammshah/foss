"""
Application Facade: Central integration endpoint for Reserve Juggler.
Unifies Macro-Panic Sentinel, System-Wide Self-Learning, Explainability,
Cross-team Adapters, and the Interactive Dashboard.

Service Interface:
    ApplicationFacade.get_dashboard_snapshot(as_of_timestamp: Optional[str] = None) -> DashboardSnapshot
Version: 1.0.0
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.contracts import (
    ApprovalStatus,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    LearningEvent,
    RiskSignal,
    ServiceStatus,
    TreasuryState,
    UserRole,
)
from reserve_juggler.events.models import SentinelEvaluationResult
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.explainability.models import DecisionExplanation
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.integration.adapters import Team1DataForecastAdapter, Team2DecisionRiskAdapter
from reserve_juggler.integration.stress_engine import StressLabSimulationEngine
from reserve_juggler.learning.models import CandidateUpdate, SystemPerformanceReport
from reserve_juggler.learning.service import LearningService

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """
    Consolidated, versioned state snapshot for the Reserve Juggler Dashboard.
    Schema Version: 1.0.0
    """
    snapshot_version: str
    as_of_timestamp: str
    treasury_state: TreasuryState
    forecast_bundle: ForecastBundle
    sentinel_result: SentinelEvaluationResult
    latest_decision: DecisionRecord
    decision_explanation: DecisionExplanation
    system_performance: SystemPerformanceReport
    active_scenario: str
    data_source_mode: str  # "SYNTHETIC_DEMO", "LIVE", "PARTIAL", "UNINTEGRATED"
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    incident_log: List[Dict[str, Any]] = field(default_factory=list)
    active_candidates: List[CandidateUpdate] = field(default_factory=list)


class ApplicationFacade:
    """
    Facade managing cross-subsystem orchestration and dashboard data delivery.
    """

    FACADE_VERSION = "1.0.0"

    def __init__(self, persistence_file: Optional[str] = None, use_synthetic: bool = True):
        self.use_synthetic = use_synthetic
        mode_str = "SYNTHETIC_DEMO" if use_synthetic else "PRODUCTION"

        self.sentinel = MacroPanicSentinel(use_synthetic_mode=use_synthetic)
        self.learning_service = LearningService(persistence_file=persistence_file)
        self.explanation_service = DecisionExplanationService()
        self.team1_adapter = Team1DataForecastAdapter(mode=mode_str)
        self.team2_adapter = Team2DecisionRiskAdapter(mode=mode_str)
        self.stress_engine = StressLabSimulationEngine()

        self.current_scenario: str = "NORMAL"
        self._audit_trail: List[Dict[str, Any]] = []
        self._incident_log: List[Dict[str, Any]] = []
        self._latest_decision_record: Optional[DecisionRecord] = None

        self._bootstrap_demo_telemetry()

    def _bootstrap_demo_telemetry(self) -> None:
        """Seeds initial verified telemetry into SQLite event store."""
        now = datetime.now(timezone.utc).isoformat()
        
        # Historical Decision 1
        d1_id = "DEC-HIST-001"
        self.learning_service.record(
            LearningEvent(
                event_id="EV-D1",
                versioned_event_type="DECISION_GENERATED",
                module="DECISION_ENGINE",
                timestamp=now,
                correlation_id=d1_id,
                payload={
                    "decision_id": d1_id,
                    "selected_action": "DRAW_REPO_30D",
                    "target_asset": "REPO-30D-GCF",
                    "amount": 10_000_000.0,
                    "estimated_costs": {"total": 5_200.0},
                    "gate_outcomes": [{"gate_name": "MandatoryReserveFloorGate", "status": "PASS"}],
                },
                associated_versions={"decision_engine": "v1.2.0"},
            )
        )
        self.learning_service.record(
            LearningEvent(
                event_id="EV-O1",
                versioned_event_type="OUTCOME_OBSERVED",
                module="EXECUTION",
                timestamp=now,
                correlation_id=d1_id,
                payload={
                    "realized_amount": 10_000_000.0,
                    "realized_costs": {"total": 5_240.0},
                    "liquidity_shortfall": 0.0,
                    "settlement_status": "SETTLED",
                },
            )
        )
        self.learning_service.record(
            LearningEvent(
                event_id="EV-H1",
                versioned_event_type="HUMAN_APPROVAL_RECORDED",
                module="HUMAN_OVERSIGHT",
                timestamp=now,
                correlation_id=d1_id,
                payload={"status": "APPROVED", "actor_name": "Chief Treasurer", "stated_reason": "Standard operational liquidity draw."},
            )
        )

        # Historical Decision 2
        d2_id = "DEC-HIST-002"
        self.learning_service.record(
            LearningEvent(
                event_id="EV-D2",
                versioned_event_type="DECISION_GENERATED",
                module="DECISION_ENGINE",
                timestamp=now,
                correlation_id=d2_id,
                payload={
                    "decision_id": d2_id,
                    "selected_action": "LIQUIDATE_SOVEREIGN_BOND",
                    "target_asset": "SOV-UST-2Y",
                    "amount": 12_000_000.0,
                    "estimated_costs": {"total": 14_000.0},
                    "gate_outcomes": [{"gate_name": "MandatoryReserveFloorGate", "status": "PASS"}],
                },
                associated_versions={"decision_engine": "v1.2.0"},
            )
        )
        self.learning_service.record(
            LearningEvent(
                event_id="EV-O2",
                versioned_event_type="OUTCOME_OBSERVED",
                module="EXECUTION",
                timestamp=now,
                correlation_id=d2_id,
                payload={
                    "realized_amount": 12_000_000.0,
                    "realized_costs": {"total": 14_110.0},
                    "liquidity_shortfall": 0.0,
                    "settlement_status": "SETTLED",
                },
            )
        )
        self.learning_service.record(
            LearningEvent(
                event_id="EV-H2",
                versioned_event_type="HUMAN_APPROVAL_RECORDED",
                module="HUMAN_OVERSIGHT",
                timestamp=now,
                correlation_id=d2_id,
                payload={"status": "APPROVED", "actor_name": "Senior Cash Manager", "stated_reason": "Portfolio duration rebalancing."},
            )
        )

        # Pre-seed a candidate update in shadow mode
        self.learning_service.propose_update(
            component="FORECASTING",
            parameter_key="ensemble_combining_weights",
            proposed_value={"lstm_weight": 0.45, "arimax_weight": 0.55},
            current_value={"lstm_weight": 0.50, "arimax_weight": 0.50},
            description="Re-weight ARIMAX for tighter confidence intervals during volatile trading hours.",
            author="Quant Research Team",
            holdout_evaluation_score=0.92,
            holdout_sample_size=40,
            stress_shortfall_count=0,
        )

        self._audit_trail.append({
            "timestamp": now,
            "action": "SYSTEM_INITIALIZED",
            "actor": "System Bootstrap",
            "details": f"Reserve Juggler loaded in mode '{'SYNTHETIC_DEMO' if self.use_synthetic else 'PRODUCTION'}'.",
        })

    def switch_scenario(self, scenario_name: str) -> None:
        """Switch active operational scenario."""
        self.current_scenario = scenario_name
        self.sentinel.set_synthetic_scenario(scenario_name)
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "SCENARIO_SWITCHED",
            "actor": "User Dashboard",
            "details": f"Operational scenario transitioned to {scenario_name}",
        })

    def get_dashboard_snapshot(self, as_of_timestamp: Optional[str] = None) -> DashboardSnapshot:
        """Gathers real-time cross-team snapshot for the dashboard."""
        now_iso = as_of_timestamp or datetime.now(timezone.utc).isoformat()

        # 1. Team 1 State & Forecasts
        treasury_state = self.team1_adapter.get_treasury_state(now_iso, scenario=self.current_scenario)
        forecast_bundle, fc_event = self.team1_adapter.get_forecast_bundle(now_iso, scenario=self.current_scenario)
        self.learning_service.record(fc_event)

        # 2. Sentinel Evaluation
        sentinel_result = self.sentinel.evaluate(now_iso, treasury_state=treasury_state)
        for sig in sentinel_result.active_signals:
            self.learning_service.record(
                LearningEvent(
                    event_id=f"EV-SIG-{sig.signal_id}-{uuid.uuid4().hex[:6]}",
                    versioned_event_type="MACRO_SIGNAL_GENERATED",
                    module="SENTINEL",
                    timestamp=now_iso,
                    correlation_id=sig.signal_id,
                    payload={
                        "category": sig.category,
                        "severity": sig.severity,
                        "corroboration": sig.corroboration_status,
                        "relevance": sig.relevance_to_treasury,
                    },
                )
            )

        # 3. Team 2 Decision Generation
        if self._latest_decision_record is None or self._latest_decision_record.approval_status not in (
            ApprovalStatus.PENDING_APPROVAL.value, "PENDING"
        ):
            decision_record, dec_event = self.team2_adapter.compute_next_best_action(
                treasury_state=treasury_state,
                forecast_bundle=forecast_bundle,
                risk_signals=sentinel_result.active_signals,
                as_of_iso=now_iso,
                scenario=self.current_scenario,
            )
            self._latest_decision_record = decision_record
            self.learning_service.record(dec_event)
        else:
            decision_record = self._latest_decision_record

        # 4. Explainability Service
        explanation = self.explanation_service.explain(
            decision_record=decision_record,
            forecast_bundle=forecast_bundle,
            treasury_state=treasury_state,
        )

        # 5. System Learning Performance
        perf_report = self.learning_service.evaluate_system_performance()

        # Overall data source mode
        if self.use_synthetic:
            ds_mode = "SYNTHETIC_DEMO"
        elif self.team1_adapter.is_connected and self.team2_adapter.is_connected:
            ds_mode = "LIVE"
        else:
            ds_mode = "UNINTEGRATED"

        return DashboardSnapshot(
            snapshot_version=self.FACADE_VERSION,
            as_of_timestamp=now_iso,
            treasury_state=treasury_state,
            forecast_bundle=forecast_bundle,
            sentinel_result=sentinel_result,
            latest_decision=decision_record,
            decision_explanation=explanation,
            system_performance=perf_report,
            active_scenario=self.current_scenario,
            data_source_mode=ds_mode,
            audit_log=list(self._audit_trail),
            incident_log=list(self._incident_log),
            active_candidates=self.learning_service.get_all_candidates(),
        )

    def submit_human_action(
        self,
        action_type: str,
        actor_name: str,
        actor_role: str,
        stated_reason: str,
        modified_amount: Optional[float] = None,
        modified_action: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Processes human approval, modification, or cancellation with live revalidation
        against current TreasuryState.
        """
        if not self._latest_decision_record:
            return (False, "No pending decision available to act upon.")

        current_state = self.team1_adapter.get_treasury_state(scenario=self.current_scenario)

        success, msg, updated_dec, outcome, learning_ev = self.explanation_service.execute_human_workflow(
            decision_record=self._latest_decision_record,
            action_type=action_type,
            actor_name=actor_name,
            actor_role=actor_role,
            stated_reason=stated_reason,
            modified_amount=modified_amount,
            modified_action=modified_action,
            treasury_state=current_state,
        )

        if success:
            self._latest_decision_record = updated_dec
            if learning_ev:
                self.learning_service.record(learning_ev)
            
            if outcome and outcome.settlement_status not in ("CANCELLED", "REJECTED"):
                self.learning_service.record(
                    LearningEvent(
                        event_id=f"EV-OUT-{outcome.outcome_id}",
                        versioned_event_type="OUTCOME_OBSERVED",
                        module="EXECUTION",
                        timestamp=outcome.execution_timestamp,
                        correlation_id=outcome.correlation_id,
                        payload={
                            "realized_amount": outcome.realized_amount,
                            "realized_costs": outcome.realized_costs,
                            "liquidity_shortfall": outcome.liquidity_shortfall,
                            "settlement_status": outcome.settlement_status,
                        },
                    )
                )

            self._audit_trail.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": f"DECISION_{action_type}",
                "actor": f"{actor_name} ({actor_role})",
                "details": f"Decision {updated_dec.decision_id}: {msg}. Reason: '{stated_reason}'",
            })

        return (success, msg)

    def run_stress_lab_scenario(
        self,
        deposit_shock_pct: float = 0.0,
        rate_shock_bps: float = 0.0,
        repo_delay_days: int = 0,
        credit_unavailable: bool = False,
        equity_drawdown_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs on-demand stress lab simulation comparing 3 strategies under identical shocks.
        """
        state = self.team1_adapter.get_treasury_state(scenario=self.current_scenario)
        return self.stress_engine.run_simulation(
            base_treasury_state=state,
            deposit_shock_pct=deposit_shock_pct,
            rate_shock_bps=rate_shock_bps,
            repo_delay_days=repo_delay_days,
            credit_unavailable=credit_unavailable,
            equity_drawdown_pct=equity_drawdown_pct,
        )
