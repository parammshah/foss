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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.events.models import SentinelEvaluationResult
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.explainability.models import DecisionExplanation
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.integration.adapters import Team1DataForecastAdapter, Team2DecisionRiskAdapter
from reserve_juggler.integration.contracts import (
    ApprovalStatus,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    LearningEvent,
    RiskSignal,
    TreasuryState,
    UserRole,
)
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
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    incident_log: List[Dict[str, Any]] = field(default_factory=list)
    active_candidates: List[CandidateUpdate] = field(default_factory=list)


class ApplicationFacade:
    """
    Facade managing cross-subsystem orchestration and dashboard data delivery.
    """

    FACADE_VERSION = "1.0.0"

    def __init__(self, persistence_file: Optional[str] = None, use_synthetic: bool = True):
        self.sentinel = MacroPanicSentinel(use_synthetic_mode=use_synthetic)
        self.learning_service = LearningService(persistence_file=persistence_file)
        self.explanation_service = DecisionExplanationService()
        self.team1_adapter = Team1DataForecastAdapter()
        self.team2_adapter = Team2DecisionRiskAdapter()

        self.current_scenario: str = "NORMAL"
        self._audit_trail: List[Dict[str, Any]] = []
        self._incident_log: List[Dict[str, Any]] = []
        self._latest_decision_record: Optional[DecisionRecord] = None

        # Seed with initial simulated learning events to provide rich baseline telemetry
        self._bootstrap_demo_telemetry()

    def _bootstrap_demo_telemetry(self) -> None:
        """Seeds demo history with realistic matched events and audit entries."""
        now = datetime.now(timezone.utc).isoformat()
        
        # Decision 1
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

        # Decision 2
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

        # Initial audit log
        self._audit_trail.append({
            "timestamp": now,
            "action": "SYSTEM_INITIALIZED",
            "actor": "System Bootstrap",
            "details": "Reserve Juggler Team 3 Services loaded in hackathon demonstration mode.",
        })

    def switch_scenario(self, scenario_name: str) -> None:
        """Switch active operational scenario (e.g. NORMAL, BANKING_STRESS, RATE_HIKE_SHOCK)."""
        self.current_scenario = scenario_name
        self.sentinel.set_synthetic_scenario(scenario_name)
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "SCENARIO_SWITCHED",
            "actor": "User Dashboard",
            "details": f"Operational scenario transitioned to {scenario_name}",
        })

    def get_dashboard_snapshot(self, as_of_timestamp: Optional[str] = None) -> DashboardSnapshot:
        """
        Gathers real-time cross-team snapshot for the dashboard.
        """
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
                    event_id=f"EV-SIG-{sig.signal_id}",
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
        if self._latest_decision_record is None or self._latest_decision_record.approval_status != ApprovalStatus.PENDING.value:
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
        """Processes human approval, modification, or rejection."""
        if not self._latest_decision_record:
            return (False, "No pending decision available to act upon.")

        success, msg, updated_dec, outcome, learning_ev = self.explanation_service.execute_human_workflow(
            decision_record=self._latest_decision_record,
            action_type=action_type,
            actor_name=actor_name,
            actor_role=actor_role,
            stated_reason=stated_reason,
            modified_amount=modified_amount,
            modified_action=modified_action,
        )

        if success:
            self._latest_decision_record = updated_dec
            self.learning_service.record(learning_ev)
            
            # Record simulated execution outcome in event store
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
                "timestamp": outcome.execution_timestamp,
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
    ) -> Dict[str, Any]:
        """
        Runs on-demand stress lab simulation comparing strategies under combined shocks.
        """
        state = self.team1_adapter.get_treasury_state()
        cash = state.available_cash
        assets_value = sum(p.market_value for p in state.asset_positions)
        reserve = state.protected_reserve_floor

        # Apply deposit shock
        simulated_outflow = 15_000_000.0 * (1.0 + deposit_shock_pct / 100.0)
        
        # Apply bond yield rate shock to sovereign bonds
        bond_loss = 0.0
        if rate_shock_bps > 0:
            for p in state.asset_positions:
                if p.asset_class == "SOVEREIGN_BONDS":
                    # loss = value * duration * delta_y
                    bond_loss += p.market_value * p.duration_years * (rate_shock_bps / 10000.0)

        remaining_cash = cash - simulated_outflow
        shortfall = max(0.0, reserve - remaining_cash)

        return {
            "deposit_shock_pct": deposit_shock_pct,
            "rate_shock_bps": rate_shock_bps,
            "repo_delay_days": repo_delay_days,
            "credit_unavailable": credit_unavailable,
            "simulated_outflow": round(simulated_outflow, 2),
            "bond_mtm_loss": round(bond_loss, 2),
            "post_shock_cash": round(remaining_cash, 2),
            "reserve_floor": round(reserve, 2),
            "is_shortfall": shortfall > 0,
            "shortfall_amount": round(shortfall, 2),
            "recommended_hedge": "DRAW_REPO_OVN" if not credit_unavailable else "LIQUIDATE_UST_2Y",
        }
