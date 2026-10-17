"""
LearningService: System-Wide Self-Learning Feedback Service.
Observes outcomes across the entire treasury lifecycle, matches predictions to observations,
detects drift, executes genuine champion vs candidate holdout evaluations, updates active
runtime execution configuration upon promotion, and supports audited rollback.

Exposed Integration Services:
    LearningService.record(event: LearningEvent) -> None
    LearningService.evaluate_system_performance(start: Optional[str], end: Optional[str]) -> SystemPerformanceReport
    LearningService.propose_update(...) -> CandidateUpdate
    LearningService.evaluate_in_shadow_mode(...) -> CandidateUpdate
    LearningService.promote_candidate(...) -> Tuple[bool, str]
    LearningService.rollback_component(...) -> Tuple[bool, str]
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.contracts import LearningEvent
from reserve_juggler.learning.evaluation_engine import CandidateEvaluationEngine
from reserve_juggler.learning.governance import SafetyInvariantValidator, VersionRegistry
from reserve_juggler.learning.matcher import OutcomeMatcher
from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.models import (
    CandidateStatus,
    CandidateUpdate,
    SystemPerformanceReport,
)
from reserve_juggler.learning.runtime_registry import RuntimeComponentRegistry
from reserve_juggler.learning.store import EventStore

logger = logging.getLogger(__name__)


class LearningService:
    """
    Central self-learning coordination service for Reserve Juggler.
    """

    SERVICE_VERSION = "1.0.0"

    def __init__(self, persistence_file: Optional[str] = None):
        self.store = EventStore(persistence_file=persistence_file)
        self.matcher = OutcomeMatcher(self.store)
        self.evaluator = SystemPerformanceEvaluator(self.store)
        self.validator = SafetyInvariantValidator()
        self.registry = VersionRegistry(persistence_store=self.store)
        self.runtime_registry = RuntimeComponentRegistry(persistence_store=self.store)
        self.evaluation_engine = CandidateEvaluationEngine(
            validator=self.validator,
            runtime_registry=self.runtime_registry,
            store=self.store,
        )
        self._candidates: Dict[str, CandidateUpdate] = {}

        # Recover persisted candidates if durable store exists
        if hasattr(self.store, "sqlite_store") and self.store.sqlite_store:
            try:
                for c in self.store.sqlite_store.get_all_candidates():
                    if c:
                        self._candidates[c.candidate_id] = c
            except Exception as e:
                logger.debug("Could not recover persisted candidates: %s", e)

    def record(self, event: LearningEvent, ignore_duplicates: bool = False) -> None:
        """
        Record any versioned event in the treasury lifecycle.
        Called by data ingestion, forecasting, TSS, sentinel, decision engine, and human oversight.
        """
        self.store.record(event, ignore_duplicates=ignore_duplicates)

    def evaluate_system_performance(
        self, start_iso: Optional[str] = None, end_iso: Optional[str] = None
    ) -> SystemPerformanceReport:
        """
        Calculates end-to-end performance metrics, error rates, calibration,
        and drift across all subsystems.
        """
        matched_decisions = self.matcher.match_decisions()
        matched_forecasts = self.matcher.match_forecasts()
        return self.evaluator.evaluate(
            matched_decisions=matched_decisions,
            matched_forecasts=matched_forecasts,
            start_iso=start_iso,
            end_iso=end_iso,
            active_versions=self.registry.get_active_versions(),
        )

    def propose_update(
        self,
        component: str,
        parameter_key: str,
        proposed_value: Any,
        description: str,
        author: str,
        current_value: Optional[Any] = None,
        holdout_evaluation_score: float = 0.0,
        holdout_sample_size: int = 10,
        stress_shortfall_count: int = 0,
    ) -> CandidateUpdate:
        """
        Propose a candidate model, parameter, or threshold update.
        Initializes validation checks against safety invariants.
        """
        cid = f"CAND-{component[:4].upper()}-{uuid.uuid4().hex[:6].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Retrieve current live parameter if not provided
        cur_val = current_value if current_value is not None else self.runtime_registry.get_parameter(component, parameter_key, "default")

        candidate = CandidateUpdate(
            candidate_id=cid,
            component=component.upper(),
            parameter_key=parameter_key,
            current_value=cur_val,
            proposed_value=proposed_value,
            description=description,
            author=author,
            created_at=now_iso,
            status=CandidateStatus.PROPOSED,
            holdout_evaluation_score=holdout_evaluation_score,
            holdout_sample_size=holdout_sample_size,
            stress_shortfall_count=stress_shortfall_count,
        )

        # Run static safety invariant validation immediately
        passed, violations = self.validator.validate_candidate(candidate)
        candidate.safety_check_passed = passed
        candidate.safety_violations = violations

        if not passed:
            candidate.status = CandidateStatus.REJECTED
            candidate.rejection_reason = "; ".join(violations)
            logger.warning("Candidate %s rejected at proposal: %s", cid, candidate.rejection_reason)
        else:
            candidate.status = CandidateStatus.PROPOSED

        self._candidates[cid] = candidate
        
        # Save to durable store if available
        if hasattr(self.store, "sqlite_store"):
            try:
                self.store.sqlite_store.save_candidate(candidate)
            except Exception as e:
                logger.error("Failed to persist candidate: %s", str(e))

        # Record learning event
        self.record(
            LearningEvent(
                event_id=f"EV-CAND-{cid}",
                versioned_event_type="CANDIDATE_UPDATE_PROPOSED",
                module="SYSTEM_LEARNING",
                timestamp=now_iso,
                correlation_id=cid,
                payload={
                    "candidate_id": cid,
                    "component": component,
                    "parameter": parameter_key,
                    "passed_safety": passed,
                    "violations": violations,
                },
                associated_versions=self.registry.get_active_versions(),
            ),
            ignore_duplicates=True,
        )

        return candidate

    def evaluate_in_shadow_mode(
        self, candidate_id: str, sample_size: int = 30, evaluation_records: Optional[List[Dict[str, Any]]] = None
    ) -> CandidateUpdate:
        """
        Executes genuine holdout evaluation comparing candidate vs active champion
        on chronological holdout data and stress scenarios.
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise ValueError(f"Candidate {candidate_id} not found")

        if candidate.status == CandidateStatus.REJECTED:
            return candidate

        report = self.evaluation_engine.evaluate_candidate(
            candidate=candidate,
            sample_size=sample_size,
            evaluation_records=evaluation_records,
        )
        candidate.validation_report = report
        candidate.holdout_sample_size = report.validation_sample_size
        candidate.stress_shortfall_count = report.stress_shortfall_count
        candidate.safety_check_passed = report.safety_invariants_passed
        candidate.safety_violations = report.violations

        if not report.is_promotable:
            candidate.status = CandidateStatus.REJECTED
            candidate.rejection_reason = "; ".join(report.violations)
        else:
            candidate.status = CandidateStatus.SHADOW_MODE
            candidate.shadow_mode_runs = report.shadow_runs
            candidate.shadow_win_rate = report.shadow_win_rate

        # Update in store
        if hasattr(self.store, "sqlite_store"):
            try:
                self.store.sqlite_store.save_candidate(candidate)
            except Exception as e:
                logger.error("Failed to persist candidate: %s", str(e))

        return candidate

    def promote_candidate(
        self,
        candidate_id: str,
        approver: str,
        approver_role: str,
        justification: str,
        new_version_tag: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Promotes an authorized candidate update to active production version.
        Requires genuine validation completion, safety invariants passing, and human authorization.
        Directly mutates live operational runtime parameter configuration.
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return (False, f"Candidate {candidate_id} not found.")

        # Enforce role authorization: Only HEAD_OF_TREASURY, RISK_OFFICER
        if approver_role not in ("HEAD_OF_TREASURY", "RISK_OFFICER"):
            return (False, f"Permission Denied: Role '{approver_role}' cannot authorize model promotion.")

        # Verify candidate passed validation
        if candidate.status == CandidateStatus.REJECTED:
            return (False, f"Cannot promote rejected candidate: {candidate.rejection_reason}")

        # Check recovery state
        if self.runtime_registry.is_recovery_required(candidate.component):
            return (
                False,
                f"Promotion blocked: Component '{candidate.component}' is in RECOVERY_REQUIRED state "
                f"({self.runtime_registry.get_recovery_reason(candidate.component)}). Promotion prohibited.",
            )

        # Enforce that component has a registered active runtime consumer
        if not self.runtime_registry.has_active_provider(candidate.component):
            return (
                False,
                f"Promotion blocked: Component '{candidate.component}' does not have an active registered runtime consumer (UNINTEGRATED).",
            )

        # Check validation report
        if not candidate.validation_report:
            # Auto-evaluate before promotion
            self.evaluate_in_shadow_mode(candidate_id, sample_size=max(candidate.holdout_sample_size, 20))
            if candidate.status == CandidateStatus.REJECTED:
                return (False, f"Candidate validation failed: {candidate.rejection_reason}")

        if candidate.validation_report and not candidate.validation_report.is_promotable:
            return (False, f"Promotion blocked by validation report: {'; '.join(candidate.validation_report.violations)}")

        if candidate.status != CandidateStatus.SHADOW_MODE:
            return (
                False,
                f"Promotion blocked: Candidate '{candidate_id}' has not successfully executed in shadow mode "
                f"(status={candidate.status.value}). Untested candidates cannot be promoted.",
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        version_tag = new_version_tag or f"v{datetime.now().strftime('%Y%m%d%H%M')}"
        prev_version = self.registry.get_active_versions().get(candidate.component, "v1.0.0")
        prev_status = candidate.status

        # 1. Mutate live operational runtime parameter
        rt_ok, rt_msg, old_val = self.runtime_registry.apply_runtime_update(
            component=candidate.component,
            parameter_key=candidate.parameter_key,
            new_value=candidate.proposed_value,
            require_active_provider=True,
        )
        if not rt_ok:
            return (False, f"Promotion aborted: {rt_msg}")

        # 2. Register promotion in version registry
        self.registry.register_promotion(
            component=candidate.component,
            new_version=version_tag,
            author=candidate.author,
            approver=approver,
            approver_role=approver_role,
            justification=justification,
        )

        candidate.status = CandidateStatus.APPROVED
        candidate.promoted_at = now_iso
        candidate.promoted_by = approver
        candidate.promotion_role = approver_role

        # 3. Transactional update of candidate in durable store
        if hasattr(self.store, "sqlite_store"):
            try:
                self.store.sqlite_store.save_candidate(candidate)
            except Exception as e:
                # Rollback live runtime change and reconcile version and status (Part E3)
                rev_ok, rev_msg = self.runtime_registry.restore_runtime_state(candidate.component)
                if not rev_ok:
                    self.runtime_registry.mark_recovery_required(
                        candidate.component,
                        f"Failed to revert runtime state after candidate persistence failure: {rev_msg}",
                    )
                self.registry._active_versions[candidate.component] = prev_version
                candidate.status = prev_status
                candidate.promoted_at = None
                candidate.promoted_by = None
                candidate.promotion_role = None
                logger.error("Audit persistence failed: %s. Rolling back runtime and version state.", str(e))
                return (False, f"Promotion aborted: durable audit persistence failed: {e}")

        # 4. Record learning event transactionally
        try:
            self.record(
                LearningEvent(
                    event_id=f"EV-PROM-{candidate_id}",
                    versioned_event_type="MODEL_UPDATE_PROMOTED",
                    module="SYSTEM_LEARNING",
                    timestamp=now_iso,
                    correlation_id=candidate_id,
                    payload={
                        "candidate_id": candidate_id,
                        "component": candidate.component,
                        "new_version": version_tag,
                        "parameter_key": candidate.parameter_key,
                        "old_value": old_val,
                        "new_value": candidate.proposed_value,
                        "runtime_updated": rt_ok,
                        "approver": approver,
                        "approver_role": approver_role,
                        "justification": justification,
                    },
                    associated_versions=self.registry.get_active_versions(),
                )
            )
        except Exception as e:
            rev_ok, rev_msg = self.runtime_registry.restore_runtime_state(candidate.component)
            if not rev_ok:
                self.runtime_registry.mark_recovery_required(
                    candidate.component,
                    f"Failed to revert runtime state after promotion audit event failure: {rev_msg}",
                )
            self.registry._active_versions[candidate.component] = prev_version
            candidate.status = prev_status
            candidate.promoted_at = None
            candidate.promoted_by = None
            candidate.promotion_role = None
            if hasattr(self.store, "sqlite_store"):
                try:
                    self.store.sqlite_store.save_candidate(candidate)
                except Exception:
                    pass
            logger.error("Audit event record failed: %s. Rolling back runtime parameter.", str(e))
            return (False, f"Promotion aborted: durable event persistence failed: {e}")

        # Durably associate version tag with runtime config (Part B2)
        try:
            self.runtime_registry.record_promoted_version(candidate.component, version_tag)
        except Exception as e:
            rev_ok, rev_msg = self.runtime_registry.restore_runtime_state(candidate.component, prev_version)
            if not rev_ok:
                self.runtime_registry.mark_recovery_required(
                    candidate.component,
                    f"Failed to restore prior runtime version after promoted config failure: {rev_msg}",
                )
            self.registry._active_versions[candidate.component] = prev_version
            candidate.status = prev_status
            candidate.promoted_at = None
            candidate.promoted_by = None
            candidate.promotion_role = None
            if hasattr(self.store, "sqlite_store"):
                try:
                    self.store.sqlite_store.save_candidate(candidate)
                except Exception:
                    pass
            logger.error("Failed to durably record promoted version: %s", e)
            return (False, f"Promotion aborted: durable runtime version configuration persistence failed: {e}")

        return (True, f"Candidate {candidate_id} successfully promoted to active version {version_tag}. Runtime state updated.")

    def rollback_component(
        self,
        component: str,
        target_version: str,
        author: str,
        approver_role: str,
        justification: str,
    ) -> Tuple[bool, str]:
        """
        Rolls back a subsystem to a prior approved version and restores live runtime parameter state.
        Fails closed if target_version does not have a restorable runtime configuration.
        """
        comp = component.upper()
        prev_version = self.registry.get_active_versions().get(comp, "v1.0.0")

        # Check target-specific restorable configuration exists before mutating state
        if not self.runtime_registry.has_version_config(comp, target_version):
            return (
                False,
                f"Rollback rejected: Target version '{target_version}' has no valid restorable runtime configuration for {comp}.",
            )

        # 1. Restore live operational runtime parameter configuration to requested target version
        rt_ok, rt_msg = self.runtime_registry.restore_runtime_state(comp, target_version)
        if not rt_ok:
            return (False, f"Runtime rollback failed: {rt_msg}")

        # 2. Reconcile version registry
        success, msg = self.registry.rollback(
            component=comp,
            target_version=target_version,
            author=author,
            approver_role=approver_role,
            justification=justification,
        )

        if not success:
            # Revert runtime state if version governance rejected
            rev_ok, rev_msg = self.runtime_registry.restore_runtime_state(comp, prev_version)
            if not rev_ok:
                self.runtime_registry.mark_recovery_required(
                    comp,
                    f"Failed to restore prior runtime state after governance rejection: {rev_msg}",
                )
            return (False, f"Version rollback failed: {msg}")

        try:
            self.record(
                LearningEvent(
                    event_id=f"EV-ROLLBACK-{comp}",
                    versioned_event_type="MODEL_ROLLED_BACK",
                    module="SYSTEM_LEARNING",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    correlation_id=f"ROLLBACK-{comp}",
                    payload={
                        "component": comp,
                        "target_version": target_version,
                        "approver_role": approver_role,
                        "justification": justification,
                        "runtime_restored": True,
                    },
                    associated_versions=self.registry.get_active_versions(),
                )
            )
        except Exception as e:
            # Revert runtime state and version registry state on audit failure
            rev_ok, rev_msg = self.runtime_registry.restore_runtime_state(comp, prev_version)
            self.registry._active_versions[comp] = prev_version
            if not rev_ok:
                self.runtime_registry.mark_recovery_required(
                    comp,
                    f"Failed to restore prior runtime state after rollback audit logging failure: {rev_msg}",
                )
            logger.error("Audit logging failed during rollback: %s. Rolling back to %s.", e, prev_version)
            return (False, f"Rollback aborted: durable audit persistence failed: {e}")

        return (True, f"Component {comp} successfully rolled back to {target_version}. Runtime parameters restored.")

    def get_candidate(self, candidate_id: str) -> Optional[CandidateUpdate]:
        return self._candidates.get(candidate_id)

    def get_all_candidates(self) -> List[CandidateUpdate]:
        return list(self._candidates.values())
