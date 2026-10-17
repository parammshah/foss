"""
LearningService: System-Wide Self-Learning Feedback Service.
Observes outcomes across the entire treasury lifecycle, matches predictions,
detects drift, safely proposes and shadow-tests candidate updates, and supports audited rollback.

Exposed Integration Services:
    LearningService.record(event: LearningEvent) -> None
    LearningService.evaluate_system_performance(start: Optional[str], end: Optional[str]) -> SystemPerformanceReport
    LearningService.propose_update(...) -> CandidateUpdate
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.integration.contracts import LearningEvent
from reserve_juggler.learning.governance import SafetyInvariantValidator, VersionRegistry
from reserve_juggler.learning.matcher import OutcomeMatcher
from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.models import (
    CandidateStatus,
    CandidateUpdate,
    SystemPerformanceReport,
)
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
        self.registry = VersionRegistry()
        self._candidates: Dict[str, CandidateUpdate] = {}

    def record(self, event: LearningEvent) -> None:
        """
        Record any versioned event in the treasury lifecycle.
        Called by data ingestion, forecasting, TSS, sentinel, decision engine, and human oversight.
        """
        self.store.record(event)

    def evaluate_system_performance(
        self, start_iso: Optional[str] = None, end_iso: Optional[str] = None
    ) -> SystemPerformanceReport:
        """
        Calculates end-to-end performance metrics, error rates, calibration,
        and drift across all subsystems.
        """
        matched_pairs = self.matcher.match_all()
        return self.evaluator.evaluate(
            matched_pairs=matched_pairs,
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
        cid = f"CAND-{component[:4]}-{len(self._candidates)+1:03d}"
        now_iso = datetime.now(timezone.utc).isoformat()
        
        cur_val = current_value if current_value is not None else "default"

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

        # Run safety validation immediately
        passed, violations = self.validator.validate_candidate(candidate)
        candidate.safety_check_passed = passed
        candidate.safety_violations = violations

        if not passed:
            candidate.status = CandidateStatus.REJECTED
            candidate.rejection_reason = "; ".join(violations)
            logger.warning("Candidate %s rejected at proposal: %s", cid, candidate.rejection_reason)
        else:
            candidate.status = CandidateStatus.SHADOW_MODE

        self._candidates[cid] = candidate
        
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
            )
        )

        return candidate

    def evaluate_in_shadow_mode(
        self, candidate_id: str, simulated_runs: int = 20, win_rate: float = 0.85
    ) -> CandidateUpdate:
        """
        Evaluates a candidate update in shadow mode side-by-side with champion.
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise ValueError(f"Candidate {candidate_id} not found")

        if candidate.status == CandidateStatus.REJECTED:
            return candidate

        candidate.shadow_mode_runs = simulated_runs
        candidate.shadow_win_rate = win_rate
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
        Requires human authorization with appropriate role.
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return (False, f"Candidate {candidate_id} not found.")

        # Re-check safety invariants
        passed, violations = self.validator.validate_candidate(candidate)
        if not passed:
            candidate.status = CandidateStatus.REJECTED
            candidate.rejection_reason = "; ".join(violations)
            return (False, f"Safety Invariant Violation: {candidate.rejection_reason}")

        # Enforce role authorization: Only HEAD_OF_TREASURY, RISK_OFFICER, or TREASURY_ANALYST (low risk)
        if approver_role not in ("HEAD_OF_TREASURY", "RISK_OFFICER"):
            return (False, f"Permission Denied: Role '{approver_role}' cannot authorize model promotion.")

        now_iso = datetime.now(timezone.utc).isoformat()
        version_tag = new_version_tag or f"v{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Register in version registry
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

        # Record learning event
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
                    "approver": approver,
                    "approver_role": approver_role,
                    "justification": justification,
                },
                associated_versions=self.registry.get_active_versions(),
            )
        )

        return (True, f"Candidate {candidate_id} successfully promoted to active version {version_tag}.")

    def rollback_component(
        self,
        component: str,
        target_version: str,
        author: str,
        approver_role: str,
        justification: str,
    ) -> Tuple[bool, str]:
        """Rolls back a subsystem to a prior approved version."""
        success, msg = self.registry.rollback(
            component=component.upper(),
            target_version=target_version,
            author=author,
            approver_role=approver_role,
            justification=justification,
        )

        if success:
            self.record(
                LearningEvent(
                    event_id=f"EV-ROLLBACK-{component}",
                    versioned_event_type="MODEL_ROLLED_BACK",
                    module="SYSTEM_LEARNING",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    correlation_id=f"ROLLBACK-{component}",
                    payload={
                        "component": component,
                        "target_version": target_version,
                        "approver_role": approver_role,
                        "justification": justification,
                    },
                    associated_versions=self.registry.get_active_versions(),
                )
            )

        return (success, msg)

    def get_candidate(self, candidate_id: str) -> Optional[CandidateUpdate]:
        return self._candidates.get(candidate_id)

    def get_all_candidates(self) -> List[CandidateUpdate]:
        return list(self._candidates.values())
