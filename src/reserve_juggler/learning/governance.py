"""
Governance, Safety Invariants, and Version Rollback Registry.
Enforces non-negotiable safety rules on candidate updates and records audited rollback history.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from reserve_juggler.learning.models import CandidateUpdate

logger = logging.getLogger(__name__)


class SafetyInvariantViolation(Exception):
    """Raised when an update violates fundamental treasury safety invariants."""
    pass


class SafetyInvariantValidator:
    """
    Enforces non-negotiable safety rules on candidate updates:
    - Never remove or lower mandatory liquidity reserves below statutory floors.
    - Never bypass credit limits or asset eligibility.
    - Never rewrite statutory tax rules.
    - Never disable hard safety constraints.
    - Never approve own high-risk policy changes.
    - Reject candidates that improve accuracy but increase liquidity shortfalls.
    """

    FORBIDDEN_PARAMETER_PATTERNS = [
        "bypass_reserve",
        "disable_gate",
        "override_tax_rate",
        "ignore_credit_limit",
        "force_unverified_liquidation",
        "disable_firewall",
        "reduce_reserve",
    ]

    ALLOWED_COMPONENTS = {
        "FORECASTING",
        "DECISION_OPTIMIZER",
        "SENTINEL_RISK",
        "SCENARIO_GENERATOR",
        "EXPLAINABILITY",
    }

    def validate_candidate(self, candidate: CandidateUpdate) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        # 1. Component check
        if candidate.component not in self.ALLOWED_COMPONENTS:
            violations.append(
                f"Unauthorized component '{candidate.component}'. Only approved subsystems may be updated."
            )

        # 2. Forbidden parameter checks
        param_lower = candidate.parameter_key.lower()
        for forbidden in self.FORBIDDEN_PARAMETER_PATTERNS:
            if forbidden in param_lower:
                violations.append(
                    f"Safety Invariant Violation: Parameter '{candidate.parameter_key}' violates safety firewall rule '{forbidden}'."
                )

        # 3. Mandatory Reserve protection
        if "reserve_floor" in param_lower or "mandatory_reserve" in param_lower:
            if isinstance(candidate.proposed_value, (int, float)) and isinstance(candidate.current_value, (int, float)):
                if candidate.proposed_value < candidate.current_value:
                    violations.append(
                        "Safety Invariant Violation: Autonomous learning may never reduce the mandatory liquidity reserve floor."
                    )

        # 4. Tax code integrity
        if "tax" in param_lower or "statutory" in param_lower:
            violations.append(
                "Safety Invariant Violation: Statutory tax rules and schedules cannot be modified by machine learning."
            )

        # 5. Stress shortfall check: Never promote a candidate that induces liquidity deficit
        if candidate.stress_shortfall_count > 0:
            violations.append(
                f"Safety Invariant Violation: Candidate generated {candidate.stress_shortfall_count} liquidity shortfalls under stress testing. Promotion prohibited."
            )

        # 6. Sample size check (avoid fabricating improvement)
        if candidate.holdout_sample_size < 3:
            violations.append(
                f"Inconclusive holdout data: Sample size ({candidate.holdout_sample_size}) is insufficient for safe statistical validation."
            )

        return (len(violations) == 0, violations)


@dataclass
class VersionAuditEntry:
    entry_id: str
    component: str
    previous_version: str
    new_version: str
    timestamp: str
    author: str
    approver: str
    approver_role: str
    action_type: str  # "PROMOTION", "ROLLBACK"
    justification: str


class VersionRegistry:
    """
    Tracks version lineage and rollback history for all subsystems.
    """

    def __init__(self, persistence_store: Optional[Any] = None):
        self.persistence_store = persistence_store
        self._active_versions: Dict[str, str] = {
            "FORECASTING": "v1.4.0",
            "DECISION_OPTIMIZER": "v1.2.0",
            "SENTINEL_RISK": "v1.0.0",
            "SCENARIO_GENERATOR": "v1.1.0",
            "EXPLAINABILITY": "v1.0.0",
        }
        self._version_history: Dict[str, List[str]] = {
            k: [v] for k, v in self._active_versions.items()
        }
        self._audit_trail: List[VersionAuditEntry] = []

    def get_active_versions(self) -> Dict[str, str]:
        return dict(self._active_versions)

    def get_active_version(self, component: str) -> str:
        return self._active_versions.get(component.upper(), "v1.0.0")

    def register_promotion(
        self,
        component: str,
        new_version: str,
        author: str,
        approver: str,
        approver_role: str,
        justification: str,
    ) -> None:
        """Register a promoted candidate as the new active version."""
        comp = component.upper()
        prev = self._active_versions.get(comp, "v1.0.0")
        self._active_versions[comp] = new_version
        self._version_history.setdefault(comp, []).append(new_version)

        entry = VersionAuditEntry(
            entry_id=f"AUD-{len(self._audit_trail)+1}",
            component=comp,
            previous_version=prev,
            new_version=new_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            author=author,
            approver=approver,
            approver_role=approver_role,
            action_type="PROMOTION",
            justification=justification,
        )
        self._audit_trail.append(entry)

        if self.persistence_store and hasattr(self.persistence_store, "sqlite_store"):
            try:
                self.persistence_store.sqlite_store.save_version_audit({
                    "entry_id": entry.entry_id,
                    "component": entry.component,
                    "previous_version": entry.previous_version,
                    "new_version": entry.new_version,
                    "timestamp": entry.timestamp,
                    "author": entry.author,
                    "approver": entry.approver,
                    "approver_role": entry.approver_role,
                    "action_type": entry.action_type,
                    "justification": entry.justification,
                })
            except Exception as e:
                logger.error("Failed to persist version audit: %s", str(e))

    def rollback(
        self,
        component: str,
        target_version: str,
        author: str,
        approver_role: str,
        justification: str,
    ) -> Tuple[bool, str]:
        """Roll back an active component to a previously approved version."""
        comp = component.upper()
        if approver_role not in ("HEAD_OF_TREASURY", "RISK_OFFICER", "AUDITOR"):
            return (False, f"Permission Denied: Role '{approver_role}' cannot authorize model rollback.")

        history = self._version_history.get(comp, [])
        if target_version not in history:
            return (False, f"Target version '{target_version}' not found in approved version history for {comp}.")

        prev = self._active_versions.get(comp, "v1.0.0")
        self._active_versions[comp] = target_version

        entry = VersionAuditEntry(
            entry_id=f"AUD-{len(self._audit_trail)+1}",
            component=comp,
            previous_version=prev,
            new_version=target_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            author=author,
            approver=author,
            approver_role=approver_role,
            action_type="ROLLBACK",
            justification=justification,
        )
        self._audit_trail.append(entry)

        if self.persistence_store and hasattr(self.persistence_store, "sqlite_store"):
            try:
                self.persistence_store.sqlite_store.save_version_audit({
                    "entry_id": entry.entry_id,
                    "component": entry.component,
                    "previous_version": entry.previous_version,
                    "new_version": entry.new_version,
                    "timestamp": entry.timestamp,
                    "author": entry.author,
                    "approver": entry.approver,
                    "approver_role": entry.approver_role,
                    "action_type": entry.action_type,
                    "justification": entry.justification,
                })
            except Exception as e:
                logger.error("Failed to persist version audit: %s", str(e))

        return (True, f"Successfully rolled back {comp} from {prev} to {target_version}.")

    def get_audit_trail(self) -> List[VersionAuditEntry]:
        return list(self._audit_trail)
