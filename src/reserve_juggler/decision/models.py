"""
Decision models and DecisionRecord implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from reserve_juggler.decision.protocols import DecisionRecord


@dataclass
class StandardDecisionRecord:
    """Canonical DecisionRecord implementation matching shared integration protocols."""

    decision_id: str
    timestamp: datetime
    selected_action: dict[str, Any]
    alternatives: list[dict[str, Any]]
    score: dict[str, Any]
    reserve: dict[str, Any]
    eligible_surplus: float
    cost_breakdown: dict[str, float]
    hard_gate_results: dict[str, Any]
    reason_codes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert decision record to JSON-serializable dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "selected_action": self.selected_action,
            "alternatives": self.alternatives,
            "score": self.score,
            "reserve": self.reserve,
            "eligible_surplus": round(self.eligible_surplus, 2),
            "cost_breakdown": {k: round(v, 4) for k, v in self.cost_breakdown.items()},
            "hard_gate_results": self.hard_gate_results,
            "reason_codes": list(self.reason_codes),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert decision record to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @property
    def target_asset(self) -> str:
        if isinstance(self.selected_action, dict):
            return str(self.selected_action.get("target_asset") or self.selected_action.get("asset_id") or "CASH")
        return "CASH"

    @property
    def amount(self) -> float:
        if isinstance(self.selected_action, dict):
            return float(self.selected_action.get("amount") or self.selected_action.get("amount_usd") or 0.0)
        return 0.0

    @property
    def action_type(self) -> str:
        if isinstance(self.selected_action, dict):
            return str(self.selected_action.get("action_type") or self.selected_action.get("type") or "HOLD_CASH")
        return str(self.selected_action)

    @property
    def estimated_costs(self) -> dict[str, float]:
        return self.cost_breakdown

    @property
    def hard_gate_outcomes(self) -> list[Any]:
        from reserve_juggler.contracts import HardGateOutcome, GateStatus
        res = []
        for name, outcome in self.hard_gate_results.items():
            if isinstance(outcome, dict):
                st = GateStatus.PASS if outcome.get("passed", False) else GateStatus.FAIL
                res.append(HardGateOutcome(
                    gate_name=name,
                    status=st,
                    threshold_value=float(outcome.get("threshold", 0.0) or 0.0),
                    observed_value=float(outcome.get("observed", 0.0) or 0.0),
                    reason=str(outcome.get("reason", "")),
                ))
            elif isinstance(outcome, HardGateOutcome):
                res.append(outcome)
        return res

    @property
    def model_versions(self) -> dict[str, str]:
        return self.metadata.get("model_versions", {"decision_engine": "v2.0.0"})

    @property
    def treasury_safety_score(self) -> Optional[float]:
        if isinstance(self.score, dict):
            val = self.score.get("treasury_safety_score") or self.score.get("composite_score")
            return float(val) if val is not None else None
        return None

    @property
    def tss_factors(self) -> dict[str, float]:
        if isinstance(self.score, dict):
            return self.score.get("tss_factors") or self.score.get("factors") or {}
        return {}
