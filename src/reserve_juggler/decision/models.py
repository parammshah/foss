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
