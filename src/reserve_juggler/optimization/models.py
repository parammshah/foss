"""
Models and types for Operations Research treasury action optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class ActionType(str, Enum):
    """Feasible treasury actions."""

    KEEP_CASH = "KEEP_CASH"
    PLACE_REPO = "PLACE_REPO"
    WAIT_REPO_MATURITY = "WAIT_REPO_MATURITY"
    BREAK_REPO_EARLY = "BREAK_REPO_EARLY"
    SELL_BOND = "SELL_BOND"
    SELL_EQUITY = "SELL_EQUITY"
    DRAW_BRIDGE_CREDIT = "DRAW_BRIDGE_CREDIT"
    REPAY_BRIDGE_CREDIT = "REPAY_BRIDGE_CREDIT"


@dataclass(frozen=True)
class TierExposureLimits:
    """Configurable experimental limits for new allocations by TSS mode."""

    tier2_max_pct: float  # Sovereign bonds cap as % of eligible surplus
    tier3_max_pct: float  # Equities cap as % of eligible surplus

    def to_dict(self) -> dict[str, float]:
        return {
            "tier2_max_pct": self.tier2_max_pct,
            "tier3_max_pct": self.tier3_max_pct,
        }


def get_tier_limits_for_tss(tss_score: float) -> TierExposureLimits:
    """Get prototype tier exposure limits based on Treasury Safety Score.

    TSS 0–39: Tier 2 cap 0%, Tier 3 cap 0%
    TSS 40–59: Tier 2 cap 10%, Tier 3 cap 0%
    TSS 60–79: Tier 2 cap 25%, Tier 3 cap 5%
    TSS 80–100: Tier 2 cap 35%, Tier 3 cap 10%
    """
    if tss_score < 40.0:
        return TierExposureLimits(tier2_max_pct=0.0, tier3_max_pct=0.0)
    elif tss_score < 60.0:
        return TierExposureLimits(tier2_max_pct=0.10, tier3_max_pct=0.0)
    elif tss_score < 80.0:
        return TierExposureLimits(tier2_max_pct=0.25, tier3_max_pct=0.05)
    else:
        return TierExposureLimits(tier2_max_pct=0.35, tier3_max_pct=0.10)


@dataclass
class TreasuryAction:
    """Candidate or selected treasury action with full cost and risk accounting."""

    action_type: ActionType
    amount: float
    asset_id: Optional[str] = None
    expected_net_value: float = 0.0
    total_estimated_cost: float = 0.0
    risk_penalty: float = 0.0
    is_feasible: bool = True
    hard_gate_passed: bool = True
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    timing_hours: float = 0.0
    repayment_feasibility: bool = True
    reason_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "asset_id": self.asset_id,
            "amount": round(self.amount, 2),
            "expected_net_value": round(self.expected_net_value, 4),
            "total_estimated_cost": round(self.total_estimated_cost, 4),
            "risk_penalty": round(self.risk_penalty, 4),
            "is_feasible": self.is_feasible,
            "hard_gate_passed": self.hard_gate_passed,
            "cost_breakdown": {k: round(v, 4) for k, v in self.cost_breakdown.items()},
            "timing_hours": round(self.timing_hours, 2),
            "repayment_feasibility": self.repayment_feasibility,
            "reason_code": self.reason_code,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """Result of optimize_treasury_actions."""

    selected_action: TreasuryAction
    candidate_actions: list[TreasuryAction]
    objective_value: float
    eligible_surplus: float
    tier_caps_applied: TierExposureLimits
    tier_breaches_detected: list[str] = field(default_factory=list)
    remediation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action.to_dict(),
            "candidate_actions": [a.to_dict() for a in self.candidate_actions],
            "objective_value": round(self.objective_value, 4),
            "eligible_surplus": round(self.eligible_surplus, 2),
            "tier_caps_applied": self.tier_caps_applied.to_dict(),
            "tier_breaches_detected": list(self.tier_breaches_detected),
            "remediation_notes": list(self.remediation_notes),
        }
