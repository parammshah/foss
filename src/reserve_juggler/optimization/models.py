"""
Models and types for Operations Research treasury action optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional


class ActionType(str, Enum):
    """Feasible treasury actions."""

    KEEP_CASH = "KEEP_CASH"
    PLACE_REPO = "PLACE_REPO"
    BUY_BOND = "BUY_BOND"
    BUY_EQUITY = "BUY_EQUITY"
    SELL_BOND = "SELL_BOND"
    SELL_EQUITY = "SELL_EQUITY"
    BREAK_REPO_EARLY = "BREAK_REPO_EARLY"
    DRAW_BRIDGE_CREDIT = "DRAW_BRIDGE_CREDIT"
    REPAY_BRIDGE_CREDIT = "REPAY_BRIDGE_CREDIT"
    WAIT_REPO_MATURITY = "WAIT_REPO_MATURITY"
    UNRESOLVED_FUNDING_SHORTFALL = "UNRESOLVED_FUNDING_SHORTFALL"


class OptimizationMode(str, Enum):
    """Optimizer operational mode."""

    EMERGENCY_FUNDING = "EMERGENCY_FUNDING"
    SURPLUS_ALLOCATION = "SURPLUS_ALLOCATION"


@dataclass(frozen=True)
class TierExposureLimits:
    """Configurable limits for new allocations and portfolio exposure by TSS mode."""

    tier2_max_pct: float  # Sovereign bonds cap as % of eligible surplus (new allocation)
    tier3_max_pct: float  # Equities cap as % of eligible surplus (new allocation)
    tier2_portfolio_cap_pct: float = 0.35  # Maximum Tier 2 allocation as % of total portfolio
    tier3_portfolio_cap_pct: float = 0.10  # Maximum Tier 3 allocation as % of total portfolio

    def to_dict(self) -> dict[str, float]:
        return {
            "tier2_max_pct": self.tier2_max_pct,
            "tier3_max_pct": self.tier3_max_pct,
            "tier2_portfolio_cap_pct": self.tier2_portfolio_cap_pct,
            "tier3_portfolio_cap_pct": self.tier3_portfolio_cap_pct,
        }


def get_tier_limits_for_tss(tss_score: float) -> TierExposureLimits:
    """Get prototype tier exposure limits based on Treasury Safety Score.

    TSS 0–39: Tier 2 cap 0%, Tier 3 cap 0%
    TSS 40–59: Tier 2 cap 10%, Tier 3 cap 0%
    TSS 60–79: Tier 2 cap 25%, Tier 3 cap 5%
    TSS 80–100: Tier 2 cap 35%, Tier 3 cap 10%
    """
    if tss_score < 40.0:
        return TierExposureLimits(
            tier2_max_pct=0.0,
            tier3_max_pct=0.0,
            tier2_portfolio_cap_pct=0.0,
            tier3_portfolio_cap_pct=0.0,
        )
    elif tss_score < 60.0:
        return TierExposureLimits(
            tier2_max_pct=0.10,
            tier3_max_pct=0.0,
            tier2_portfolio_cap_pct=0.10,
            tier3_portfolio_cap_pct=0.0,
        )
    elif tss_score < 80.0:
        return TierExposureLimits(
            tier2_max_pct=0.25,
            tier3_max_pct=0.05,
            tier2_portfolio_cap_pct=0.25,
            tier3_portfolio_cap_pct=0.05,
        )
    else:
        return TierExposureLimits(
            tier2_max_pct=0.35,
            tier3_max_pct=0.10,
            tier2_portfolio_cap_pct=0.35,
            tier3_portfolio_cap_pct=0.10,
        )


@dataclass
class HumanOverride:
    """Explicit, auditable human override that supersedes candidate rejection."""

    candidate_id: str
    authorized_by: str
    override_reason: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "authorized_by": self.authorized_by,
            "override_reason": self.override_reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class TreasuryAction:
    """Candidate or selected treasury action with full cost, risk, and funding plan accounting."""

    action_type: ActionType
    amount: float
    candidate_id: str = ""
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

    # Resolution and funding states
    payout_funded: bool = True
    funding_plan_resolved: bool = True
    repayment_plan_feasible: bool = True

    # Time-indexed funding plan attributes
    required_payout_amount: float = 0.0
    existing_cash_applied: float = 0.0
    gross_proceeds: float = 0.0
    net_proceeds: float = 0.0
    transaction_cost: float = 0.0
    availability_delay_hours: float = 0.0
    payment_deadline_hours: float = 0.0
    funded_by_deadline: float = 0.0
    residual_unfunded_amount: float = 0.0
    is_partial_funding: bool = False
    secondary_action_needed: bool = False

    # Bridge-specific audit fields
    principal_borrowed: float = 0.0
    apr: float = 0.0
    annual_day_count_convention: float = 8760.0
    interest_cost: float = 0.0
    expected_repayment_time_hours: Optional[float] = None
    expected_repayment_proceeds: float = 0.0
    repayment_source_desc: str = ""
    payouts_due_before_repayment: float = 0.0
    residual_debt_after_repayment: float = 0.0
    credit_availability_after_draw: float = 0.0
    ranking_is_provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type.value,
            "asset_id": self.asset_id,
            "amount": round(self.amount, 2),
            "expected_net_value": round(self.expected_net_value, 4),
            "total_estimated_cost": round(self.total_estimated_cost, 4),
            "risk_penalty": round(self.risk_penalty, 4),
            "is_feasible": self.is_feasible,
            "hard_gate_passed": self.hard_gate_passed,
            "payout_funded": self.payout_funded,
            "funding_plan_resolved": self.funding_plan_resolved,
            "repayment_plan_feasible": self.repayment_plan_feasible,
            "cost_breakdown": {k: round(v, 4) for k, v in self.cost_breakdown.items()},
            "timing_hours": round(self.timing_hours, 2),
            "repayment_feasibility": self.repayment_feasibility,
            "reason_code": self.reason_code,
            "required_payout_amount": round(self.required_payout_amount, 2),
            "existing_cash_applied": round(self.existing_cash_applied, 2),
            "gross_proceeds": round(self.gross_proceeds, 2),
            "net_proceeds": round(self.net_proceeds, 2),
            "transaction_cost": round(self.transaction_cost, 2),
            "availability_delay_hours": round(self.availability_delay_hours, 2),
            "payment_deadline_hours": round(self.payment_deadline_hours, 2),
            "funded_by_deadline": round(self.funded_by_deadline, 2),
            "residual_unfunded_amount": round(self.residual_unfunded_amount, 2),
            "is_partial_funding": self.is_partial_funding,
            "secondary_action_needed": self.secondary_action_needed,
            "principal_borrowed": round(self.principal_borrowed, 2),
            "apr": self.apr,
            "interest_cost": round(self.interest_cost, 4),
            "expected_repayment_time_hours": (
                round(self.expected_repayment_time_hours, 2)
                if self.expected_repayment_time_hours is not None
                else None
            ),
            "expected_repayment_proceeds": round(self.expected_repayment_proceeds, 2),
            "repayment_source_desc": self.repayment_source_desc,
            "payouts_due_before_repayment": round(self.payouts_due_before_repayment, 2),
            "residual_debt_after_repayment": round(self.residual_debt_after_repayment, 2),
            "credit_availability_after_draw": round(self.credit_availability_after_draw, 2),
            "ranking_is_provisional": self.ranking_is_provisional,
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
    optimization_mode: str = "SURPLUS_ALLOCATION"  # 'EMERGENCY_FUNDING' vs 'SURPLUS_ALLOCATION'
    unresolved_shortfall: float = 0.0
    is_unresolved_shortfall: bool = False
    human_override: Optional[HumanOverride] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action.to_dict(),
            "candidate_actions": [a.to_dict() for a in self.candidate_actions],
            "objective_value": round(self.objective_value, 4),
            "eligible_surplus": round(self.eligible_surplus, 2),
            "tier_caps_applied": self.tier_caps_applied.to_dict(),
            "tier_breaches_detected": list(self.tier_breaches_detected),
            "remediation_notes": list(self.remediation_notes),
            "optimization_mode": self.optimization_mode,
            "unresolved_shortfall": round(self.unresolved_shortfall, 2),
            "is_unresolved_shortfall": self.is_unresolved_shortfall,
            "human_override": self.human_override.to_dict() if self.human_override else None,
        }

