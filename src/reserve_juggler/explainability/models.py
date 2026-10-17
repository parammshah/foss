"""
Data models for Decision Explanation and Human Oversight.
Ensures zero fabricated fallbacks and accurate quantile terminology.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from reserve_juggler.contracts import (
    AlternativeAction,
    ApprovalStatus,
    GateStatus,
    HardGateOutcome,
    UserRole,
)


@dataclass
class CostBreakdown:
    interest_cost: float
    tax_impact: float
    liquidation_penalty: float
    spread_slippage: float
    total_cost: float


@dataclass
class ForecastContextSummary:
    is_available: bool
    horizon_1h_flow: Optional[float]
    horizon_6h_flow: Optional[float]
    horizon_24h_flow: Optional[float]
    interval_p05: Optional[float]
    interval_p95: Optional[float]
    forecast_confidence: Optional[float]
    quality_status: str
    quantile_description: str = (
        "Quantiles represent cumulative cash flow distribution percentiles (P05 lower tail, P95 upper tail), "
        "not bank survival probabilities."
    )


@dataclass
class AlternativeComparisonEntry:
    action_type: str
    target_asset: str
    amount: float
    estimated_total_cost: float
    cost_delta_vs_selected: float
    settlement_horizon_hours: int
    rejection_reason: str
    is_feasible: bool


@dataclass
class DecisionExplanation:
    decision_id: str
    timestamp: str
    recommended_action: str
    target_asset: str
    recommended_amount: float
    protected_reserve_floor: Optional[float]
    eligible_surplus: float
    # TSS
    treasury_safety_score: Optional[float]
    tss_factors: Dict[str, float]
    # Detailed Contexts
    forecast_context: ForecastContextSummary
    cost_breakdown: CostBreakdown
    settlement_days: int
    settlement_note: str
    hard_gates: List[HardGateOutcome]
    all_hard_gates_passed: bool
    gate_evaluation_status: str  # "PASS", "FAIL", "NOT_EVALUATED", "UNKNOWN"
    # Grounded Rationale
    primary_justification: str
    grounded_narrative: str
    alternatives_comparison: List[AlternativeComparisonEntry]
    disclosed_missing_information: List[str]
    # Lineage
    model_versions: Dict[str, str]
    policy_reference: str
    governance_notice: str = (
        "Internal governance decision support. Does not constitute statutory regulatory certification."
    )

    @property
    def primary_rationale(self) -> str:
        return self.primary_justification

    @property
    def rejected_alternatives(self) -> List[AlternativeComparisonEntry]:
        return [alt for alt in self.alternatives_comparison if not alt.is_feasible or alt.rejection_reason]


@dataclass
class HumanActionRecord:
    action_id: str
    decision_id: str
    actor_name: str
    actor_role: str
    action_type: str  # "APPROVED", "REJECTED", "MODIFIED", "CANCELLED", "OVERRIDDEN"
    timestamp: str
    stated_reason: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    revalidation_passed: bool = False
    revalidation_notes: List[str] = field(default_factory=list)
    simulated_execution_status: str = "PENDING"
