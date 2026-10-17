"""
Data models for Decision Explanation and Human Oversight.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from reserve_juggler.integration.contracts import (
    AlternativeAction, HardGateOutcome, UserRole, ApprovalStatus
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
    horizon_1h_flow: float
    horizon_6h_flow: float
    horizon_24h_flow: float
    interval_p05: float
    interval_p95: float
    forecast_confidence: float
    quality_status: str


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
    protected_reserve_floor: float
    eligible_surplus: float
    # TSS
    treasury_safety_score: float
    tss_factors: Dict[str, float]
    # Detailed Contexts
    forecast_context: ForecastContextSummary
    cost_breakdown: CostBreakdown
    settlement_days: int
    settlement_note: str
    hard_gates: List[HardGateOutcome]
    all_hard_gates_passed: bool
    # Grounded Rationale
    primary_justification: str
    grounded_narrative: str
    alternatives_comparison: List[AlternativeComparisonEntry]
    disclosed_missing_information: List[str]
    # Lineage
    model_versions: Dict[str, str]
    policy_reference: str


@dataclass
class HumanActionRecord:
    action_id: str
    decision_id: str
    actor_name: str
    actor_role: str
    action_type: str  # "APPROVED", "REJECTED", "MODIFIED", "OVERRIDDEN"
    timestamp: str
    stated_reason: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    simulated_execution_status: str = "SIMULATED_SUCCESS"
