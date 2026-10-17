"""
Data models and state representations for System-Wide Self-Learning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    EVALUATING = "EVALUATING"
    SHADOW_MODE = "SHADOW_MODE"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class SubsystemName(str, Enum):
    FORECASTING = "FORECASTING"
    DECISION_OPTIMIZER = "DECISION_OPTIMIZER"
    SENTINEL_RISK = "SENTINEL_RISK"
    SAFETY_TSS = "SAFETY_TSS"
    EXPLAINABILITY = "EXPLAINABILITY"


@dataclass
class MatchedOutcomePair:
    correlation_id: str
    decision_id: str
    timestamp: str
    # Predicted
    predicted_outflow: float
    expected_cost: float
    selected_action: str
    target_asset: str
    model_version: str
    # Realized
    realized_amount: float
    realized_cost: float
    liquidity_shortfall: float
    settlement_status: str
    # Error computations
    cost_error: float = 0.0
    cost_error_pct: float = 0.0
    is_shortfall: bool = False


@dataclass
class ForecastPerformanceMetrics:
    total_predictions: int
    mape: float
    rmse: float
    interval_coverage_pct: float  # Percent of actuals inside [p05, p95]
    drift_detected: bool
    drift_p_value: float
    status: str  # "STABLE", "DEGRADED", "INCONCLUSIVE"


@dataclass
class DecisionPerformanceMetrics:
    total_decisions: int
    avg_cost_slippage_bps: float
    total_shortfalls: int
    total_shortfall_amount: float
    hard_gate_pass_rate: float
    human_override_rate: float
    human_approval_rate: float


@dataclass
class SentinelPerformanceMetrics:
    total_signals: int
    precision: float
    false_alarm_rate: float
    corroboration_rate: float


@dataclass
class SystemPerformanceReport:
    start_time: str
    end_time: str
    total_learning_events: int
    forecast_metrics: ForecastPerformanceMetrics
    decision_metrics: DecisionPerformanceMetrics
    sentinel_metrics: SentinelPerformanceMetrics
    overall_system_health: str  # "EXCELLENT", "HEALTHY", "WARNING", "ACTION_REQUIRED"
    active_versions: Dict[str, str]
    inconclusive_reason: Optional[str] = None


@dataclass
class CandidateUpdate:
    candidate_id: str
    component: str  # e.g. "FORECASTING", "DECISION_OPTIMIZER", "SENTINEL_RISK"
    parameter_key: str
    current_value: Any
    proposed_value: Any
    description: str
    author: str
    created_at: str
    status: CandidateStatus = CandidateStatus.PROPOSED
    # Validation results
    holdout_evaluation_score: float = 0.0  # Accuracy metric or loss improvement
    holdout_sample_size: int = 0
    stress_shortfall_count: int = 0  # CRITICAL: must be 0
    safety_check_passed: bool = False
    safety_violations: List[str] = field(default_factory=list)
    shadow_mode_runs: int = 0
    shadow_win_rate: float = 0.0
    # Audit trail
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    promotion_role: Optional[str] = None
    rejection_reason: Optional[str] = None
