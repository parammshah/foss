"""
Reserve Juggler: Enterprise Treasury Optimization Engine.
Team 2: Safety Score, Capital Firewall, Tax-Aware Operations Research & Next Best Action.
"""

from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.optimization.optimizer import optimize_treasury_actions
from reserve_juggler.risk.capital_firewall import (
    calculate_holding_time_gate,
    classify_hot_core_capital,
)
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value

__version__ = "0.2.0"

__all__ = [
    "TreasuryDecisionService",
    "calculate_treasury_safety_score",
    "assess_liquidity_reserve",
    "evaluate_hard_safety_gates",
    "classify_hot_core_capital",
    "calculate_holding_time_gate",
    "optimize_treasury_actions",
    "evaluate_tax_adjusted_value",
]
