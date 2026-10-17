"""Data schema definitions, field standards, and column configurations."""

from typing import Dict, List, Set

# Raw challenge columns
TIMESTAMP_COL = "timestamp"
DEPOSITS_COL = "deposits_usd"
WITHDRAWALS_COL = "withdrawals_usd"
NET_FLOW_COL = "net_flow_usd"
REPO_SWEEP_OPEN_COL = "repo_sweep_window_open"
REPO_SWEEP_MATURE_COL = "repo_sweep_mature"
TIER1_REPO_RATE_COL = "tier1_repo_rate_apy"
TIER2_BOND_YIELD_COL = "tier2_bond_yield_pct"
TIER2_BOND_PRICE_COL = "tier2_bond_price_index"
TIER2_BOND_DURATION_COL = "tier2_bond_duration_years"
TIER3_STOCK_PRICE_COL = "tier3_stock_index_price"
TIER3_STOCK_VOLATILITY_COL = "tier3_stock_volatility_pct"

# Supplementary penalty and trading columns
TIER1_EARLY_EXIT_FEE_COL = "penalty_rate_tier1_early_exit"
TIER2_EMERGENCY_HAIRCUT_COL = "haircut_rate_tier2_emergency"
TIER3_EMERGENCY_SLIPPAGE_COL = "slippage_rate_tier3_emergency"
MARKET_HOURS_OPEN_COL = "market_hours_open"

# Documented core fields from challenge brief
MANDATORY_NUMERIC_COLUMNS: List[str] = [
    DEPOSITS_COL,
    WITHDRAWALS_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_YIELD_COL,
    TIER2_BOND_PRICE_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
]

RATE_COLUMNS: Set[str] = {
    TIER1_REPO_RATE_COL,
    TIER2_BOND_YIELD_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIER1_EARLY_EXIT_FEE_COL,
    TIER2_EMERGENCY_HAIRCUT_COL,
    TIER3_EMERGENCY_SLIPPAGE_COL,
}

# Default financial assumptions
DEFAULT_ASSUMPTIONS: Dict[str, float] = {
    TIER1_EARLY_EXIT_FEE_COL: 0.005,      # 0.5% early break fee
    TIER2_EMERGENCY_HAIRCUT_COL: 0.02,     # 2.0% emergency bond haircut
    TIER3_EMERGENCY_SLIPPAGE_COL: 0.15,    # 15.0% emergency equity slippage
    TIER2_BOND_DURATION_COL: 1.9,          # ~1.9 years duration for 2-year bonds
}

# Repo timing conventions
REPO_SWEEP_OPEN_HOUR = 17   # 17:00 local operating time
REPO_SWEEP_MATURE_HOUR = 8   # 08:00 local operating time next morning
