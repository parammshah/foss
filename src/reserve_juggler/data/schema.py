"""Data schema definitions, field standards, and column configurations for Reserve Juggler."""

from typing import Dict, List, Set

# Raw and canonical challenge columns
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
TIER2_BOND_MOD_DURATION_COL = "tier2_bond_mod_duration"
TIER3_STOCK_PRICE_COL = "tier3_stock_index_price"
TIER3_STOCK_VOLATILITY_COL = "tier3_stock_volatility_pct"

# Supplementary penalty and trading columns (actual dataset names and aliases)
TIER1_EARLY_EXIT_FEE_COL = "penalty_rate_tier1_early_exit"
TIER1_EARLY_REDEMPTION_HAIRCUT_COL = "tier1_early_redemption_haircut_pct"

TIER2_EMERGENCY_HAIRCUT_COL = "haircut_rate_tier2_emergency"
TIER2_BOND_LIQUIDATION_HAIRCUT_COL = "tier2_bond_liquidation_haircut_pct"

TIER3_EMERGENCY_SLIPPAGE_COL = "slippage_rate_tier3_emergency"
TIER3_STOCK_LIQUIDATION_HAIRCUT_COL = "tier3_stock_liquidation_haircut_pct"

EMERGENCY_OVERDRAFT_PENALTY_APR_COL = "emergency_overdraft_penalty_apr_pct"
MARKET_HOURS_OPEN_COL = "market_hours_open"

# Bi-directional column mappings between actual dataset headers and canonical internal names
ACTUAL_TO_CANONICAL_COLUMN_MAP: Dict[str, str] = {
    TIER2_BOND_MOD_DURATION_COL: TIER2_BOND_DURATION_COL,
    TIER1_EARLY_REDEMPTION_HAIRCUT_COL: TIER1_EARLY_EXIT_FEE_COL,
    TIER2_BOND_LIQUIDATION_HAIRCUT_COL: TIER2_EMERGENCY_HAIRCUT_COL,
    TIER3_STOCK_LIQUIDATION_HAIRCUT_COL: TIER3_EMERGENCY_SLIPPAGE_COL,
}

CANONICAL_TO_ACTUAL_COLUMN_MAP: Dict[str, str] = {
    v: k for k, v in ACTUAL_TO_CANONICAL_COLUMN_MAP.items()
}

# Mandatory numeric columns for model ingestion
MANDATORY_NUMERIC_COLUMNS: List[str] = [
    DEPOSITS_COL,
    WITHDRAWALS_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_YIELD_COL,
    TIER2_BOND_PRICE_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
]

# Columns representing rates, yields, penalties, and volatilities that require decimal normalization
PERCENTAGE_COLUMNS: Set[str] = {
    TIER1_REPO_RATE_COL,
    TIER2_BOND_YIELD_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIER1_EARLY_EXIT_FEE_COL,
    TIER1_EARLY_REDEMPTION_HAIRCUT_COL,
    TIER2_EMERGENCY_HAIRCUT_COL,
    TIER2_BOND_LIQUIDATION_HAIRCUT_COL,
    TIER3_EMERGENCY_SLIPPAGE_COL,
    TIER3_STOCK_LIQUIDATION_HAIRCUT_COL,
    EMERGENCY_OVERDRAFT_PENALTY_APR_COL,
}

# Backward compatibility alias
RATE_COLUMNS = PERCENTAGE_COLUMNS

# Default financial assumptions if not present in the CSV
DEFAULT_ASSUMPTIONS: Dict[str, float] = {
    TIER1_EARLY_EXIT_FEE_COL: 0.005,      # 0.5% early break fee
    TIER2_EMERGENCY_HAIRCUT_COL: 0.02,     # 2.0% emergency bond haircut
    TIER3_EMERGENCY_SLIPPAGE_COL: 0.15,    # 15.0% emergency equity slippage
    TIER2_BOND_DURATION_COL: 1.92,        # 1.92 years modified duration
    EMERGENCY_OVERDRAFT_PENALTY_APR_COL: 0.12,  # 12.0% emergency overdraft penalty APR
}

# Constant simulation parameters observed in the actual dataset
OBSERVED_ACTUAL_CONSTANTS: Dict[str, float] = {
    TIER2_BOND_MOD_DURATION_COL: 1.92,
    TIER3_STOCK_VOLATILITY_COL: 18.0,
    TIER1_EARLY_REDEMPTION_HAIRCUT_COL: 0.5,
    TIER2_BOND_LIQUIDATION_HAIRCUT_COL: 2.0,
    TIER3_STOCK_LIQUIDATION_HAIRCUT_COL: 15.0,
    EMERGENCY_OVERDRAFT_PENALTY_APR_COL: 12.0,
}

# Repo timing conventions
REPO_SWEEP_OPEN_HOUR = 17   # 17:00 local operating time
REPO_SWEEP_MATURE_HOUR = 8   # 08:00 local operating time next morning
