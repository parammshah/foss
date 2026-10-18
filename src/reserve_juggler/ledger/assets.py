"""Asset tier specifications, valuation mechanics, and liquidation calculators."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional
from reserve_juggler.contracts import AssetPosition, AssetTier, ensure_utc


class AssetTierConfig:
    """Configurable financial parameters for an asset tier."""

    def __init__(
        self,
        tier: AssetTier,
        name: str,
        annual_rate: float,
        early_exit_fee_rate: float = 0.0,
        liquidation_haircut_rate: float = 0.0,
        settlement_delay_hours: int = 0,
        annualized_volatility: Optional[float] = None,
        duration_years: Optional[float] = None,
    ) -> None:
        self.tier = tier
        self.name = name
        self.annual_rate = annual_rate
        self.early_exit_fee_rate = early_exit_fee_rate
        self.liquidation_haircut_rate = liquidation_haircut_rate
        self.settlement_delay_hours = settlement_delay_hours
        self.annualized_volatility = annualized_volatility
        self.duration_years = duration_years


# Default challenge configurations (fully overridable)
DEFAULT_TIER0_CONFIG = AssetTierConfig(
    tier=AssetTier.TIER_0_CASH,
    name="Operational Cash",
    annual_rate=0.0,
    settlement_delay_hours=0,
)

DEFAULT_TIER1_CONFIG = AssetTierConfig(
    tier=AssetTier.TIER_1_REPO,
    name="Overnight Repo",
    annual_rate=0.043,  # ~4.3% APY
    early_exit_fee_rate=0.005,  # 0.5% early break fee
    settlement_delay_hours=0,
)

DEFAULT_TIER2_CONFIG = AssetTierConfig(
    tier=AssetTier.TIER_2_BONDS,
    name="Two-Year Sovereign Bonds",
    annual_rate=0.048,  # ~4.8% yield
    liquidation_haircut_rate=0.02,  # 2.0% emergency liquidation haircut
    settlement_delay_hours=0,  # 0 for emergency liquidation, 1-2 for standard secondary market
    duration_years=1.9,
)

DEFAULT_TIER3_CONFIG = AssetTierConfig(
    tier=AssetTier.TIER_3_EQUITIES,
    name="Equity Index",
    annual_rate=0.100,  # ~10.0% expected annual return
    liquidation_haircut_rate=0.15,  # 15.0% emergency liquidation slippage
    settlement_delay_hours=0,
    annualized_volatility=0.180,  # 18% annualized volatility
)


def calculate_liquidatable_value(
    tier: AssetTier,
    market_value_usd: float,
    early_exit_fee_rate: float,
    liquidation_haircut_rate: float,
    is_matured: bool = False,
) -> float:
    """Computes immediately realizable net cash proceeds from liquidating a position."""
    if tier == AssetTier.TIER_0_CASH:
        return market_value_usd

    if tier == AssetTier.TIER_1_REPO:
        # If matured, 100% of principal + interest is realizable without fee.
        # If broken before maturity, deduct early exit penalty.
        fee = 0.0 if is_matured else market_value_usd * early_exit_fee_rate
        return max(0.0, market_value_usd - fee)

    if tier in (AssetTier.TIER_2_BONDS, AssetTier.TIER_3_EQUITIES):
        # Apply haircut / market slippage
        net = market_value_usd * (1.0 - liquidation_haircut_rate)
        return max(0.0, net)

    return market_value_usd
