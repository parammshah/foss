"""
Capital Firewall: Hot/Core Capital Partitioning and Holding-Time Gatekeeper.

Enforces economic segmentation between operational hot money and investable core capital,
and applies holding-time screens for bond investments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from reserve_juggler.decision.protocols import (
        AssetPosition,
        ForecastBundle,
        TreasuryState,
    )
from reserve_juggler.risk.models import (
    CapitalClassification,
    HoldingTimeGateResult,
    ReserveAssessment,
)


def classify_hot_core_capital(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    reserve_assessment: ReserveAssessment,
) -> CapitalClassification:
    """Classify treasury assets into Hot Money and Core Capital.

    Hot Money:
    Short-term, variable funding strictly needed for daily or intraday obligations,
    mandatory customer payouts, and forecast operational deficits.
    Must be kept in liquid cash or short-dated repos where the maturity lockup
    is completely safe and settles prior to obligations.

    Core Capital:
    Structural surplus remaining after full reserve and stress deductions.
    Eligible portions may be evaluated for higher-tier investments (sovereign bonds
    and capped growth equities) subject to tier exposure limits and holding gates.
    """
    total_asset_value = float(state.available_cash) + sum(
        float(p.market_value) for p in state.asset_positions
    )

    # Hot money requirement is defined by total required reserves
    hot_money_required = (
        reserve_assessment.protected_capital + reserve_assessment.future_obligation_reserve
    )

    # Core capital is portfolio value beyond hot money requirement
    core_capital = max(0.0, total_asset_value - hot_money_required)

    # The portion of core capital that is currently unencumbered cash ready for investment
    # If forecast indicates structural net cash burn (withdrawals > deposits), all surplus is Hot Money protecting payouts
    is_net_cash_burn = False
    longest_point = None
    if forecast_bundle.point_forecasts:
        longest_point = max(forecast_bundle.point_forecasts, key=lambda p: float(p.forecast_horizon))
        if float(longest_point.withdrawal_p50) > float(longest_point.deposit_p50) or float(longest_point.withdrawal_p90) > float(longest_point.deposit_p10):
            is_net_cash_burn = True

    if is_net_cash_burn and longest_point is not None:
        investable_core = 0.0
        rationale = (
            f"Treasury partitioned into ${hot_money_required:,.2f} Hot Money "
            f"(protecting payouts and obligations). Core capital of ${core_capital:,.2f} has $0.00 "
            f"investable surplus because forecast indicates net cash burn "
            f"(P50 withdrawals exceed deposits over {longest_point.forecast_horizon:.0f}h horizon)."
        )
    else:
        investable_core = min(core_capital, reserve_assessment.eligible_investable_surplus)
        rationale = (
            f"Treasury partitioned into ${hot_money_required:,.2f} Hot Money "
            f"(protecting payouts and obligations) and ${core_capital:,.2f} Core Capital "
            f"(with ${investable_core:,.2f} immediate unencumbered investable surplus)."
        )

    return CapitalClassification(
        hot_money=hot_money_required,
        core_capital=core_capital,
        investable_core_capital=investable_core,
        total_treasury_assets=total_asset_value,
        rationale=rationale,
    )


def calculate_holding_time_gate(
    asset_position: AssetPosition,
    expected_holding_days: float,
) -> HoldingTimeGateResult:
    """Calculate the preliminary minimum holding-time screen for bond investments.

    For sovereign bonds (Tier 2):
    Calculates the breakeven accrual period where gross earned yield equals the liquidation haircut:
        minimum_days = (haircut / annual_yield) * 365

    Example:
    Haircut = 0.02 (2.0%), annual yield = 0.048 (4.8%):
    minimum_days = (0.02 / 0.048) * 365 = 152.08 days.

    Important Constraints:
    - This is a preliminary screen, not a guaranteed holding period. Market losses, taxes,
      fees, and future liquidity needs must be evaluated separately.
    - Simplistic coupon accrual must NEVER be applied to growth equities (Tier 3),
      as equities have market price volatility, non-linear slippage, and non-contractual returns.
    """
    instrument_type = getattr(asset_position, "instrument_type", "").lower()
    tier = getattr(asset_position, "tier", 1)
    haircut = float(getattr(asset_position, "haircut", 0.0))
    annual_yield = float(getattr(asset_position, "annual_yield", 0.0))
    asset_id = getattr(asset_position, "asset_id", "UNKNOWN")

    # Equities: simplistic bond formula is explicitly invalid
    if tier == 3 or instrument_type == "equity":
        return HoldingTimeGateResult(
            passed=False,
            asset_id=asset_id,
            instrument_type=instrument_type,
            minimum_holding_days=float("inf"),
            expected_holding_days=expected_holding_days,
            haircut=haircut,
            annual_yield=annual_yield,
            notes=(
                "REJECTED: Simplistic holding-time formula cannot be applied to equities. "
                "Equities must be evaluated under volatility, drawdowns, and slippage hurdles."
            ),
        )

    # Cash / Zero haircut
    if haircut <= 0.0:
        return HoldingTimeGateResult(
            passed=True,
            asset_id=asset_id,
            instrument_type=instrument_type,
            minimum_holding_days=0.0,
            expected_holding_days=expected_holding_days,
            haircut=haircut,
            annual_yield=annual_yield,
            notes="Zero liquidation haircut; holding time gate automatically passes.",
        )

    if annual_yield <= 0.0:
        return HoldingTimeGateResult(
            passed=False,
            asset_id=asset_id,
            instrument_type=instrument_type,
            minimum_holding_days=float("inf"),
            expected_holding_days=expected_holding_days,
            haircut=haircut,
            annual_yield=annual_yield,
            notes="Positive liquidation haircut with zero or negative yield can never break even.",
        )

    # Standard formula for bonds: (haircut / annual_yield) * 365
    minimum_days = (haircut / annual_yield) * 365.0
    passed = expected_holding_days >= minimum_days

    notes = (
        f"Breakeven holding period is {minimum_days:.1f} days (haircut {haircut*100:.1f}%, yield {annual_yield*100:.1f}%). "
        f"Planned holding is {expected_holding_days:.1f} days. "
        + ("Passed hurdle." if passed else "Failed hurdle: liquidated prior to breakeven accrual.")
    )

    return HoldingTimeGateResult(
        passed=passed,
        asset_id=asset_id,
        instrument_type=instrument_type,
        minimum_holding_days=minimum_days,
        expected_holding_days=expected_holding_days,
        haircut=haircut,
        annual_yield=annual_yield,
        notes=notes,
    )
