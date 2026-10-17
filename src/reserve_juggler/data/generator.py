"""Realistic synthetic generator for 8,760 hourly observations (1 full simulated year).

Implements diurnal volume peaks, merchant settlement batches, weekly seasonality,
overnight repo sweep schedules, interest rate processes, bond yield curves, and
equity index geometric Brownian motion with volatility clustering.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    MARKET_HOURS_OPEN_COL,
    NET_FLOW_COL,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_OPEN_COL,
    TIER1_EARLY_EXIT_FEE_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_DURATION_COL,
    TIER2_BOND_PRICE_COL,
    TIER2_BOND_YIELD_COL,
    TIER2_EMERGENCY_HAIRCUT_COL,
    TIER3_EMERGENCY_SLIPPAGE_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


class RealisticTreasuryDataGenerator:
    """Generates 8,760 hourly observations of realistic digital banking treasury operations."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        start_time: Optional[datetime] = None,
        hours: int = 8760,
    ) -> pd.DataFrame:
        """Simulate exactly `hours` hourly steps (default 8,760 hours = 365 days)."""
        if start_time is None:
            start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]

        # Precompute time arrays
        hour_arr = np.array([ts.hour for ts in timestamps])
        dow_arr = np.array([ts.weekday() for ts in timestamps])  # 0=Monday, 6=Sunday
        doy_arr = np.array([ts.timetuple().tm_yday for ts in timestamps])

        # -------------------------------------------------------------------
        # 1. Customer Cash Flows (Deposits and Withdrawals)
        # Average volume: ~$20,000,000 / hour
        # Diurnal pattern: low at night (01:00 - 05:00: ~5M-8M),
        # high during business hours (11:00 - 16:00: ~25M-35M).
        # Merchant settlement spikes at 23:00 / 00:00 (~28M-32M).
        # -------------------------------------------------------------------
        base_diurnal = 15_000_000.0 + 10_000_000.0 * np.sin(np.pi * (hour_arr - 6) / 12.0)
        base_diurnal = np.maximum(base_diurnal, 6_000_000.0)

        # Merchant settlement batch spike at 23:00
        merchant_spike = np.where(hour_arr == 23, 12_000_000.0, 0.0)

        # Day of week multiplier (weekends have lower customer transaction activity)
        dow_factor = np.where(dow_arr < 5, 1.0, 0.65)
        # Monday morning surge
        monday_surge = np.where((dow_arr == 0) & (hour_arr >= 9) & (hour_arr <= 13), 5_000_000.0, 0.0)

        # Base mean flows
        mean_deposits = (base_diurnal + monday_surge) * dow_factor + self.rng.normal(0, 1_500_000.0, hours)
        mean_withdrawals = (base_diurnal + merchant_spike) * dow_factor + self.rng.normal(0, 1_600_000.0, hours)

        # Ensure positive flows with log-normal noise
        deposits = np.maximum(mean_deposits, 1_000_000.0)
        withdrawals = np.maximum(mean_withdrawals, 1_000_000.0)

        # Introduce occasional liquidity shocks (e.g. corporate tax dates or payroll Fridays)
        shock_mask = (dow_arr == 4) & (hour_arr == 16) & (self.rng.random(hours) < 0.25)
        withdrawals[shock_mask] += self.rng.uniform(10_000_000.0, 20_000_000.0, np.sum(shock_mask))

        net_flow = deposits - withdrawals

        # -------------------------------------------------------------------
        # 2. Repo Sweep Schedule (Tier 1)
        # Window opens at 17:00, mature at 08:00 next morning
        # -------------------------------------------------------------------
        repo_open = (hour_arr == 17).astype(int)
        repo_mature = (hour_arr == 8).astype(int)

        # Tier 1 Repo Rate APY: ~4.3% with small mean-reverting drift
        # dt = 1 / 8760
        repo_rate = np.zeros(hours)
        repo_rate[0] = 0.0430
        for i in range(1, hours):
            # Vasicek mean-reverting process
            repo_rate[i] = repo_rate[i-1] + 0.1 * (0.0430 - repo_rate[i-1]) * (1/8760) + 0.002 * np.sqrt(1/8760) * self.rng.normal()
        repo_rate = np.clip(repo_rate, 0.02, 0.08)

        # -------------------------------------------------------------------
        # 3. Two-Year Sovereign Bonds (Tier 2)
        # Yield ~4.8%, duration ~1.9 years, bond price index around 100.0
        # Price change: dP/P ~ -Duration * dYield + yield_accrual
        # -------------------------------------------------------------------
        bond_yield = np.zeros(hours)
        bond_yield[0] = 0.0480
        bond_price = np.zeros(hours)
        bond_price[0] = 100.0
        duration = 1.9

        for i in range(1, hours):
            # Mean reversion towards 4.8%
            dy = 0.15 * (0.0480 - bond_yield[i-1]) * (1/8760) + 0.003 * np.sqrt(1/8760) * self.rng.normal()
            bond_yield[i] = np.clip(bond_yield[i-1] + dy, 0.025, 0.09)
            # Price movement
            price_ret = -duration * dy + (bond_yield[i-1] / 8760.0)
            bond_price[i] = bond_price[i-1] * (1.0 + price_ret)

        # -------------------------------------------------------------------
        # 4. Equity Index (Tier 3)
        # Expected return ~10.0%, annualized volatility ~18% (0.18)
        # Geometric Brownian Motion with stochastic volatility clustering
        # -------------------------------------------------------------------
        stock_vol = np.zeros(hours)
        stock_vol[0] = 0.180
        stock_price = np.zeros(hours)
        stock_price[0] = 5000.0

        annual_mu = 0.10
        hourly_dt = 1.0 / 8760.0

        for i in range(1, hours):
            # Volatility clustering (AR(1) for volatility)
            stock_vol[i] = np.clip(stock_vol[i-1] + 0.2 * (0.18 - stock_vol[i-1]) * hourly_dt + 0.04 * np.sqrt(hourly_dt) * self.rng.normal(), 0.08, 0.45)
            # Stock price return
            hourly_vol = stock_vol[i] * np.sqrt(hourly_dt)
            hourly_mu = (annual_mu - 0.5 * (stock_vol[i]**2)) * hourly_dt
            ret = np.exp(hourly_mu + hourly_vol * self.rng.normal())
            stock_price[i] = stock_price[i-1] * ret

        # -------------------------------------------------------------------
        # 5. Market Trading Hours and Penalties
        # -------------------------------------------------------------------
        is_weekday = dow_arr < 5
        is_market_hour = (hour_arr >= 14) & (hour_arr < 21)
        market_open = (is_weekday & is_market_hour).astype(int)

        early_exit_fee = np.full(hours, 0.005)      # 0.5% early break penalty
        emergency_haircut = np.full(hours, 0.02)    # 2.0% bond liquidation haircut
        emergency_slippage = np.full(hours, 0.15)   # 15.0% equity liquidation slippage

        # Build clean DataFrame
        df = pd.DataFrame({
            TIMESTAMP_COL: [ts.isoformat() for ts in timestamps],
            DEPOSITS_COL: np.round(deposits, 2),
            WITHDRAWALS_COL: np.round(withdrawals, 2),
            NET_FLOW_COL: np.round(net_flow, 2),
            REPO_SWEEP_OPEN_COL: repo_open,
            REPO_SWEEP_MATURE_COL: repo_mature,
            TIER1_REPO_RATE_COL: np.round(repo_rate, 5),
            TIER2_BOND_YIELD_COL: np.round(bond_yield, 5),
            TIER2_BOND_PRICE_COL: np.round(bond_price, 4),
            TIER2_BOND_DURATION_COL: np.full(hours, duration),
            TIER3_STOCK_PRICE_COL: np.round(stock_price, 2),
            TIER3_STOCK_VOLATILITY_COL: np.round(stock_vol, 4),
            TIER1_EARLY_EXIT_FEE_COL: early_exit_fee,
            TIER2_EMERGENCY_HAIRCUT_COL: emergency_haircut,
            TIER3_EMERGENCY_SLIPPAGE_COL: emergency_slippage,
            MARKET_HOURS_OPEN_COL: market_open,
        })

        return df

    def save_to_csv(self, file_path: str | Path, hours: int = 8760) -> Path:
        """Generate and save the challenge dataset to disk."""
        df = self.generate(hours=hours)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
