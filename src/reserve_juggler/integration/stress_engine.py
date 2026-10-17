"""
Stress Lab Financial Simulation Engine.
Implements multi-asset liquidity accounting and evaluates 3 benchmark strategies
under identical initial conditions, financial parameters, and macro stress shocks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from reserve_juggler.contracts import TreasuryState


@dataclass
class StrategyStressOutcome:
    strategy_name: str
    starting_cash: float
    total_inflows: float
    total_outflows: float
    bond_mtm_loss: float
    equity_loss: float
    repo_penalty: float
    emergency_credit_drawn: float
    emergency_credit_cost: float
    final_cash: float
    final_portfolio_value: float
    reserve_breach: bool
    liquidity_shortfall: float
    strategy_description: str


class StressLabSimulationEngine:
    """
    Simulates multi-asset treasury dynamics and compares 3 benchmark strategies:
    1. All-Cash Benchmark
    2. Greedy Gross-Yield Benchmark
    3. Adaptive Reserve Juggler Strategy
    """

    def run_simulation(
        self,
        base_treasury_state: TreasuryState,
        deposit_shock_pct: float = 0.0,
        rate_shock_bps: float = 0.0,
        repo_delay_days: int = 0,
        credit_unavailable: bool = False,
        equity_drawdown_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executes comparative stress simulation across all 3 strategies with identical inputs.
        """
        reserve_floor = base_treasury_state.protected_reserve_floor
        baseline_outflow = 20_500_000.0  # Mandatory obligations
        shocked_outflow = baseline_outflow * (1.0 + deposit_shock_pct / 100.0)
        inflows = 8_000_000.0

        # Bond yield loss: duration * yield_change
        delta_y = rate_shock_bps / 10000.0

        # --- STRATEGY 1: All-Cash Benchmark ---
        # Portfolio is entirely cash ($220M), zero duration, zero yield optimization
        cash_start_1 = 220_000_000.0
        final_cash_1 = cash_start_1 + inflows - shocked_outflow
        shortfall_1 = max(0.0, reserve_floor - final_cash_1)
        res_1 = StrategyStressOutcome(
            strategy_name="All-Cash Benchmark",
            starting_cash=cash_start_1,
            total_inflows=inflows,
            total_outflows=shocked_outflow,
            bond_mtm_loss=0.0,
            equity_loss=0.0,
            repo_penalty=0.0,
            emergency_credit_drawn=0.0,
            emergency_credit_cost=0.0,
            final_cash=round(final_cash_1, 2),
            final_portfolio_value=round(final_cash_1, 2),
            reserve_breach=(final_cash_1 < reserve_floor),
            liquidity_shortfall=round(shortfall_1, 2),
            strategy_description="Zero term investment; massive cash drag and inflationary erosion.",
        )

        # --- STRATEGY 2: Greedy Gross-Yield Benchmark ---
        # Holds minimal cash ($5M), $100M 10Y bonds, $60M 2Y bonds, $35M equities, $20M locked 30D repo
        cash_start_2 = 5_000_000.0
        bond_loss_2 = (100_000_000.0 * 8.25 * delta_y) + (60_000_000.0 * 1.92 * delta_y)
        eq_loss_2 = 35_000_000.0 * (equity_drawdown_pct / 100.0)
        
        # When outflow hits, greedy strategy runs out of cash immediately
        net_need_2 = shocked_outflow - (cash_start_2 + inflows)
        repo_pen_2 = 0.0
        credit_drawn_2 = 0.0
        credit_cost_2 = 0.0

        if net_need_2 > 0:
            if not credit_unavailable:
                credit_drawn_2 = min(net_need_2, 50_000_000.0)
                credit_cost_2 = credit_drawn_2 * 0.085 * (30 / 365)  # 30 days draw at 8.5%
                net_need_2 -= credit_drawn_2

            # Forced emergency bond liquidation with 8% fire-sale haircut
            if net_need_2 > 0:
                bond_loss_2 += net_need_2 * 0.08

        final_cash_2 = max(0.0, cash_start_2 + inflows + credit_drawn_2 - shocked_outflow)
        shortfall_2 = max(0.0, reserve_floor - final_cash_2)
        portfolio_val_2 = (
            final_cash_2 + 100_000_000.0 + 60_000_000.0 + 35_000_000.0 + 20_000_000.0
            - bond_loss_2 - eq_loss_2 - credit_cost_2 - credit_drawn_2
        )

        res_2 = StrategyStressOutcome(
            strategy_name="Greedy Gross-Yield Benchmark",
            starting_cash=cash_start_2,
            total_inflows=inflows,
            total_outflows=shocked_outflow,
            bond_mtm_loss=round(bond_loss_2, 2),
            equity_loss=round(eq_loss_2, 2),
            repo_penalty=round(repo_pen_2, 2),
            emergency_credit_drawn=round(credit_drawn_2, 2),
            emergency_credit_cost=round(credit_cost_2, 2),
            final_cash=round(final_cash_2, 2),
            final_portfolio_value=round(portfolio_val_2, 2),
            reserve_breach=(final_cash_2 < reserve_floor),
            liquidity_shortfall=round(shortfall_2, 2),
            strategy_description="Over-allocates to long duration; suffers severe fire-sale losses during liquidity squeeze.",
        )

        # --- STRATEGY 3: Adaptive Reserve Juggler ---
        # Uses real base_treasury_state positions: $45M cash, $30M OVN repo, $25M 30D repo, $60M 2Y, $40M 10Y, $20M eq
        cash_start_3 = base_treasury_state.available_cash
        bond_loss_3 = (40_000_000.0 * 8.25 * delta_y) + (60_000_000.0 * 1.92 * delta_y)
        eq_loss_3 = 20_000_000.0 * (equity_drawdown_pct / 100.0)

        # Net cash after operations
        net_cash_3 = cash_start_3 + inflows - shocked_outflow
        repo_unlocked_3 = 0.0
        repo_penalty_3 = 0.0
        credit_drawn_3 = 0.0
        credit_cost_3 = 0.0

        # If cash falls below reserve floor ($25M), adaptively draw Overnight Repo (T+0, no delay)
        if net_cash_3 < reserve_floor:
            deficit_to_floor = reserve_floor - net_cash_3
            # Draw Overnight Tri-Party Repo first ($30M accessible immediately)
            draw_ovn = min(deficit_to_floor, 30_000_000.0)
            net_cash_3 += draw_ovn
            deficit_to_floor -= draw_ovn

            # If still below floor and repo delayed, use bridge credit or 30D repo
            if deficit_to_floor > 0:
                if not credit_unavailable:
                    credit_drawn_3 = min(deficit_to_floor, 50_000_000.0)
                    credit_cost_3 = credit_drawn_3 * 0.055 * (7 / 365)  # 7-day prime bridge
                    net_cash_3 += credit_drawn_3
                    deficit_to_floor -= credit_drawn_3
                elif repo_delay_days > 0:
                    # Early break penalty for 30D repo (150 bps)
                    break_amt = min(deficit_to_floor, 25_000_000.0)
                    repo_penalty_3 = break_amt * 0.015
                    net_cash_3 += (break_amt - repo_penalty_3)
                    deficit_to_floor -= break_amt

        final_cash_3 = net_cash_3
        shortfall_3 = max(0.0, reserve_floor - final_cash_3)
        total_assets_3 = sum(p.market_value for p in base_treasury_state.asset_positions)
        portfolio_val_3 = (
            final_cash_3 + total_assets_3 - bond_loss_3 - eq_loss_3 - credit_cost_3 - credit_drawn_3 - repo_penalty_3
        )

        res_3 = StrategyStressOutcome(
            strategy_name="Adaptive Reserve Juggler",
            starting_cash=cash_start_3,
            total_inflows=inflows,
            total_outflows=shocked_outflow,
            bond_mtm_loss=round(bond_loss_3, 2),
            equity_loss=round(eq_loss_3, 2),
            repo_penalty=round(repo_penalty_3, 2),
            emergency_credit_drawn=round(credit_drawn_3, 2),
            emergency_credit_cost=round(credit_cost_3, 2),
            final_cash=round(final_cash_3, 2),
            final_portfolio_value=round(portfolio_val_3, 2),
            reserve_breach=(final_cash_3 < reserve_floor),
            liquidity_shortfall=round(shortfall_3, 2),
            strategy_description="Dynamically protects statutory reserve floor using Hot/Core capital buffers and minimal-cost financing.",
        )

        return {
            "parameters": {
                "deposit_shock_pct": deposit_shock_pct,
                "rate_shock_bps": rate_shock_bps,
                "repo_delay_days": repo_delay_days,
                "credit_unavailable": credit_unavailable,
                "equity_drawdown_pct": equity_drawdown_pct,
                "reserve_floor": reserve_floor,
                "simulated_outflows": shocked_outflow,
            },
            "strategies": {
                "all_cash": res_1,
                "greedy_yield": res_2,
                "reserve_juggler": res_3,
            },
            "is_shortfall": shortfall_3 > 0,
            "reserve_juggler_shortfall": shortfall_3 > 0,
            "shortfall_amount": shortfall_3,
            "recommended_hedge": "DRAW_OVERNIGHT_REPO" if not credit_unavailable else "LIQUIDATE_SOVEREIGN_2Y",
        }
