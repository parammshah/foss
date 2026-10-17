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
    ending_investments: float = 0.0
    reconciliation_error: float = 0.0
    reserve_deficit: float = 0.0
    customer_obligations_due: float = 0.0
    customer_payouts_paid: float = 0.0
    unpaid_payout_amount: float = 0.0
    is_reconciled: bool = True

    @property
    def ending_cash(self) -> float:
        return self.final_cash

    @property
    def ending_available_cash(self) -> float:
        return self.final_cash

    @property
    def total_portfolio_value(self) -> float:
        return self.final_portfolio_value


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
        strategy_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes comparative stress simulation across all 3 strategies with identical inputs.
        Requires explicit protected_reserve_floor on base_treasury_state.
        """
        if base_treasury_state.protected_reserve_floor is None:
            raise ValueError(
                "Required protected_reserve_floor is missing from base_treasury_state; "
                "scenario execution requires an explicit reserve floor policy."
            )
        reserve_floor = float(base_treasury_state.protected_reserve_floor)
        baseline_outflow = 20_500_000.0  # Mandatory obligations
        shocked_outflow = baseline_outflow * (1.0 + deposit_shock_pct / 100.0)
        inflows = 8_000_000.0

        # Bond yield loss: duration * yield_change
        delta_y = rate_shock_bps / 10000.0

        # Common starting total asset base across all 3 strategies ($220M)
        total_starting_assets = 220_000_000.0

        # --- STRATEGY 1: All-Cash Benchmark ---
        # Portfolio is entirely operational cash ($220M), zero duration, zero yield optimization
        cash_start_1 = total_starting_assets
        payouts_paid_1 = min(shocked_outflow, cash_start_1 + inflows)
        unpaid_payouts_1 = max(0.0, shocked_outflow - payouts_paid_1)
        final_cash_1 = max(0.0, cash_start_1 + inflows - shocked_outflow)
        deficit_1 = max(0.0, reserve_floor - final_cash_1)
        shortfall_1 = deficit_1
        starting_nav_1 = cash_start_1
        expected_ending_nav_1 = starting_nav_1 + (inflows - shocked_outflow)
        ending_inv_1 = - unpaid_payouts_1
        portfolio_val_1 = final_cash_1 + ending_inv_1
        reconcil_err_1 = round(abs(expected_ending_nav_1 - portfolio_val_1), 2)
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
            final_portfolio_value=round(portfolio_val_1, 2),
            reserve_breach=(final_cash_1 < reserve_floor),
            liquidity_shortfall=round(shortfall_1, 2),
            strategy_description="Zero term investment; massive cash drag and inflationary erosion.",
            ending_investments=round(ending_inv_1, 2),
            reconciliation_error=reconcil_err_1,
            reserve_deficit=round(deficit_1, 2),
            customer_obligations_due=round(shocked_outflow, 2),
            customer_payouts_paid=round(payouts_paid_1, 2),
            unpaid_payout_amount=round(unpaid_payouts_1, 2),
            is_reconciled=(reconcil_err_1 < 1.0),
        )
        # --- STRATEGY 2: Greedy Gross-Yield Benchmark ---
        # Starts with: $5M cash, $100M 10Y bonds, $60M 2Y bonds, $35M equities, $20M locked 30D repo (Total = $220M)
        cash_start_2 = 5_000_000.0
        rem_bonds_10y_2 = 100_000_000.0
        rem_bonds_2y_2 = 60_000_000.0
        rem_eq_2 = 35_000_000.0
        rem_repo_30d_2 = 20_000_000.0

        net_need_2 = shocked_outflow - (cash_start_2 + inflows)
        credit_drawn_2 = 0.0
        credit_cost_2 = 0.0
        realized_haircut_2 = 0.0

        bonds_sold_2 = 0.0
        if net_need_2 > 0:
            # Standby credit line if available
            if not credit_unavailable:
                credit_drawn_2 = min(net_need_2, 50_000_000.0)
                credit_cost_2 = credit_drawn_2 * 0.085 * (30 / 365)  # 30 days draw at 8.5%
                net_need_2 -= credit_drawn_2

            # Secondary market bond liquidation has T+1 settlement latency and 8% fire-sale haircut
            if net_need_2 > 0:
                bonds_sold_2 = min(net_need_2, rem_bonds_2y_2)
                bond_loss_fire = bonds_sold_2 * 0.08
                realized_haircut_2 += bond_loss_fire
                rem_bonds_2y_2 = max(0.0, rem_bonds_2y_2 - bonds_sold_2)

        # Intraday operational cash: bond liquidation proceeds cannot settle T+0 intraday
        liquid_cash_2 = cash_start_2 + inflows + credit_drawn_2
        payouts_paid_2 = min(shocked_outflow, liquid_cash_2)
        unpaid_payouts_2 = max(0.0, shocked_outflow - liquid_cash_2)
        final_cash_2 = max(0.0, liquid_cash_2 - shocked_outflow)
        deficit_2 = max(0.0, reserve_floor - final_cash_2)
        shortfall_2 = deficit_2
        reserve_breach_2 = (final_cash_2 < reserve_floor)

        t1_bond_receivable_2 = bonds_sold_2 * (1.0 - 0.08)

        bond_mtm_loss_2 = (rem_bonds_10y_2 * 8.25 * delta_y) + (rem_bonds_2y_2 * 1.92 * delta_y) + realized_haircut_2
        eq_loss_2 = rem_eq_2 * (equity_drawdown_pct / 100.0)

        # Consistent ledger balance sheet: ending investments including T+1 receivable minus credit debt and unpaid obligations
        ending_inv_2 = (
            rem_repo_30d_2 + (rem_bonds_10y_2 * (1.0 - 8.25 * delta_y))
            + (rem_bonds_2y_2 * (1.0 - 1.92 * delta_y)) + (rem_eq_2 * (1.0 - equity_drawdown_pct / 100.0))
            + t1_bond_receivable_2 - credit_drawn_2 - credit_cost_2 - unpaid_payouts_2
        )
        portfolio_val_2 = final_cash_2 + ending_inv_2

        # Independent income-statement PnL reconciliation
        starting_nav_2 = cash_start_2 + 100_000_000.0 + 60_000_000.0 + 35_000_000.0 + 20_000_000.0
        expected_ending_nav_2 = starting_nav_2 + (inflows - shocked_outflow) - bond_mtm_loss_2 - eq_loss_2 - credit_cost_2
        reconcil_err_2 = round(abs(expected_ending_nav_2 - portfolio_val_2), 2)

        res_2 = StrategyStressOutcome(
            strategy_name="Greedy Gross-Yield Benchmark",
            starting_cash=cash_start_2,
            total_inflows=inflows,
            total_outflows=shocked_outflow,
            bond_mtm_loss=round(bond_mtm_loss_2, 2),
            equity_loss=round(eq_loss_2, 2),
            repo_penalty=0.0,
            emergency_credit_drawn=round(credit_drawn_2, 2),
            emergency_credit_cost=round(credit_cost_2, 2),
            final_cash=round(final_cash_2, 2),
            final_portfolio_value=round(portfolio_val_2, 2),
            reserve_breach=reserve_breach_2,
            liquidity_shortfall=round(shortfall_2, 2),
            strategy_description="Over-allocates to long duration; suffers severe fire-sale losses during liquidity squeeze.",
            ending_investments=round(ending_inv_2, 2),
            reconciliation_error=reconcil_err_2,
            reserve_deficit=round(deficit_2, 2),
            customer_obligations_due=round(shocked_outflow, 2),
            customer_payouts_paid=round(payouts_paid_2, 2),
            unpaid_payout_amount=round(unpaid_payouts_2, 2),
            is_reconciled=(reconcil_err_2 < 1.0),
        )

        # --- STRATEGY 3: Adaptive Reserve Juggler ---
        # Starts with: $45M cash, $30M OVN repo, $25M 30D repo, $60M 2Y, $40M 10Y, $20M eq (Total = $220M)
        buf_mult = 1.0
        if strategy_overrides and "liquidity_buffer_multiplier" in strategy_overrides:
            buf_mult = float(strategy_overrides["liquidity_buffer_multiplier"])

        # When candidate changes liquidity buffer multiplier, cash and buffer allocations shift
        # relative to baseline buffer ratio (1.25 = $45M cash baseline)
        mult_factor = (buf_mult / 1.25) if buf_mult != 1.0 else 1.0
        cash_start_3 = base_treasury_state.available_cash * mult_factor
        rem_repo_ovn_3 = 30_000_000.0 * mult_factor
        rem_repo_30d_3 = 25_000_000.0 * mult_factor

        # Conserve starting total assets ($220M) by adjusting 2Y sovereign bond allocation
        buffer_shift = (base_treasury_state.available_cash + 30_000_000.0 + 25_000_000.0) - (cash_start_3 + rem_repo_ovn_3 + rem_repo_30d_3)
        rem_bonds_2y_3 = 60_000_000.0 + buffer_shift
        rem_bonds_10y_3 = 40_000_000.0
        rem_eq_3 = 20_000_000.0

        starting_nav_3 = cash_start_3 + rem_repo_ovn_3 + rem_repo_30d_3 + rem_bonds_2y_3 + rem_bonds_10y_3 + rem_eq_3

        net_cash_3 = cash_start_3 + inflows - shocked_outflow
        repo_penalty_3 = 0.0
        credit_drawn_3 = 0.0
        credit_cost_3 = 0.0

        # If cash falls below reserve floor, dynamically activate tiered liquidity buffer
        if net_cash_3 < reserve_floor:
            deficit_to_floor = reserve_floor - net_cash_3

            # Tier 1: Overnight Tri-Party Repo ($30M). Only accessible if repo_delay_days == 0!
            # If repo_delay_days > 0, repo settlement is delayed and funds cannot be drawn intraday!
            if repo_delay_days == 0 and rem_repo_ovn_3 > 0:
                draw_ovn = min(deficit_to_floor, rem_repo_ovn_3)
                net_cash_3 += draw_ovn
                rem_repo_ovn_3 -= draw_ovn  # Deduct principal: no double-counting!
                deficit_to_floor -= draw_ovn

            # Tier 2: Prime Bridge Credit Facility (if credit is available)
            if deficit_to_floor > 0 and not credit_unavailable:
                credit_drawn_3 = min(deficit_to_floor, 50_000_000.0)
                credit_cost_3 = credit_drawn_3 * 0.055 * (7 / 365)  # 7-day prime bridge at 5.5%
                net_cash_3 += credit_drawn_3
                deficit_to_floor -= credit_drawn_3

            # Tier 3: Break 30D Term Repo with 150 bps early exit charge (requires early termination delay: repo_delay_days > 0)
            if deficit_to_floor > 0 and repo_delay_days > 0 and rem_repo_30d_3 > 0:
                avail_draw = rem_repo_30d_3 * buf_mult if buf_mult < 1.0 else rem_repo_30d_3
                break_amt = min(deficit_to_floor / 0.985, avail_draw)
                pen = break_amt * 0.015
                repo_penalty_3 += pen
                net_cash_3 += (break_amt - pen)
                rem_repo_30d_3 -= break_amt  # Deduct principal: no double-counting!
                deficit_to_floor -= (break_amt - pen)

        final_cash_3 = max(0.0, net_cash_3)
        unpaid_payouts_3 = max(0.0, -net_cash_3)
        deficit_3 = max(0.0, reserve_floor - final_cash_3)
        shortfall_3 = deficit_3
        liquid_cash_3 = cash_start_3 + inflows + credit_drawn_3 + (30_000_000.0 - rem_repo_ovn_3) + (25_000_000.0 - rem_repo_30d_3 - repo_penalty_3)
        payouts_paid_3 = min(shocked_outflow, liquid_cash_3)
        unpaid_payouts_3 = max(0.0, shocked_outflow - liquid_cash_3)

        bond_mtm_loss_3 = (rem_bonds_10y_3 * 8.25 * delta_y) + (rem_bonds_2y_3 * 1.92 * delta_y)
        eq_loss_3 = rem_eq_3 * (equity_drawdown_pct / 100.0)

        # Consistent ledger balance sheet: ending investments minus credit debt and unpaid obligations
        ending_inv_3 = (
            rem_repo_ovn_3 + rem_repo_30d_3
            + (rem_bonds_10y_3 * (1.0 - 8.25 * delta_y))
            + (rem_bonds_2y_3 * (1.0 - 1.92 * delta_y))
            + (rem_eq_3 * (1.0 - equity_drawdown_pct / 100.0))
            - credit_drawn_3 - credit_cost_3 - unpaid_payouts_3
        )
        portfolio_val_3 = final_cash_3 + ending_inv_3

        # Independent income-statement PnL reconciliation
        expected_ending_nav_3 = starting_nav_3 + (inflows - shocked_outflow) - bond_mtm_loss_3 - eq_loss_3 - credit_cost_3 - repo_penalty_3
        reconcil_err_3 = round(abs(expected_ending_nav_3 - portfolio_val_3), 2)

        res_3 = StrategyStressOutcome(
            strategy_name="Adaptive Reserve Juggler",
            starting_cash=cash_start_3,
            total_inflows=inflows,
            total_outflows=shocked_outflow,
            bond_mtm_loss=round(bond_mtm_loss_3, 2),
            equity_loss=round(eq_loss_3, 2),
            repo_penalty=round(repo_penalty_3, 2),
            emergency_credit_drawn=round(credit_drawn_3, 2),
            emergency_credit_cost=round(credit_cost_3, 2),
            final_cash=round(final_cash_3, 2),
            final_portfolio_value=round(portfolio_val_3, 2),
            reserve_breach=(final_cash_3 < reserve_floor),
            liquidity_shortfall=round(shortfall_3, 2),
            strategy_description="Dynamically protects statutory reserve floor using Hot/Core capital buffers and minimal-cost financing.",
            ending_investments=round(ending_inv_3, 2),
            reconciliation_error=reconcil_err_3,
            reserve_deficit=round(deficit_3, 2),
            customer_obligations_due=round(shocked_outflow, 2),
            customer_payouts_paid=round(payouts_paid_3, 2),
            unpaid_payout_amount=round(unpaid_payouts_3, 2),
            is_reconciled=(reconcil_err_3 < 1.0),
        )

        all_reconciled = res_1.is_reconciled and res_2.is_reconciled and res_3.is_reconciled

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
            "all_strategies_reconciled": all_reconciled,
            "recommended_hedge": "NO_RECOMMENDATION_OPTIMIZER_UNINTEGRATED",
        }
