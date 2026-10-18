"""
Reproducible 8,760-hour Treasury Simulation and Performance Audit on treasury_4tier_dataset.csv.
Compares:
1. Naive Strategy (Cash-Only Buffer)
2. Aggressive Yield Chaser (Unconstrained Sweeper, ignores holding gates/reserves)
3. Repaired Person 2 Engine (TSS, Capital Firewall, Liquidity Shield, Dual-Mode Optimizer)

Maintains complete double-entry accounting ledgers:
- Operational cash
- Repo holdings and normal maturities
- Sovereign bond inventory, accrued coupon yield, haircuts
- Growth equities inventory, dividend yield, haircuts
- Bridge credit principal, hourly interest accrual, repayments
- Cumulative unpaid customer obligations (balance-sheet liabilities)
- Comprehensive shortfall audit log with pre/post resource tracking
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# Ensure src and workspace root are in Python path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.optimization.models import ActionType
from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteForecastBundle,
    ConcreteForecastPoint,
    ConcreteObligation,
    ConcreteTreasuryState,
)


def get_prior_forecast_bundle(
    history_deposits: list[float],
    history_withdrawals: list[float],
    ts: datetime,
) -> ConcreteForecastBundle:
    """Generate forecast bundle using strictly prior observed information (no look-ahead).

    Forecast points represent cumulative expected cash flows through each horizon h (hours).
    Base flow over horizon h is: mean_hourly_flow * float(h).
    """
    if len(history_deposits) >= 24:
        d_mean = float(np.mean(history_deposits[-24:]))
        w_mean = float(np.mean(history_withdrawals[-24:]))
    elif history_deposits:
        d_mean = float(np.mean(history_deposits))
        w_mean = float(np.mean(history_withdrawals))
    else:
        d_mean = 10_000_000.0
        w_mean = 10_000_000.0

    f_pts = []
    for h in [1, 2, 4, 8, 12, 24, 48, 72]:
        d_base = d_mean * float(h)
        w_base = w_mean * float(h)
        f_pts.append(
            ConcreteForecastPoint(
                forecast_horizon=float(h),
                deposit_p10=max(0.0, d_base * 0.75),
                deposit_p50=d_base,
                deposit_p90=d_base * 1.35,
                withdrawal_p10=max(0.0, w_base * 0.75),
                withdrawal_p50=w_base,
                withdrawal_p90=w_base * 1.35,
                reliability_metadata={"r_squared": 0.94, "forecast_age_hours": 0.25},
            )
        )
    return ConcreteForecastBundle(point_forecasts=f_pts, timestamp=ts, quality_status="HEALTHY")


def simulate_all_strategies(dataset_path: str, audit_file_path: Optional[str] = None):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at '{dataset_path}'", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    total_hours = len(df)
    print(f"Simulating {total_hours:,} hours from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}...")

    # Shared initial conditions across all 3 benchmark strategies
    INITIAL_CASH = 30_000_000.0
    CREDIT_LIMIT = 15_000_000.0

    # -------------------------------------------------------------------------
    # 1. NAIVE CASH-ONLY STRATEGY
    # -------------------------------------------------------------------------
    cash_naive = INITIAL_CASH
    shortfalls_naive = 0.0
    shortfall_events_naive = 0
    payouts_funded_naive = 0.0
    total_payouts = 0.0
    cumulative_unpaid_naive = 0.0

    for idx, row in df.iterrows():
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        total_payouts += wth
        cash_naive += dep

        if cash_naive >= wth:
            cash_naive -= wth
            payouts_funded_naive += wth
        else:
            funded = max(0.0, cash_naive)
            gap = wth - funded
            payouts_funded_naive += funded
            shortfalls_naive += gap
            shortfall_events_naive += 1
            cumulative_unpaid_naive += gap
            cash_naive = 0.0

    # -------------------------------------------------------------------------
    # 2. AGGRESSIVE YIELD CHASER (Zero reserve, sweeps all surplus into repo)
    # -------------------------------------------------------------------------
    cash_agg = INITIAL_CASH
    repo_agg: list[tuple[float, datetime, float]] = []
    shortfalls_agg = 0.0
    shortfall_events_agg = 0
    penalties_agg = 0.0
    overdraft_interest_agg = 0.0
    yield_earned_agg = 0.0
    payouts_funded_agg = 0.0
    cumulative_unpaid_agg = 0.0

    for idx, row in df.iterrows():
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        repo_rate = float(row["tier1_repo_rate_apy"]) / 100.0
        is_open = bool(row["repo_sweep_window_open"])
        haircut = float(row["tier1_early_redemption_haircut_pct"]) / 100.0
        overdraft_apr = float(row["emergency_overdraft_penalty_apr_pct"]) / 100.0

        cash_agg += dep

        # Mature repos
        still_active = []
        for r_amt, r_mat, r_rate in repo_agg:
            if r_mat <= ts:
                interest = r_amt * (r_rate * (15.0 / 8760.0))
                cash_agg += r_amt + interest
                yield_earned_agg += interest
            else:
                still_active.append((r_amt, r_mat, r_rate))
        repo_agg = still_active

        # Fund withdrawals
        if cash_agg < wth:
            needed = wth - cash_agg
            # Break repo early with penalty
            for r_amt, r_mat, r_rate in list(repo_agg):
                if needed <= 0:
                    break
                pen = r_amt * haircut
                net_proceeds = r_amt - pen
                penalties_agg += pen
                cash_agg += net_proceeds
                repo_agg.remove((r_amt, r_mat, r_rate))
                needed -= net_proceeds

            if cash_agg < wth:
                # Emergency overdraft
                od = wth - cash_agg
                overdraft_interest_agg += od * (overdraft_apr * (1.0 / 8760.0))
                payouts_funded_agg += cash_agg
                shortfalls_agg += od
                shortfall_events_agg += 1
                cumulative_unpaid_agg += od
                cash_agg = 0.0
            else:
                cash_agg -= wth
                payouts_funded_agg += wth
        else:
            cash_agg -= wth
            payouts_funded_agg += wth

        # Aggressively sweep cash if window is open
        if is_open and cash_agg > 100_000.0:
            sweep = cash_agg - 50_000.0
            cash_agg -= sweep
            repo_agg.append((sweep, ts + timedelta(hours=15), repo_rate))

    # -------------------------------------------------------------------------
    # 3. REPAIRED PERSON 2 DECISION ENGINE
    # -------------------------------------------------------------------------
    service = TreasuryDecisionService(policy_version="2.1.0-repaired", model_version="1.5.0")
    tax_profile = create_statutory_us_corp_profile()

    from reserve_juggler.optimization.models import HumanOverride

    cash_p2 = INITIAL_CASH
    active_repos_p2: list[ConcreteAssetPosition] = []
    active_bonds_p2: list[ConcreteAssetPosition] = []
    active_equities_p2: list[ConcreteAssetPosition] = []
    pending_settlements_p2: list[dict[str, Any]] = []  # Settling asset sale proceeds: {"settle_ts": datetime, "amount": float}
    active_bridge_loan = 0.0
    active_bridge_interest_accrued = 0.0
    bridge_due_ts: Optional[datetime] = None

    # Four distinct shortfall / liquidity measures:
    # 1. Pre-action liquidity deficit detected
    pre_action_deficits_p2 = 0
    pre_action_deficit_amount_p2 = 0.0

    # 2. Mandatory payment deadline missed
    deadline_miss_incidents_p2 = 0
    shortfalls_at_deadline_p2 = 0.0

    # 3. Funding plan unresolved after permitted actions
    unresolved_funding_actions_p2 = 0

    # 4. Overdue customer obligation queue (FIFO)
    overdue_queue_p2: list[dict[str, Any]] = []

    payouts_funded_on_time_p2 = 0.0
    payouts_funded_late_p2 = 0.0
    yield_earned_p2 = 0.0
    bridge_interest_p2 = 0.0
    penalties_p2 = 0.0
    liquidation_losses_p2 = 0.0
    mtm_unrealized_p2 = 0.0
    action_counts_p2: dict[str, int] = {}
    tss_history_p2: list[float] = []

    history_deposits: list[float] = []
    history_withdrawals: list[float] = []

    cum_deposits_p2 = 0.0
    cum_withdrawals_due_p2 = 0.0

    # Machine-readable shortfall event audit log
    shortfall_audit_records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        repo_rate = float(row["tier1_repo_rate_apy"]) / 100.0
        bond_yield = float(row["tier2_bond_yield_pct"]) / 100.0
        bond_price_idx = float(row["tier2_bond_price_index"])
        stock_price_idx = float(row["tier3_stock_index_price"])
        is_open = bool(row["repo_sweep_window_open"])
        overdraft_apr = float(row["emergency_overdraft_penalty_apr_pct"]) / 100.0

        cum_deposits_p2 += dep
        cum_withdrawals_due_p2 += wth

        # Snapshot opening cash before any arrivals, settlements or maturities
        cash_opening_hour = cash_p2
        credit_free_pre = max(0.0, CREDIT_LIMIT - (active_bridge_loan + active_bridge_interest_accrued))

        # 0. Settle any maturing pending settlements (T+24h / T+48h)
        still_pending = []
        settled_sales_cash = 0.0
        for ps in pending_settlements_p2:
            if ps["settle_ts"] <= ts:
                cash_p2 += ps["amount"]
                settled_sales_cash += ps["amount"]
            else:
                still_pending.append(ps)
        pending_settlements_p2 = still_pending

        # 1. Normal repo maturities (08:00 release without early exit fees)
        still_repos = []
        repo_cash_settled = 0.0
        for rp in active_repos_p2:
            if rp.maturity and rp.maturity <= ts:
                earned = rp.market_value * (rp.annual_yield * (15.0 / 8760.0))
                cash_p2 += rp.market_value + earned
                repo_cash_settled += rp.market_value + earned
                yield_earned_p2 += earned
            else:
                still_repos.append(rp)
        active_repos_p2 = still_repos

        # Mark-to-market bond and equity holdings via price indices (no cash injection from coupons)
        # Accrued bond coupon increases bond valuation and equity, but does NOT enter cash!
        current_bond_mv = 0.0
        for bp in active_bonds_p2:
            if bp.units and bp.units > 0:
                bp.market_value = bp.units * bond_price_idx
            bp.liquidatable_value = bp.market_value * (1.0 - bp.haircut)
            current_bond_mv += bp.market_value

        current_equity_mv = 0.0
        for ep in active_equities_p2:
            if ep.units and ep.units > 0:
                ep.market_value = ep.units * stock_price_idx
            ep.liquidatable_value = ep.market_value * (1.0 - ep.haircut)
            current_equity_mv += ep.market_value

        repo_mv_pre = sum(float(rp.market_value) for rp in active_repos_p2)
        bond_mv_pre = current_bond_mv
        equity_mv_pre = current_equity_mv

        # 2. Hourly customer deposits arrive
        cash_p2 += dep

        # 3. Repay bridge loan if due (senior obligation: principal + accrued financing interest)
        if active_bridge_loan > 0.0 and bridge_due_ts is not None and ts >= bridge_due_ts:
            total_bridge_due = active_bridge_loan + active_bridge_interest_accrued
            if cash_p2 >= total_bridge_due:
                cash_p2 -= total_bridge_due
                active_bridge_loan = 0.0
                active_bridge_interest_accrued = 0.0
                bridge_due_ts = None
            else:
                if cash_p2 >= active_bridge_interest_accrued:
                    cash_p2 -= active_bridge_interest_accrued
                    active_bridge_interest_accrued = 0.0
                    active_bridge_loan -= cash_p2
                    cash_p2 = 0.0
                else:
                    active_bridge_interest_accrued -= cash_p2
                    cash_p2 = 0.0
                add_int = active_bridge_loan * (overdraft_apr * (1.0 / 8760.0))
                bridge_interest_p2 += add_int
                active_bridge_interest_accrued += add_int

        # Pre-action liquidity check: does cash cover the payout due at this deadline?
        cash_before_decision = cash_p2
        is_pre_action_deficit = cash_before_decision < wth
        if is_pre_action_deficit:
            pre_action_deficits_p2 += 1
            pre_action_deficit_amount_p2 += (wth - cash_before_decision)

        # 4. Construct prior forecast bundle without future look-ahead
        bundle = get_prior_forecast_bundle(history_deposits, history_withdrawals, ts)
        history_deposits.append(dep)
        history_withdrawals.append(wth)

        # 5. Build state and request decision
        all_positions = active_repos_p2 + active_bonds_p2 + active_equities_p2
        ob = ConcreteObligation(
            due_time=ts,  # Due at current timestamp
            amount=wth,
            is_mandatory=True,
            description="Hourly Customer Payout",
        )

        credit_free = max(0.0, CREDIT_LIMIT - (active_bridge_loan + active_bridge_interest_accrued))
        state = ConcreteTreasuryState(
            timestamp=ts,
            available_cash=cash_p2,
            asset_positions=all_positions,
            future_obligations=[ob],
            credit_availability=credit_free,
            credit_limit=CREDIT_LIMIT,
            credit_apr=overdraft_apr,
        )

        # Check for desk override if emergency bridge credit is needed
        human_override = None
        if is_pre_action_deficit and credit_free > 0.0:
            human_override = HumanOverride(
                candidate_id="DRAW_BRIDGE_CREDIT",
                authorized_by="TREASURY_EMERGENCY_DESK",
                override_reason="Mandatory customer payout protection",
                timestamp=ts,
            )

        record = service.decide(
            state=state,
            forecast_bundle=bundle,
            tax_profile=tax_profile,
            repo_overnight_rate_apy=repo_rate,
            repo_sweep_lockup_hours=15.0,
            repo_sweep_window_open=is_open,
            human_override=human_override,
        )

        sel = record.selected_action
        act_type = sel["action_type"]
        cand_id = sel.get("candidate_id", act_type)
        action_counts_p2[act_type] = action_counts_p2.get(act_type, 0) + 1
        tss_history_p2.append(record.score["composite_score"])

        executed_funding = 0.0
        credit_drawn_now = 0.0

        # Track unresolved funding plans
        if act_type in [
            ActionType.UNRESOLVED_FUNDING_SHORTFALL.value,
            ActionType.KEEP_CASH.value,
        ] and is_pre_action_deficit:
            unresolved_funding_actions_p2 += 1

        # 6. Execution Adapters for all actions
        if act_type == ActionType.DRAW_BRIDGE_CREDIT.value:
            borrowed = float(sel.get("principal_borrowed", 0.0) or sel["amount"])
            if borrowed + cash_p2 < wth and (wth - (borrowed + cash_p2)) < 0.01:
                borrowed = wth - cash_p2
            int_cost = float(sel.get("interest_cost", 0.0))
            active_bridge_loan += borrowed
            active_bridge_interest_accrued += int_cost
            bridge_due_ts = ts + timedelta(hours=2)
            cash_p2 += borrowed
            bridge_interest_p2 += int_cost
            executed_funding = borrowed
            credit_drawn_now = borrowed

        elif act_type == ActionType.BREAK_REPO_EARLY.value:
            target_id = sel.get("asset_id")
            break_amt = float(sel.get("gross_proceeds", 0.0) or sel["amount"])
            matched = False
            for rp in list(active_repos_p2):
                if target_id is None or rp.asset_id == target_id:
                    matched = True
                    pen_rate = float(rp.exit_fees) if rp.exit_fees is not None else 0.005
                    actual_break = min(rp.market_value, break_amt)
                    pen = actual_break * pen_rate
                    net_p = actual_break - pen
                    cash_p2 += net_p
                    penalties_p2 += pen
                    executed_funding = net_p
                    if actual_break >= rp.market_value - 1e-4:
                        active_repos_p2.remove(rp)
                    else:
                        rp.market_value -= actual_break
                        rp.liquidatable_value = rp.market_value * (1.0 - pen_rate)
                    break
            if not matched and active_repos_p2:
                rp = active_repos_p2[0]
                pen_rate = float(rp.exit_fees) if rp.exit_fees is not None else 0.005
                actual_break = min(rp.market_value, break_amt)
                pen = actual_break * pen_rate
                net_p = actual_break - pen
                cash_p2 += net_p
                penalties_p2 += pen
                executed_funding = net_p
                if actual_break >= rp.market_value - 1e-4:
                    active_repos_p2.remove(rp)
                else:
                    rp.market_value -= actual_break

        elif act_type == ActionType.PLACE_REPO.value:
            sweep_amt = float(sel["amount"])
            if cash_p2 >= sweep_amt and is_open:
                cash_p2 -= sweep_amt
                active_repos_p2.append(
                    ConcreteAssetPosition(
                        asset_id=f"REPO_{idx}",
                        tier=1,
                        market_value=sweep_amt,
                        liquidatable_value=sweep_amt * 0.995,
                        availability=ts + timedelta(hours=15),
                        maturity=ts + timedelta(hours=15),
                        annual_yield=repo_rate,
                        exit_fees=0.005,
                        instrument_type="repo",
                    )
                )

        elif act_type == ActionType.BUY_BOND.value:
            b_amt = float(sel["amount"])
            if cash_p2 >= b_amt:
                cash_p2 -= b_amt
                units = b_amt / bond_price_idx if bond_price_idx > 0 else 1.0
                active_bonds_p2.append(
                    ConcreteAssetPosition(
                        asset_id=f"BOND_{idx}",
                        tier=2,
                        market_value=b_amt,
                        liquidatable_value=b_amt * 0.98,
                        availability=ts + timedelta(hours=24),
                        maturity=ts + timedelta(days=730),
                        settlement_delay_hours=24.0,
                        annual_yield=bond_yield,
                        exit_fees=0.0,
                        haircut=0.02,
                        duration=2.0,
                        instrument_type="bond",
                        units=units,
                        entry_price=bond_price_idx,
                        purchase_timestamp=ts,
                    )
                )

        elif act_type == ActionType.BUY_EQUITY.value:
            eq_amt = float(sel["amount"])
            if cash_p2 >= eq_amt:
                cash_p2 -= eq_amt
                units = eq_amt / stock_price_idx if stock_price_idx > 0 else 1.0
                active_equities_p2.append(
                    ConcreteAssetPosition(
                        asset_id=f"EQ_{idx}",
                        tier=3,
                        market_value=eq_amt,
                        liquidatable_value=eq_amt * 0.85,
                        availability=ts + timedelta(hours=48),
                        maturity=ts + timedelta(days=3650),
                        settlement_delay_hours=48.0,
                        annual_yield=0.08,
                        exit_fees=0.0,
                        haircut=0.15,
                        volatility=0.18,
                        instrument_type="equity",
                        units=units,
                        entry_price=stock_price_idx,
                        purchase_timestamp=ts,
                    )
                )

        elif act_type == ActionType.SELL_BOND.value:
            target_id = sel.get("asset_id")
            for bp in list(active_bonds_p2):
                if target_id is None or bp.asset_id == target_id:
                    loss = bp.market_value * bp.haircut
                    net_p = bp.market_value - loss
                    # Settle with 24-hour delay
                    pending_settlements_p2.append({"settle_ts": ts + timedelta(hours=24), "amount": net_p})
                    liquidation_losses_p2 += loss
                    active_bonds_p2.remove(bp)
                    break

        elif act_type == ActionType.SELL_EQUITY.value:
            target_id = sel.get("asset_id")
            for ep in list(active_equities_p2):
                if target_id is None or ep.asset_id == target_id:
                    loss = ep.market_value * ep.haircut
                    net_p = ep.market_value - loss
                    # Settle with 48-hour delay
                    pending_settlements_p2.append({"settle_ts": ts + timedelta(hours=48), "amount": net_p})
                    liquidation_losses_p2 += loss
                    active_equities_p2.remove(ep)
                    break

        # 7. Settle customer withdrawal at payment deadline
        cash_available_before_payout = cash_p2
        paid_late_this_hour = 0.0

        if cash_available_before_payout >= wth:
            cash_p2 -= wth
            payouts_funded_on_time_p2 += wth
            paid_now = wth
            shortfall_now = 0.0
        else:
            paid_now = max(0.0, cash_available_before_payout)
            payouts_funded_on_time_p2 += paid_now
            shortfall_now = wth - paid_now
            shortfalls_at_deadline_p2 += shortfall_now
            deadline_miss_incidents_p2 += 1
            # Add to overdue customer liabilities queue
            overdue_queue_p2.append(
                {
                    "obligation_id": f"OBLIG_{idx}",
                    "amount": shortfall_now,
                    "due_timestamp": ts.isoformat(),
                    "hour_index": idx,
                }
            )
            cash_p2 = 0.0

        current_overdue_liability = sum(item["amount"] for item in overdue_queue_p2)

        # Record machine-readable shortfall audit entry if pre-action deficit occurred
        if is_pre_action_deficit:
            all_alternatives = record.alternatives  # Rejected alternatives (sel excluded)
            considered_plans = [sel.get("candidate_id", sel["action_type"])] + [
                c.get("candidate_id", c["action_type"]) for c in all_alternatives
            ]
            rejection_reasons = {
                c.get("candidate_id", c["action_type"]): c.get("reason_code", "REJECTED")
                for c in all_alternatives
                if not c.get("is_feasible", False)
            }

            shortfall_audit_records.append(
                {
                    "hour_index": int(idx),
                    "timestamp": ts.isoformat(),
                    # 12 Authoritative Snapshot Fields:
                    "cash_before_event": round(cash_opening_hour, 2),
                    "deposits_settled": round(dep, 2),
                    "investment_cash_income_settled": round(settled_sales_cash + repo_cash_settled, 2),
                    "bridge_draw_amount": round(credit_drawn_now, 2),
                    "cash_before_payout": round(cash_available_before_payout, 2),
                    "payout_amount": round(wth, 2),
                    "cash_after_payout": round(cash_p2, 2),
                    "bridge_principal_outstanding": round(active_bridge_loan, 2),
                    "bridge_interest_outstanding": round(active_bridge_interest_accrued, 2),
                    "payout_funded": round(paid_now, 2),
                    "funding_plan_resolved": not (
                        act_type in [
                            ActionType.UNRESOLVED_FUNDING_SHORTFALL.value,
                            ActionType.KEEP_CASH.value,
                        ] and shortfall_now > 0.01
                    ),
                    "repayment_plan_feasible": sel.get("repayment_plan_feasible", True),
                    # Supporting context & audit verification:
                    "selected_action": sel["action_type"],
                    "candidate_id": cand_id,
                    "human_override": human_override.to_dict() if human_override else None,
                    "pre_action_deficit_detected": True,
                    "mandatory_deadline_missed": shortfall_now > 0.01,
                    "funding_plan_unresolved": act_type in [
                        ActionType.UNRESOLVED_FUNDING_SHORTFALL.value,
                        ActionType.KEEP_CASH.value,
                    ] and shortfall_now > 0.01,
                    "residual_shortfall": round(shortfall_now, 2),
                    "amount_paid_on_time": round(paid_now, 2),
                    "amount_paid_late": round(paid_late_this_hour, 2),
                    "remaining_unpaid_amount": round(current_overdue_liability, 2),
                    "candidate_funding_plans_considered": considered_plans,
                    "candidate_rejection_reasons": rejection_reasons,
                    "tss_score": round(record.score["composite_score"], 2),
                    "resources_before_event": {
                        "cash": round(cash_opening_hour, 2),
                        "repo": round(repo_mv_pre, 2),
                        "bonds": round(bond_mv_pre, 2),
                        "equities": round(equity_mv_pre, 2),
                        "credit_available": round(credit_free_pre, 2),
                    },
                    "resources_after_event": {
                        "cash": round(cash_p2, 2),
                        "repo": round(sum(float(rp.market_value) for rp in active_repos_p2), 2),
                        "bonds": round(sum(float(bp.market_value) for bp in active_bonds_p2), 2),
                        "equities": round(sum(float(ep.market_value) for ep in active_equities_p2), 2),
                        "pending_settlements": round(sum(ps["amount"] for ps in pending_settlements_p2), 2),
                        "credit_available": round(max(0.0, CREDIT_LIMIT - (active_bridge_loan + active_bridge_interest_accrued)), 2),
                        "outstanding_debt": round(active_bridge_loan + active_bridge_interest_accrued, 2),
                    },
                }
            )

        # 8. Assert Double-Entry Timestep Balance Sheet Reconciliation
        invested_assets_now = (
            sum(float(rp.market_value) for rp in active_repos_p2)
            + sum(float(bp.market_value) for bp in active_bonds_p2)
            + sum(float(ep.market_value) for ep in active_equities_p2)
            + sum(ps["amount"] for ps in pending_settlements_p2)
        )
        total_assets_hour = cash_p2 + invested_assets_now
        total_liab_hour = (active_bridge_loan + active_bridge_interest_accrued) + current_overdue_liability
        equity_bs_hour = total_assets_hour - total_liab_hour

        # Unrealized MTM gain/loss
        unrealized_mtm = 0.0
        for bp in active_bonds_p2:
            if bp.units and bp.entry_price:
                unrealized_mtm += bp.market_value - (bp.units * bp.entry_price)
        for ep in active_equities_p2:
            if ep.units and ep.entry_price:
                unrealized_mtm += ep.market_value - (ep.units * ep.entry_price)

        net_pnl_hour = yield_earned_p2 - penalties_p2 - bridge_interest_p2 - liquidation_losses_p2 + unrealized_mtm
        flow_equity_hour = INITIAL_CASH + (cum_deposits_p2 - cum_withdrawals_due_p2) + net_pnl_hour

        if abs(equity_bs_hour - flow_equity_hour) >= 0.05:
            raise AssertionError(
                f"Balance sheet reconciliation failed at hour {idx} ({ts}): "
                f"BS Equity={equity_bs_hour:.2f}, Flow Equity={flow_equity_hour:.2f}, "
                f"Diff={equity_bs_hour - flow_equity_hour:.4f}"
            )

    # -------------------------------------------------------------------------
    # WRITE AUDIT REPORT FILE
    # -------------------------------------------------------------------------
    default_audit_path = audit_file_path or os.path.join(
        "tests", "decision", "shortfall_audit_report.json"
    )
    try:
        os.makedirs(os.path.dirname(default_audit_path), exist_ok=True)
        with open(default_audit_path, "w", encoding="utf-8") as f:
            json.dump(shortfall_audit_records, f, indent=2)
        print(f"Shortfall audit report successfully saved ({len(shortfall_audit_records)} incidents): {default_audit_path}")
    except Exception as e:
        print(f"Warning: Failed to save shortfall audit file: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # END-OF-SIMULATION ASSET AUDIT & IN-HORIZON REPORTING
    # -------------------------------------------------------------------------
    in_horizon_ending_cash = cash_p2
    in_horizon_ending_invested = sum(
        p.market_value for p in active_repos_p2 + active_bonds_p2 + active_equities_p2
    ) + sum(ps["amount"] for ps in pending_settlements_p2)
    in_horizon_ending_unpaid = sum(item["amount"] for item in overdue_queue_p2)
    in_horizon_ending_debt = active_bridge_loan + active_bridge_interest_accrued

    # -------------------------------------------------------------------------
    # POST-HORIZON TERMINAL LIQUIDATION ANALYSIS
    # -------------------------------------------------------------------------
    # Liquidate remaining repos, bonds, equities, and settle receivables post-horizon:
    post_horizon_cash = in_horizon_ending_cash
    post_horizon_unpaid = in_horizon_ending_unpaid

    # Mature remaining repos (haircut-free at maturity)
    for rp in active_repos_p2:
        earned = rp.market_value * (rp.annual_yield * (15.0 / 8760.0))
        post_horizon_cash += rp.market_value + earned

    # Liquidate bonds net of haircuts
    for bp in active_bonds_p2:
        post_horizon_cash += bp.market_value * (1.0 - bp.haircut)

    # Liquidate equities net of haircuts
    for ep in active_equities_p2:
        post_horizon_cash += ep.market_value * (1.0 - ep.haircut)

    # Collect pending receivables
    for ps in pending_settlements_p2:
        post_horizon_cash += ps["amount"]

    # Settle debt first if any
    if in_horizon_ending_debt > 0.0:
        debt_paid = min(post_horizon_cash, in_horizon_ending_debt)
        post_horizon_cash -= debt_paid

    # Apply remaining liquid proceeds FIFO to overdue customer liabilities
    post_horizon_recovered = min(post_horizon_cash, post_horizon_unpaid)
    terminal_unpaid_after_liquidation = post_horizon_unpaid - post_horizon_recovered
    terminal_residual_cash = post_horizon_cash - post_horizon_recovered

    # -------------------------------------------------------------------------
    # PRINT PERFORMANCE BENCHMARK COMPARISON & FULL RECONCILIATION
    # -------------------------------------------------------------------------
    ending_assets_agg = sum(r[0] for r in repo_agg)

    net_pnl_naive = 0.0
    net_pnl_agg = yield_earned_agg - penalties_agg - overdraft_interest_agg

    ending_unrealized_mtm = 0.0
    for bp in active_bonds_p2:
        if bp.units and bp.entry_price:
            ending_unrealized_mtm += bp.market_value - (bp.units * bp.entry_price)
    for ep in active_equities_p2:
        if ep.units and ep.entry_price:
            ending_unrealized_mtm += ep.market_value - (ep.units * ep.entry_price)

    net_pnl_p2 = (
        yield_earned_p2
        - penalties_p2
        - bridge_interest_p2
        - liquidation_losses_p2
        + ending_unrealized_mtm
    )

    # Balance Sheet Identities (In-Horizon):
    total_assets_naive = cash_naive
    total_liab_naive = cumulative_unpaid_naive
    equity_naive = total_assets_naive - total_liab_naive

    total_assets_agg = cash_agg + ending_assets_agg
    total_liab_agg = cumulative_unpaid_agg + overdraft_interest_agg
    equity_agg = total_assets_agg - total_liab_agg

    total_assets_p2 = in_horizon_ending_cash + in_horizon_ending_invested
    total_liab_p2 = in_horizon_ending_debt + in_horizon_ending_unpaid
    equity_p2 = total_assets_p2 - total_liab_p2

    # Flow Reconciliation:
    total_deposits_all = float(df["deposits_usd"].sum())
    total_withdrawals_all = float(df["withdrawals_usd"].sum())
    net_external_flow = total_deposits_all - total_withdrawals_all

    flow_equity_naive = INITIAL_CASH + net_external_flow + net_pnl_naive
    flow_equity_agg = INITIAL_CASH + net_external_flow + net_pnl_agg
    flow_equity_p2 = INITIAL_CASH + net_external_flow + net_pnl_p2

    discrepancy_naive = equity_naive - flow_equity_naive
    discrepancy_agg = equity_agg - flow_equity_agg
    discrepancy_p2 = equity_p2 - flow_equity_p2

    agg_deficit_naive = total_withdrawals_all - INITIAL_CASH - total_deposits_all - net_pnl_naive
    agg_deficit_p2 = total_withdrawals_all - INITIAL_CASH - total_deposits_all - net_pnl_p2

    print("\n" + "=" * 80)
    print("8,760-HOUR BENCHMARK PERFORMANCE COMPARISON (treasury_4tier_dataset.csv)")
    print("=" * 80)
    print(f"{'Metric':<38} | {'1. Naive (Cash)':<15} | {'2. Aggressive':<15} | {'3. Person 2 Engine':<15}")
    print("-" * 89)
    print(f"{'Total Obligations Due ($)':<38} | ${total_payouts:>13,.2f} | ${total_payouts:>13,.2f} | ${total_payouts:>13,.2f}")
    print(f"{'Obligations Paid On-Time ($)':<38} | ${payouts_funded_naive:>13,.2f} | ${payouts_funded_agg:>13,.2f} | ${payouts_funded_on_time_p2:>13,.2f}")
    print(f"{'Remaining Unpaid Liabilities ($)':<38} | ${cumulative_unpaid_naive:>13,.2f} | ${cumulative_unpaid_agg:>13,.2f} | ${in_horizon_ending_unpaid:>13,.2f}")
    print(f"{'Aggregate Net Resource Deficit ($)':<38} | ${agg_deficit_naive:>13,.2f} | ${agg_deficit_naive - net_pnl_agg:>13,.2f} | ${agg_deficit_p2:>13,.2f}")
    print("-" * 89)
    print(f"{'Pre-Action Deficits Detected (Count)':<38} | {shortfall_events_naive:>14,d} | {shortfall_events_agg:>14,d} | {pre_action_deficits_p2:>14,d}")
    print(f"{'Mandatory Deadlines Missed (Count)':<38} | {shortfall_events_naive:>14,d} | {shortfall_events_agg:>14,d} | {deadline_miss_incidents_p2:>14,d}")
    print(f"{'Unresolved Funding Plans (Count)':<38} | {shortfall_events_naive:>14,d} | {shortfall_events_agg:>14,d} | {unresolved_funding_actions_p2:>14,d}")
    print(f"{'On-Time Fulfillment Rate (%)':<38} | {(payouts_funded_naive/total_payouts)*100:>13.4f}% | {(payouts_funded_agg/total_payouts)*100:>13.4f}% | {(payouts_funded_on_time_p2/total_payouts)*100:>13.4f}%")
    print("-" * 89)
    print(f"{'Gross Investment Yield ($)':<38} | ${0.0:>13,.2f} | ${yield_earned_agg:>13,.2f} | ${yield_earned_p2:>13,.2f}")
    print(f"{'Early Exit Penalties ($)':<38} | ${0.0:>13,.2f} | ${penalties_agg:>13,.2f} | ${penalties_p2:>13,.2f}")
    print(f"{'Financing / Overdraft Cost ($)':<38} | ${0.0:>13,.2f} | ${overdraft_interest_agg:>13,.2f} | ${bridge_interest_p2:>13,.2f}")
    print(f"{'Liquidation Losses ($)':<38} | ${0.0:>13,.2f} | ${0.0:>13,.2f} | ${liquidation_losses_p2:>13,.2f}")
    print(f"{'Net Treasury PnL ($)':<38} | ${net_pnl_naive:>13,.2f} | ${net_pnl_agg:>13,.2f} | ${net_pnl_p2:>13,.2f}")
    print("-" * 89)
    print(f"{'Ending Cash ($)':<38} | ${cash_naive:>13,.2f} | ${cash_agg:>13,.2f} | ${in_horizon_ending_cash:>13,.2f}")
    print(f"{'Ending Invested Assets ($)':<38} | ${0.0:>13,.2f} | ${ending_assets_agg:>13,.2f} | ${in_horizon_ending_invested:>13,.2f}")
    print(f"{'  - Tier 1 Overnight Repo ($)':<38} | ${0.0:>13,.2f} | ${ending_assets_agg:>13,.2f} | ${sum(float(rp.market_value) for rp in active_repos_p2):>13,.2f}")
    print(f"{'  - Tier 2 Sovereign Bonds (T+24h) ($)':<38} | ${0.0:>13,.2f} | ${0.0:>13,.2f} | ${sum(float(bp.market_value) for bp in active_bonds_p2):>13,.2f}")
    print(f"{'  - Tier 3 Growth Equities (T+48h) ($)':<38} | ${0.0:>13,.2f} | ${0.0:>13,.2f} | ${sum(float(ep.market_value) for ep in active_equities_p2):>13,.2f}")
    print(f"{'Outstanding Credit Debt ($)':<38} | ${0.0:>13,.2f} | ${0.0:>13,.2f} | ${in_horizon_ending_debt:>13,.2f}")
    print(f"{'Total Balance Sheet Liabilities ($)':<38} | ${total_liab_naive:>13,.2f} | ${total_liab_agg:>13,.2f} | ${total_liab_p2:>13,.2f}")
    print(f"{'Net Treasury Equity (BS) ($)':<38} | ${equity_naive:>13,.2f} | ${equity_agg:>13,.2f} | ${equity_p2:>13,.2f}")
    print(f"{'Net Treasury Equity (Flow) ($)':<38} | ${flow_equity_naive:>13,.2f} | ${flow_equity_agg:>13,.2f} | ${flow_equity_p2:>13,.2f}")
    print(f"{'Reconciliation Discrepancy ($)':<38} | ${discrepancy_naive:>13,.2f} | ${discrepancy_agg:>13,.2f} | ${discrepancy_p2:>13,.2f}")
    print("-" * 89)
    print(f"{'Average Safety Score (TSS)':<38} | {'N/A':>14} | {'N/A':>14} | {np.mean(tss_history_p2):>14.2f}")
    print(f"{'Minimum Safety Score (TSS)':<38} | {'N/A':>14} | {'N/A':>14} | {np.min(tss_history_p2):>14.2f}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("POST-HORIZON TERMINAL LIQUIDATION ANALYSIS (Recovery Beyond Hour 8,760)")
    print("=" * 80)
    print(f"Post-Horizon Liquidated Asset Proceeds: ${post_horizon_cash:>15,.2f}")
    print(f"Applied to Overdue Customer Payouts:    ${post_horizon_recovered:>15,.2f}")
    print(f"Terminal Overdue Customer Liabilities:  ${terminal_unpaid_after_liquidation:>15,.2f}")
    print(f"Terminal Residual Cash:                 ${terminal_residual_cash:>15,.2f}")
    print(f"Post-Liquidation Total Fulfillment Rate: {((payouts_funded_on_time_p2 + post_horizon_recovered)/total_payouts)*100:.4f}%")
    print("=" * 80)

    print("\nPerson 2 Decision Engine Action Breakdown:")
    for act, cnt in sorted(action_counts_p2.items(), key=lambda x: -x[1]):
        pct = (cnt / total_hours) * 100
        print(f"  - {act:<30}: {cnt:>5} ({pct:>5.1f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Full-year treasury simulation on 4-tier dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to treasury_4tier_dataset.csv")
    parser.add_argument(
        "--audit-file",
        type=str,
        default=os.path.join("tests", "decision", "shortfall_audit_report.json"),
        help="Path to output shortfall audit JSON report",
    )
    args = parser.parse_args()

    simulate_all_strategies(args.dataset, audit_file_path=args.audit_file)


if __name__ == "__main__":
    main()

