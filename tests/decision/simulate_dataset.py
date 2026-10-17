"""
Reproducible 8,760-hour Treasury Simulation on treasury_4tier_dataset.csv.
Compares:
1. Naive Strategy (Cash-Only Buffer)
2. Aggressive Yield Chaser (Unconstrained Sweeper, ignores holding gates/reserves)
3. Repaired Person 2 Engine (TSS, Capital Firewall, Liquidity Shield, Dual-Mode Optimizer)
"""

import os
import sys

# Ensure src and workspace root are in Python path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

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

DATASET_PATH = r"C:\Users\bhavy\Downloads\treasury_4tier_dataset.csv"


def simulate_all_strategies():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    total_hours = len(df)
    print(f"Simulating 8,760 hours from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}...")

    # Shared initial conditions
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

    for idx, row in df.iterrows():
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        total_payouts += wth
        cash_naive += dep

        if cash_naive >= wth:
            cash_naive -= wth
            payouts_funded_naive += wth
        else:
            payouts_funded_naive += max(0.0, cash_naive)
            gap = wth - max(0.0, cash_naive)
            shortfalls_naive += gap
            shortfall_events_naive += 1
            cash_naive = 0.0

    # -------------------------------------------------------------------------
    # 2. AGGRESSIVE YIELD CHASER (Zero reserve, 100% surplus invested into repo)
    # -------------------------------------------------------------------------
    cash_agg = INITIAL_CASH
    repo_agg = []
    shortfalls_agg = 0.0
    shortfall_events_agg = 0
    penalties_agg = 0.0
    overdraft_interest_agg = 0.0
    yield_earned_agg = 0.0
    payouts_funded_agg = 0.0

    for idx, row in df.iterrows():
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        repo_rate = float(row["tier1_repo_rate_apy"]) / 100.0
        is_open = bool(row["repo_sweep_window_open"])
        is_mature = bool(row["repo_sweep_mature"])
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
            # Must break repo early with penalty!
            for r_amt, r_mat, r_rate in list(repo_agg):
                if needed <= 0:
                    break
                # Break early
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
                cash_agg = 0.0
            else:
                cash_agg -= wth
                payouts_funded_agg += wth
        else:
            cash_agg -= wth
            payouts_funded_agg += wth

        # Aggressively sweep ALL remaining cash if window is open
        if is_open and cash_agg > 100_000.0:
            sweep = cash_agg - 50_000.0
            cash_agg -= sweep
            repo_agg.append((sweep, ts + timedelta(hours=15), repo_rate))

    # -------------------------------------------------------------------------
    # 3. REPAIRED PERSON 2 DECISION ENGINE
    # -------------------------------------------------------------------------
    service = TreasuryDecisionService(policy_version="2.1.0-repaired", model_version="1.5.0")
    tax_profile = create_statutory_us_corp_profile()

    cash_p2 = INITIAL_CASH
    active_repos_p2 = []
    active_bridge_loan = 0.0
    bridge_due_ts = None
    shortfalls_p2 = 0.0
    shortfall_events_p2 = 0
    payouts_funded_p2 = 0.0
    yield_earned_p2 = 0.0
    bridge_interest_p2 = 0.0
    penalties_p2 = 0.0
    action_counts_p2 = {}
    tss_history_p2 = []

    for idx, row in df.iterrows():
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dep = float(row["deposits_usd"])
        wth = float(row["withdrawals_usd"])
        repo_rate = float(row["tier1_repo_rate_apy"]) / 100.0
        is_open = bool(row["repo_sweep_window_open"])
        overdraft_apr = float(row["emergency_overdraft_penalty_apr_pct"]) / 100.0

        cash_p2 += dep

        # Mature repos without early exit fee
        still_repos = []
        for rp in active_repos_p2:
            if rp.maturity <= ts:
                earned = rp.market_value * (rp.annual_yield * (15.0 / 8760.0))
                cash_p2 += rp.market_value + earned
                yield_earned_p2 += earned
            else:
                still_repos.append(rp)
        active_repos_p2 = still_repos

        # Repay bridge if due and cash available
        if active_bridge_loan > 0.0 and bridge_due_ts is not None and ts >= bridge_due_ts:
            if cash_p2 >= active_bridge_loan:
                cash_p2 -= active_bridge_loan
                active_bridge_loan = 0.0
                bridge_due_ts = None

        ob = ConcreteObligation(
            due_time=ts + timedelta(hours=1),
            amount=wth,
            is_mandatory=True,
            description="Hourly Customer Payout",
        )

        # 72-hour forecast bundle
        f_pts = []
        for h in [1, 2, 4, 8, 12, 24, 48, 72]:
            f_pts.append(
                ConcreteForecastPoint(
                    forecast_horizon=float(h),
                    deposit_p10=max(0.0, dep * 0.75 * (h / 24.0)),
                    deposit_p50=dep * 1.0 * (h / 24.0),
                    deposit_p90=dep * 1.35 * (h / 24.0),
                    withdrawal_p10=max(0.0, wth * 0.75 * (h / 24.0)),
                    withdrawal_p50=wth * 1.0 * (h / 24.0),
                    withdrawal_p90=wth * 1.35 * (h / 24.0),
                    reliability_metadata={"r_squared": 0.94, "forecast_age_hours": 0.25},
                )
            )
        bundle = ConcreteForecastBundle(point_forecasts=f_pts, timestamp=ts, quality_status="HEALTHY")

        credit_free = max(0.0, CREDIT_LIMIT - active_bridge_loan)
        state = ConcreteTreasuryState(
            timestamp=ts,
            available_cash=cash_p2,
            asset_positions=list(active_repos_p2),
            future_obligations=[ob],
            credit_availability=credit_free,
            credit_limit=CREDIT_LIMIT,
            credit_apr=overdraft_apr,
        )

        record = service.decide(
            state=state,
            forecast_bundle=bundle,
            tax_profile=tax_profile,
            repo_overnight_rate_apy=repo_rate,
            repo_sweep_lockup_hours=15.0,
        )

        sel = record.selected_action
        act_type = sel["action_type"]
        action_counts_p2[act_type] = action_counts_p2.get(act_type, 0) + 1
        tss_history_p2.append(record.score["composite_score"])

        # Execute decision
        if act_type == ActionType.DRAW_BRIDGE_CREDIT.value:
            borrowed = float(sel.get("principal_borrowed", 0.0) or sel["amount"])
            int_cost = float(sel.get("interest_cost", 0.0))
            active_bridge_loan += borrowed
            bridge_due_ts = ts + timedelta(hours=2)
            cash_p2 += borrowed - int_cost
            bridge_interest_p2 += int_cost

        elif act_type == ActionType.BREAK_REPO_EARLY.value:
            for rp in list(active_repos_p2):
                pen = rp.market_value * rp.exit_fees
                net_p = rp.market_value - pen
                cash_p2 += net_p
                penalties_p2 += pen
                active_repos_p2.remove(rp)
                break

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

        # Pay customer withdrawal
        if cash_p2 >= wth:
            cash_p2 -= wth
            payouts_funded_p2 += wth
        else:
            payouts_funded_p2 += max(0.0, cash_p2)
            gap = wth - max(0.0, cash_p2)
            shortfalls_p2 += gap
            shortfall_events_p2 += 1
            cash_p2 = 0.0

    # -------------------------------------------------------------------------
    # PRINT PERFORMANCE BENCHMARK COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8,760-HOUR BENCHMARK PERFORMANCE COMPARISON (treasury_4tier_dataset.csv)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'1. Naive (Cash)':<15} | {'2. Aggressive':<15} | {'3. Person 2 Engine':<15}")
    print("-" * 86)
    print(f"{'Total Obligations Due ($)':<35} | ${total_payouts:>13,.0f} | ${total_payouts:>13,.0f} | ${total_payouts:>13,.0f}")
    print(f"{'Total Obligations Funded ($)':<35} | ${payouts_funded_naive:>13,.0f} | ${payouts_funded_agg:>13,.0f} | ${payouts_funded_p2:>13,.0f}")
    print(f"{'Total Payout Shortfalls ($)':<35} | ${shortfalls_naive:>13,.0f} | ${shortfalls_agg:>13,.0f} | ${shortfalls_p2:>13,.0f}")
    print(f"{'Shortfall Incidents (Count)':<35} | {shortfall_events_naive:>14,d} | {shortfall_events_agg:>14,d} | {shortfall_events_p2:>14,d}")
    print(f"{'Payout Fulfillment Rate (%)':<35} | {(payouts_funded_naive/total_payouts)*100:>13.4f}% | {(payouts_funded_agg/total_payouts)*100:>13.4f}% | {(payouts_funded_p2/total_payouts)*100:>13.4f}%")
    print("-" * 86)
    print(f"{'Gross Investment Yield ($)':<35} | ${0.0:>13,.2f} | ${yield_earned_agg:>13,.2f} | ${yield_earned_p2:>13,.2f}")
    print(f"{'Early Exit Penalties ($)':<35} | ${0.0:>13,.2f} | ${penalties_agg:>13,.2f} | ${penalties_p2:>13,.2f}")
    print(f"{'Financing / Overdraft Cost ($)':<35} | ${0.0:>13,.2f} | ${overdraft_interest_agg:>13,.2f} | ${bridge_interest_p2:>13,.2f}")
    net_pnl_naive = 0.0
    net_pnl_agg = yield_earned_agg - penalties_agg - overdraft_interest_agg
    net_pnl_p2 = yield_earned_p2 - penalties_p2 - bridge_interest_p2
    print(f"{'Net Treasury PnL ($)':<35} | ${net_pnl_naive:>13,.2f} | ${net_pnl_agg:>13,.2f} | ${net_pnl_p2:>13,.2f}")
    print("-" * 86)
    print(f"{'Average Safety Score (TSS)':<35} | {'N/A':>14} | {'N/A':>14} | {np.mean(tss_history_p2):>14.2f}")
    print(f"{'Minimum Safety Score (TSS)':<35} | {'N/A':>14} | {'N/A':>14} | {np.min(tss_history_p2):>14.2f}")
    print("=" * 80)
    print("\nPerson 2 Decision Engine Action Breakdown:")
    for act, cnt in sorted(action_counts_p2.items(), key=lambda x: -x[1]):
        pct = (cnt / total_hours) * 100
        print(f"  - {act:<30}: {cnt:>5} ({pct:>5.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    simulate_all_strategies()
