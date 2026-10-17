"""Treasury ledger implementing hourly double-entry accounting, asset tiers, and action simulation."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from reserve_juggler.contracts import (
    AssetPosition,
    AssetTier,
    CreditFacility,
    EventPublisher,
    ExecutionOutcome,
    ForecastBundle,
    LearningEvent,
    NoOpEventPublisher,
    Obligation,
    ObligationType,
    TreasuryState,
    ensure_utc,
)
from reserve_juggler.ledger.assets import (
    AssetTierConfig,
    DEFAULT_TIER0_CONFIG,
    DEFAULT_TIER1_CONFIG,
    DEFAULT_TIER2_CONFIG,
    DEFAULT_TIER3_CONFIG,
    calculate_liquidatable_value,
)
from reserve_juggler.ledger.credit import CreditManager


@dataclass
class InitialTreasuryConfig:
    """Explicitly configurable opening conditions for the treasury ledger.

    Zero values are allowed, but the structure must be explicitly defined rather than silently guessed.
    """
    opening_cash_usd: float
    initial_assets: Dict[str, AssetPosition] = field(default_factory=dict)
    credit_facility: Optional[CreditFacility] = None
    transaction_cost_rate: float = 0.0005  # 5 bps standard brokerage transaction cost
    tier0_config: AssetTierConfig = field(default_factory=lambda: DEFAULT_TIER0_CONFIG)
    tier1_config: AssetTierConfig = field(default_factory=lambda: DEFAULT_TIER1_CONFIG)
    tier2_config: AssetTierConfig = field(default_factory=lambda: DEFAULT_TIER2_CONFIG)
    tier3_config: AssetTierConfig = field(default_factory=lambda: DEFAULT_TIER3_CONFIG)


class TreasuryLedger:
    """Core hourly treasury accounting ledger for digital bank operations."""

    def __init__(
        self,
        config: Optional[InitialTreasuryConfig] = None,
        event_publisher: Optional[EventPublisher] = None,
    ) -> None:
        if config is None:
            config = InitialTreasuryConfig(opening_cash_usd=25_000_000.0)
        self.config = config
        self.publisher = event_publisher or NoOpEventPublisher()

        # Operational cash balance (Tier 0)
        self.cash_usd: float = float(config.opening_cash_usd)

        # Asset positions: dict of asset_id -> AssetPosition
        self.positions: Dict[str, AssetPosition] = copy.deepcopy(config.initial_assets)

        # Pending obligations
        self.obligations: List[Obligation] = []

        # Credit Facility Manager
        self.credit_manager: Optional[CreditManager] = None
        if config.credit_facility is not None:
            self.credit_manager = CreditManager(config.credit_facility)

        # Accounting accumulators
        self.cumulative_deposits_usd: float = 0.0
        self.cumulative_withdrawals_usd: float = 0.0
        self.cumulative_interest_earned_usd: float = 0.0
        self.cumulative_financing_costs_usd: float = 0.0
        self.cumulative_penalties_paid_usd: float = 0.0
        self.cumulative_transaction_costs_usd: float = 0.0
        self.cumulative_shortfall_usd: float = 0.0

        # Current marked-to-market prices
        self.market_prices: Dict[str, float] = {
            "T0_CASH": 1.0,
            "T1_REPO": 1.0,
            "T2_BONDS": 100.0,
            "T3_EQUITIES": 5000.0,
        }

        # Warnings log
        self._warnings: List[str] = []

    def get_state(self, as_of_timestamp: datetime) -> TreasuryState:
        """Returns the canonical point-in-time TreasuryState."""
        utc_ts = ensure_utc(as_of_timestamp)
        facility_contract = self.credit_manager.facility if self.credit_manager else None

        return TreasuryState(
            timestamp=utc_ts,
            operational_cash_usd=self.cash_usd,
            asset_positions=copy.deepcopy(self.positions),
            obligations=[copy.deepcopy(ob) for ob in self.obligations if not ob.is_settled],
            credit_facility=facility_contract,
            warnings=list(self._warnings),
        )

    def add_obligation(self, obligation: Obligation) -> None:
        """Register an upcoming payout or settlement obligation."""
        self.obligations.append(obligation)

    def update_market_prices(
        self,
        bond_price_index: Optional[float] = None,
        stock_price_index: Optional[float] = None,
        tier1_repo_rate_apy: Optional[float] = None,
        tier2_bond_yield_pct: Optional[float] = None,
        tier3_stock_volatility_pct: Optional[float] = None,
    ) -> None:
        """Mark positions to market using latest market feed."""
        if bond_price_index is not None:
            self.market_prices["T2_BONDS"] = bond_price_index
        if stock_price_index is not None:
            self.market_prices["T3_EQUITIES"] = stock_price_index

        # Update marked-to-market asset positions
        for asset_id, pos in list(self.positions.items()):
            if pos.tier == AssetTier.TIER_2_BONDS and bond_price_index is not None:
                new_mkt = pos.units * bond_price_index
                new_liq = calculate_liquidatable_value(
                    tier=pos.tier,
                    market_value_usd=new_mkt,
                    early_exit_fee_rate=pos.early_exit_fee_rate,
                    liquidation_haircut_rate=pos.liquidation_haircut_rate,
                )
                self.positions[asset_id] = AssetPosition(
                    asset_id=pos.asset_id,
                    tier=pos.tier,
                    units=pos.units,
                    unit_price_usd=bond_price_index,
                    market_value_usd=new_mkt,
                    liquidatable_value_usd=new_liq,
                    availability_timestamp=pos.availability_timestamp,
                    maturity_timestamp=pos.maturity_timestamp,
                    settlement_delay_hours=pos.settlement_delay_hours,
                    annual_yield_or_return=tier2_bond_yield_pct if tier2_bond_yield_pct is not None else pos.annual_yield_or_return,
                    early_exit_fee_rate=pos.early_exit_fee_rate,
                    liquidation_haircut_rate=pos.liquidation_haircut_rate,
                    duration_years=pos.duration_years,
                    accrued_income_usd=pos.accrued_income_usd,
                    cost_basis_usd=pos.cost_basis_usd,
                )
            elif pos.tier == AssetTier.TIER_3_EQUITIES and stock_price_index is not None:
                new_mkt = pos.units * stock_price_index
                new_liq = calculate_liquidatable_value(
                    tier=pos.tier,
                    market_value_usd=new_mkt,
                    early_exit_fee_rate=pos.early_exit_fee_rate,
                    liquidation_haircut_rate=pos.liquidation_haircut_rate,
                )
                self.positions[asset_id] = AssetPosition(
                    asset_id=pos.asset_id,
                    tier=pos.tier,
                    units=pos.units,
                    unit_price_usd=stock_price_index,
                    market_value_usd=new_mkt,
                    liquidatable_value_usd=new_liq,
                    availability_timestamp=pos.availability_timestamp,
                    maturity_timestamp=pos.maturity_timestamp,
                    settlement_delay_hours=pos.settlement_delay_hours,
                    annual_yield_or_return=pos.annual_yield_or_return,
                    early_exit_fee_rate=pos.early_exit_fee_rate,
                    liquidation_haircut_rate=pos.liquidation_haircut_rate,
                    annualized_volatility=tier3_stock_volatility_pct if tier3_stock_volatility_pct is not None else pos.annualized_volatility,
                    accrued_income_usd=pos.accrued_income_usd,
                    cost_basis_usd=pos.cost_basis_usd,
                )

    def process_hourly_step(
        self,
        current_timestamp: datetime,
        deposits_usd: float,
        withdrawals_usd: float,
        repo_sweep_mature: bool,
    ) -> Tuple[float, float]:
        """Executes 1 hourly bank operations step:
        1. Checks for repo maturity (e.g. at 08:00): credits principal + earned interest to cash.
        2. Accrues hourly yield / interest on bonds, repo, and credit facility.
        3. Credits customer deposits and debits customer withdrawals.
        4. Settles due obligations.
        5. Handles any liquidity shortfall via emergency credit facility if needed.

        Returns (net_cash_flow_usd, shortfall_usd).
        """
        ts = ensure_utc(current_timestamp)
        hourly_dt = 1.0 / 8760.0

        # 1. Repo maturity check
        if repo_sweep_mature:
            for asset_id, pos in list(self.positions.items()):
                if pos.tier == AssetTier.TIER_1_REPO:
                    principal = pos.market_value_usd
                    interest = pos.accrued_income_usd
                    total_payout = principal + interest
                    self.cash_usd += total_payout
                    self.cumulative_interest_earned_usd += interest
                    del self.positions[asset_id]

                    # Emit repo maturity learning event
                    evt = LearningEvent(
                        event_id=f"repo_mat_{uuid.uuid4().hex[:8]}",
                        event_type="repo_matured",
                        source_module="reserve_juggler.ledger",
                        event_timestamp=ts,
                        recording_timestamp=datetime.now(timezone.utc),
                        payload={
                            "asset_id": asset_id,
                            "principal_usd": principal,
                            "interest_earned_usd": interest,
                            "total_credited_usd": total_payout,
                        },
                    )
                    self.publisher.publish(evt)

        # 2. Hourly income accrual on remaining positions
        for asset_id, pos in list(self.positions.items()):
            if pos.tier == AssetTier.TIER_1_REPO:
                hourly_rate = pos.annual_yield_or_return * hourly_dt
                accrued = pos.market_value_usd * hourly_rate
                new_accrued = pos.accrued_income_usd + accrued
                self.positions[asset_id] = pos.model_copy(update={"accrued_income_usd": new_accrued})
            elif pos.tier == AssetTier.TIER_2_BONDS:
                hourly_yield = pos.annual_yield_or_return * hourly_dt
                accrued = pos.market_value_usd * hourly_yield
                new_accrued = pos.accrued_income_usd + accrued
                self.positions[asset_id] = pos.model_copy(update={"accrued_income_usd": new_accrued})

        # Accrue credit facility financing costs
        if self.credit_manager is not None:
            cost = self.credit_manager.accrue_hourly()
            self.cumulative_financing_costs_usd += cost

        # 3. Process customer cash flows
        self.cash_usd += deposits_usd
        self.cumulative_deposits_usd += deposits_usd

        # 4. Process customer withdrawals and due obligations
        obligations_due = [ob for ob in self.obligations if not ob.is_settled and ob.timestamp <= ts]
        total_obligations_usd = sum(ob.amount_usd for ob in obligations_due)
        total_payout_demand = withdrawals_usd + total_obligations_usd

        shortfall = 0.0
        if self.cash_usd >= total_payout_demand:
            self.cash_usd -= total_payout_demand
            self.cumulative_withdrawals_usd += withdrawals_usd
            for ob in obligations_due:
                # Mark settled
                idx = self.obligations.index(ob)
                self.obligations[idx] = ob.model_copy(update={"is_settled": True})
        else:
            # Operational cash deficit
            deficit = total_payout_demand - self.cash_usd
            drawn_from_credit = 0.0

            if self.credit_manager and self.credit_manager.available_capacity_usd > 0:
                drawn_from_credit = self.credit_manager.draw(deficit)
                self.cash_usd += drawn_from_credit

            if self.cash_usd >= total_payout_demand:
                self.cash_usd -= total_payout_demand
                self.cumulative_withdrawals_usd += withdrawals_usd
                for ob in obligations_due:
                    idx = self.obligations.index(ob)
                    self.obligations[idx] = ob.model_copy(update={"is_settled": True})
            else:
                # Unfunded liquidity shortfall!
                shortfall = total_payout_demand - self.cash_usd
                self.cash_usd = 0.0
                self.cumulative_shortfall_usd += shortfall
                self._warnings.append(f"CRITICAL SHORTFALL at {ts}: ${shortfall:,.2f}")

                # Emit shortfall learning event
                evt = LearningEvent(
                    event_id=f"shortfall_{uuid.uuid4().hex[:8]}",
                    event_type="liquidity_shortfall",
                    source_module="reserve_juggler.ledger",
                    event_timestamp=ts,
                    recording_timestamp=datetime.now(timezone.utc),
                    payload={
                        "shortfall_usd": shortfall,
                        "payout_demand_usd": total_payout_demand,
                        "credit_drawn_usd": drawn_from_credit,
                    },
                )
                self.publisher.publish(evt)

        net_cash_flow = deposits_usd - withdrawals_usd
        return net_cash_flow, shortfall

    def simulate_action(
        self,
        state: TreasuryState,
        action: Union[Dict[str, Any], Any],
        forecast_bundle: Optional[ForecastBundle] = None,
    ) -> Tuple[TreasuryState, ExecutionOutcome]:
        """Simulate the execution of an allocation action against a given state.

        Supported actions:
        - {"type": "HOLD"}
        - {"type": "SWEEP_REPO", "amount_usd": float}
        - {"type": "BREAK_REPO", "asset_id": str, "amount_usd": float}
        - {"type": "BUY_BONDS", "amount_usd": float}
        - {"type": "SELL_BONDS", "asset_id": str, "amount_usd": float}
        - {"type": "BUY_EQUITIES", "amount_usd": float}
        - {"type": "SELL_EQUITIES", "asset_id": str, "amount_usd": float}
        - {"type": "DRAW_CREDIT", "amount_usd": float}
        - {"type": "REPAY_CREDIT", "amount_usd": float}
        """
        # Create an independent sandbox clone of self
        sandbox = TreasuryLedger(self.config, event_publisher=NoOpEventPublisher())
        sandbox.cash_usd = state.operational_cash_usd
        sandbox.positions = copy.deepcopy(state.asset_positions)
        sandbox.obligations = copy.deepcopy(state.obligations)
        if state.credit_facility:
            sandbox.credit_manager = CreditManager(state.credit_facility)
        sandbox.market_prices = dict(self.market_prices)

        action_dict = action if isinstance(action, dict) else (action.model_dump() if hasattr(action, "model_dump") else dict(action))
        action_type = action_dict.get("type", "HOLD").upper()
        amount_usd = float(action_dict.get("amount_usd", 0.0))
        asset_id = str(action_dict.get("asset_id", ""))
        decision_id = str(action_dict.get("decision_id", f"sim_{uuid.uuid4().hex[:8]}"))

        costs = 0.0
        penalties = 0.0
        losses = 0.0
        shortfall = 0.0
        status = "SUCCESS"
        cash_movements: Dict[str, float] = {}

        if action_type == "HOLD":
            status = "SUCCESS"

        elif action_type == "SWEEP_REPO":
            # Invest into overnight repo: minimum available cash required
            if amount_usd > sandbox.cash_usd:
                status = "REJECTED_INSUFFICIENT_CASH"
            else:
                sandbox.cash_usd -= amount_usd
                repo_id = f"T1_REPO_{state.timestamp.strftime('%Y%m%d_%H')}"
                sandbox.positions[repo_id] = AssetPosition(
                    asset_id=repo_id,
                    tier=AssetTier.TIER_1_REPO,
                    units=amount_usd,
                    unit_price_usd=1.0,
                    market_value_usd=amount_usd,
                    liquidatable_value_usd=amount_usd * (1.0 - sandbox.config.tier1_config.early_exit_fee_rate),
                    annual_yield_or_return=sandbox.config.tier1_config.annual_rate,
                    early_exit_fee_rate=sandbox.config.tier1_config.early_exit_fee_rate,
                    maturity_timestamp=state.timestamp + timedelta(hours=15),
                    cost_basis_usd=amount_usd,
                )
                cash_movements["sweep_repo"] = -amount_usd

        elif action_type == "BREAK_REPO":
            # Early break of repo position before maturity
            target_id = asset_id or next((k for k, v in sandbox.positions.items() if v.tier == AssetTier.TIER_1_REPO), None)
            if not target_id or target_id not in sandbox.positions:
                status = "REJECTED_ASSET_NOT_FOUND"
            else:
                pos = sandbox.positions[target_id]
                break_amount = min(amount_usd if amount_usd > 0 else pos.market_value_usd, pos.market_value_usd)
                penalty = break_amount * pos.early_exit_fee_rate
                net_proceeds = break_amount - penalty

                penalties += penalty
                sandbox.cash_usd += net_proceeds
                cash_movements["break_repo_proceeds"] = net_proceeds
                cash_movements["early_exit_penalty"] = -penalty

                remaining_val = pos.market_value_usd - break_amount
                if remaining_val <= 1.0:
                    del sandbox.positions[target_id]
                else:
                    sandbox.positions[target_id] = pos.model_copy(update={
                        "units": remaining_val,
                        "market_value_usd": remaining_val,
                        "liquidatable_value_usd": remaining_val * (1.0 - pos.early_exit_fee_rate),
                    })

        elif action_type == "BUY_BONDS":
            fee = amount_usd * sandbox.config.transaction_cost_rate
            total_cost = amount_usd + fee
            if total_cost > sandbox.cash_usd:
                status = "REJECTED_INSUFFICIENT_CASH"
            else:
                sandbox.cash_usd -= total_cost
                costs += fee
                price = sandbox.market_prices.get("T2_BONDS", 100.0)
                units = amount_usd / price
                pos_id = asset_id or "T2_BONDS_PORTFOLIO"

                existing = sandbox.positions.get(pos_id)
                new_units = units + (existing.units if existing else 0.0)
                new_mkt = new_units * price
                new_cost = amount_usd + (existing.cost_basis_usd if existing else 0.0)
                new_liq = calculate_liquidatable_value(
                    tier=AssetTier.TIER_2_BONDS,
                    market_value_usd=new_mkt,
                    early_exit_fee_rate=0.0,
                    liquidation_haircut_rate=sandbox.config.tier2_config.liquidation_haircut_rate,
                )
                sandbox.positions[pos_id] = AssetPosition(
                    asset_id=pos_id,
                    tier=AssetTier.TIER_2_BONDS,
                    units=new_units,
                    unit_price_usd=price,
                    market_value_usd=new_mkt,
                    liquidatable_value_usd=new_liq,
                    annual_yield_or_return=sandbox.config.tier2_config.annual_rate,
                    liquidation_haircut_rate=sandbox.config.tier2_config.liquidation_haircut_rate,
                    duration_years=sandbox.config.tier2_config.duration_years,
                    cost_basis_usd=new_cost,
                )
                cash_movements["buy_bonds"] = -amount_usd
                cash_movements["transaction_fee"] = -fee

        elif action_type == "SELL_BONDS":
            pos_id = asset_id or "T2_BONDS_PORTFOLIO"
            if pos_id not in sandbox.positions:
                status = "REJECTED_ASSET_NOT_FOUND"
            else:
                pos = sandbox.positions[pos_id]
                sell_val = min(amount_usd if amount_usd > 0 else pos.market_value_usd, pos.market_value_usd)
                haircut = sell_val * pos.liquidation_haircut_rate
                fee = sell_val * sandbox.config.transaction_cost_rate
                net_proceeds = sell_val - haircut - fee

                losses += haircut
                costs += fee
                sandbox.cash_usd += net_proceeds
                cash_movements["sell_bonds_proceeds"] = net_proceeds
                cash_movements["liquidation_haircut"] = -haircut

                rem_val = pos.market_value_usd - sell_val
                if rem_val <= 1.0:
                    del sandbox.positions[pos_id]
                else:
                    new_units = rem_val / pos.unit_price_usd
                    sandbox.positions[pos_id] = pos.model_copy(update={
                        "units": new_units,
                        "market_value_usd": rem_val,
                        "liquidatable_value_usd": rem_val * (1.0 - pos.liquidation_haircut_rate),
                    })

        elif action_type == "BUY_EQUITIES":
            fee = amount_usd * sandbox.config.transaction_cost_rate
            total_cost = amount_usd + fee
            if total_cost > sandbox.cash_usd:
                status = "REJECTED_INSUFFICIENT_CASH"
            else:
                sandbox.cash_usd -= total_cost
                costs += fee
                price = sandbox.market_prices.get("T3_EQUITIES", 5000.0)
                units = amount_usd / price
                pos_id = asset_id or "T3_EQUITIES_INDEX"

                existing = sandbox.positions.get(pos_id)
                new_units = units + (existing.units if existing else 0.0)
                new_mkt = new_units * price
                new_cost = amount_usd + (existing.cost_basis_usd if existing else 0.0)
                new_liq = calculate_liquidatable_value(
                    tier=AssetTier.TIER_3_EQUITIES,
                    market_value_usd=new_mkt,
                    early_exit_fee_rate=0.0,
                    liquidation_haircut_rate=sandbox.config.tier3_config.liquidation_haircut_rate,
                )
                sandbox.positions[pos_id] = AssetPosition(
                    asset_id=pos_id,
                    tier=AssetTier.TIER_3_EQUITIES,
                    units=new_units,
                    unit_price_usd=price,
                    market_value_usd=new_mkt,
                    liquidatable_value_usd=new_liq,
                    annual_yield_or_return=sandbox.config.tier3_config.annual_rate,
                    liquidation_haircut_rate=sandbox.config.tier3_config.liquidation_haircut_rate,
                    annualized_volatility=sandbox.config.tier3_config.annualized_volatility,
                    cost_basis_usd=new_cost,
                )
                cash_movements["buy_equities"] = -amount_usd
                cash_movements["transaction_fee"] = -fee

        elif action_type == "SELL_EQUITIES":
            pos_id = asset_id or "T3_EQUITIES_INDEX"
            if pos_id not in sandbox.positions:
                status = "REJECTED_ASSET_NOT_FOUND"
            else:
                pos = sandbox.positions[pos_id]
                sell_val = min(amount_usd if amount_usd > 0 else pos.market_value_usd, pos.market_value_usd)
                slippage = sell_val * pos.liquidation_haircut_rate
                fee = sell_val * sandbox.config.transaction_cost_rate
                net_proceeds = sell_val - slippage - fee

                losses += slippage
                costs += fee
                sandbox.cash_usd += net_proceeds
                cash_movements["sell_equities_proceeds"] = net_proceeds
                cash_movements["slippage_loss"] = -slippage

                rem_val = pos.market_value_usd - sell_val
                if rem_val <= 1.0:
                    del sandbox.positions[pos_id]
                else:
                    new_units = rem_val / pos.unit_price_usd
                    sandbox.positions[pos_id] = pos.model_copy(update={
                        "units": new_units,
                        "market_value_usd": rem_val,
                        "liquidatable_value_usd": rem_val * (1.0 - pos.liquidation_haircut_rate),
                    })

        elif action_type == "DRAW_CREDIT":
            if not sandbox.credit_manager:
                status = "REJECTED_NO_CREDIT_FACILITY"
            else:
                drawn = sandbox.credit_manager.draw(amount_usd)
                if drawn <= 0.0:
                    status = "REJECTED_LIMIT_EXCEEDED"
                else:
                    sandbox.cash_usd += drawn
                    cash_movements["draw_credit"] = drawn

        elif action_type == "REPAY_CREDIT":
            if not sandbox.credit_manager:
                status = "REJECTED_NO_CREDIT_FACILITY"
            else:
                repay_amt = min(amount_usd, sandbox.cash_usd)
                actual_repaid = sandbox.credit_manager.repay(repay_amt)
                sandbox.cash_usd -= actual_repaid
                cash_movements["repay_credit"] = -actual_repaid

        else:
            status = f"REJECTED_UNKNOWN_ACTION_{action_type}"

        new_state = sandbox.get_state(state.timestamp)
        outcome = ExecutionOutcome(
            outcome_id=f"out_{uuid.uuid4().hex[:8]}",
            decision_id=decision_id,
            execution_status=status,
            realized_cash_movements=cash_movements,
            costs_usd=costs,
            penalties_usd=penalties,
            losses_usd=losses,
            shortfall_usd=shortfall,
            timestamp=state.timestamp,
            reconciliation_status="RECONCILED",
        )
        return new_state, outcome
