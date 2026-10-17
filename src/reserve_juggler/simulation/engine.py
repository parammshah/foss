"""Simulation engine for stepping the treasury ledger through multi-hour operating periods."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

from reserve_juggler.contracts import (
    EventPublisher,
    ExecutionOutcome,
    ForecastBundle,
    LearningEvent,
    NoOpEventPublisher,
    TreasuryState,
    ensure_utc,
)
from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    REPO_SWEEP_MATURE_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_PRICE_COL,
    TIER2_BOND_YIELD_COL,
    TIER3_STOCK_PRICE_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.ledger.ledger import TreasuryLedger
from reserve_juggler.simulation.actions import ActionType, TreasuryAction


@dataclass
class SimulationStepRecord:
    step_index: int
    timestamp: datetime
    deposits_usd: float
    withdrawals_usd: float
    net_flow_usd: float
    cash_before_usd: float
    cash_after_usd: float
    total_assets_market_value_usd: float
    total_liquidatable_value_usd: float
    shortfall_usd: float
    action_taken: Optional[TreasuryAction] = None
    outcome: Optional[ExecutionOutcome] = None
    forecast_bundle: Optional[ForecastBundle] = None


@dataclass
class SimulationResult:
    total_steps: int
    start_timestamp: datetime
    end_timestamp: datetime
    cumulative_deposits_usd: float
    cumulative_withdrawals_usd: float
    cumulative_shortfall_usd: float
    cumulative_penalties_usd: float
    cumulative_costs_usd: float
    final_cash_usd: float
    final_total_assets_usd: float
    step_records: List[SimulationStepRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Simulation Total Hours: {self.total_steps}",
            f"Time Range: {self.start_timestamp} -> {self.end_timestamp}",
            f"Total Deposits Processed: ${self.cumulative_deposits_usd:,.2f}",
            f"Total Withdrawals Processed: ${self.cumulative_withdrawals_usd:,.2f}",
            f"Total Liquidity Shortfall: ${self.cumulative_shortfall_usd:,.2f}",
            f"Total Execution Costs: ${self.cumulative_costs_usd:,.2f}",
            f"Total Penalties Paid: ${self.cumulative_penalties_usd:,.2f}",
            f"Final Operational Cash: ${self.final_cash_usd:,.2f}",
            f"Final Balance Sheet Assets: ${self.final_total_assets_usd:,.2f}",
        ]
        return "\n".join(lines)


class SimulationEngine:
    """Orchestrates hourly simulation of banking treasury operations."""

    def __init__(
        self,
        ledger: TreasuryLedger,
        timeline_df: pd.DataFrame,
        forecast_service: Optional[ForecastService] = None,
        event_publisher: Optional[EventPublisher] = None,
    ) -> None:
        self.ledger = ledger
        self.timeline_df = timeline_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.timeline_df[TIMESTAMP_COL]):
            self.timeline_df[TIMESTAMP_COL] = pd.to_datetime(self.timeline_df[TIMESTAMP_COL], utc=True)
        self.timeline_df = self.timeline_df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

        self.forecast_service = forecast_service
        self.publisher = event_publisher or NoOpEventPublisher()

    def run(
        self,
        start_step: int = 0,
        num_steps: Optional[int] = None,
        policy_callback: Optional[Callable[[TreasuryState, Optional[ForecastBundle]], Optional[TreasuryAction]]] = None,
    ) -> SimulationResult:
        """Run the simulation for num_steps beginning at start_step."""
        max_available = len(self.timeline_df) - start_step
        steps_to_run = min(num_steps if num_steps is not None else max_available, max_available)

        records: List[SimulationStepRecord] = []
        cum_shortfall = 0.0
        cum_costs = 0.0
        cum_penalties = 0.0
        cum_deps = 0.0
        cum_wids = 0.0

        for i in range(steps_to_run):
            step_idx = start_step + i
            row = self.timeline_df.iloc[step_idx]
            ts = ensure_utc(row[TIMESTAMP_COL].to_pydatetime() if hasattr(row[TIMESTAMP_COL], "to_pydatetime") else row[TIMESTAMP_COL])

            dep = float(row.get(DEPOSITS_COL, 0.0))
            wid = float(row.get(WITHDRAWALS_COL, 0.0))
            repo_mature = bool(row.get(REPO_SWEEP_MATURE_COL, False))

            cum_deps += dep
            cum_wids += wid

            # 1. Update market prices
            bond_price = float(row.get(TIER2_BOND_PRICE_COL, 100.0))
            stock_price = float(row.get(TIER3_STOCK_PRICE_COL, 5000.0))
            repo_rate = float(row.get(TIER1_REPO_RATE_COL, 0.043))
            bond_yield = float(row.get(TIER2_BOND_YIELD_COL, 0.048))
            stock_vol = float(row.get(TIER3_STOCK_VOLATILITY_COL, 0.18))

            self.ledger.update_market_prices(
                bond_price_index=bond_price,
                stock_price_index=stock_price,
                tier1_repo_rate_apy=repo_rate,
                tier2_bond_yield_pct=bond_yield,
                tier3_stock_volatility_pct=stock_vol,
            )

            # 2. Get current state & generate forecast if service is attached
            current_state = self.ledger.get_state(ts)
            forecast_bundle: Optional[ForecastBundle] = None
            if self.forecast_service is not None:
                try:
                    forecast_bundle = self.forecast_service.forecast(as_of_timestamp=ts, horizons_hours=[1, 6, 24])
                except Exception:
                    pass

            # 3. Policy decision & execution
            action: Optional[TreasuryAction] = None
            outcome: Optional[ExecutionOutcome] = None
            cash_before = self.ledger.cash_usd

            if policy_callback is not None:
                action = policy_callback(current_state, forecast_bundle)
                if action is not None:
                    new_state, outcome = self.ledger.simulate_action(current_state, action.model_dump(), forecast_bundle)
                    # Apply action outcome to real ledger
                    self.ledger.cash_usd = new_state.operational_cash_usd
                    self.ledger.positions = new_state.asset_positions
                    cum_costs += outcome.costs_usd
                    cum_penalties += outcome.penalties_usd

            # 4. Process customer cash flows and repo maturity
            net_flow, shortfall = self.ledger.process_hourly_step(
                current_timestamp=ts,
                deposits_usd=dep,
                withdrawals_usd=wid,
                repo_sweep_mature=repo_mature,
            )
            cum_shortfall += shortfall

            # Record step snapshot
            post_state = self.ledger.get_state(ts)
            rec = SimulationStepRecord(
                step_index=step_idx,
                timestamp=ts,
                deposits_usd=dep,
                withdrawals_usd=wid,
                net_flow_usd=net_flow,
                cash_before_usd=cash_before,
                cash_after_usd=post_state.operational_cash_usd,
                total_assets_market_value_usd=post_state.total_balance_sheet_assets_usd,
                total_liquidatable_value_usd=post_state.total_liquidatable_value_usd,
                shortfall_usd=shortfall,
                action_taken=action,
                outcome=outcome,
                forecast_bundle=forecast_bundle,
            )
            records.append(rec)

            # Emit learning event for simulation step
            now_utc = datetime.now(timezone.utc)
            evt = LearningEvent(
                event_id=f"sim_step_{uuid.uuid4().hex[:8]}",
                event_type="simulation_step_completed",
                source_module="reserve_juggler.simulation.engine",
                event_timestamp=ts,
                recording_timestamp=now_utc,
                payload={
                    "step_index": step_idx,
                    "operational_cash_usd": post_state.operational_cash_usd,
                    "total_liquidatable_usd": post_state.total_liquidatable_value_usd,
                    "shortfall_usd": shortfall,
                },
            )
            self.publisher.publish(evt)

        final_state = self.ledger.get_state(records[-1].timestamp if records else self.timeline_df[TIMESTAMP_COL].iloc[0])
        return SimulationResult(
            total_steps=steps_to_run,
            start_timestamp=records[0].timestamp if records else self.timeline_df[TIMESTAMP_COL].iloc[0],
            end_timestamp=records[-1].timestamp if records else self.timeline_df[TIMESTAMP_COL].iloc[0],
            cumulative_deposits_usd=cum_deps,
            cumulative_withdrawals_usd=cum_wids,
            cumulative_shortfall_usd=cum_shortfall,
            cumulative_penalties_usd=cum_penalties,
            cumulative_costs_usd=cum_costs,
            final_cash_usd=final_state.operational_cash_usd,
            final_total_assets_usd=final_state.total_balance_sheet_assets_usd,
            step_records=records,
        )
