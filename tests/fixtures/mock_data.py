"""
Shared test fixtures and sample data for Team 3 test suites.
"""

from __future__ import annotations
from datetime import datetime, timezone
from reserve_juggler.contracts import (
    AssetPosition,
    DecisionRecord,
    Obligation,
    TreasuryState,
)


def make_sample_treasury_state(available_cash: float = 50_000_000.0) -> TreasuryState:
    now_iso = datetime.now(timezone.utc).isoformat()
    return TreasuryState(
        timestamp=now_iso,
        available_cash=available_cash,
        asset_positions=[
            AssetPosition(
                asset_id="REPO-01",
                name="Tri-Party Overnight Repo",
                asset_class="REPO",
                market_value=30_000_000.0,
                accessible_liquidity=30_000_000.0,
                settlement_days=0,
                yield_pct=5.3,
            )
        ],
        obligations=[
            Obligation(
                obligation_id="OBL-01",
                due_timestamp=now_iso,
                amount=15_000_000.0,
                description="Payroll settlement",
            )
        ],
        authorized_credit_availability=25_000_000.0,
        protected_reserve_floor=20_000_000.0,
    )
