"""Action definitions and schemas for treasury asset allocation and rebalancing."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    HOLD = "HOLD"
    SWEEP_REPO = "SWEEP_REPO"
    BREAK_REPO = "BREAK_REPO"
    BUY_BONDS = "BUY_BONDS"
    SELL_BONDS = "SELL_BONDS"
    BUY_EQUITIES = "BUY_EQUITIES"
    SELL_EQUITIES = "SELL_EQUITIES"
    DRAW_CREDIT = "DRAW_CREDIT"
    REPAY_CREDIT = "REPAY_CREDIT"


class TreasuryAction(BaseModel):
    """Structured allocation action submitted to the treasury ledger/simulator."""
    type: ActionType = Field(description="Action identifier")
    amount_usd: float = Field(default=0.0, ge=0.0, description="Amount in USD to sweep, buy, sell, draw, or repay")
    asset_id: Optional[str] = Field(default=None, description="Specific target asset ID if applicable")
    decision_id: Optional[str] = Field(default=None, description="Correlated decision ID from optimizer")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Supplementary execution parameters")
