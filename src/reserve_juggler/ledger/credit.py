"""Credit facility management, borrowing capacity rules, and financing cost accrual."""

from __future__ import annotations

from typing import Dict, Any, Optional
from reserve_juggler.contracts import CreditFacility


class CreditManager:
    """Manages emergency borrowing, repayment, and hourly APR / fee accruals."""

    def __init__(self, facility: CreditFacility) -> None:
        self._facility = facility
        self.accrued_interest_usd: float = 0.0
        self.accrued_commitment_fee_usd: float = 0.0

    @property
    def facility(self) -> CreditFacility:
        return self._facility

    @property
    def available_capacity_usd(self) -> float:
        return self._facility.available_capacity_usd

    def accrue_hourly(self) -> float:
        """Accrues 1 hour of borrowing cost (APR / 8760) and undrawn commitment fees.

        Returns total cost accrued in USD for this hour.
        """
        hourly_dt = 1.0 / 8760.0
        interest_cost = self._facility.utilized_amount_usd * self._facility.apr * hourly_dt
        undrawn_balance = self.available_capacity_usd
        commitment_cost = undrawn_balance * self._facility.fee_rate * hourly_dt

        self.accrued_interest_usd += interest_cost
        self.accrued_commitment_fee_usd += commitment_cost
        return interest_cost + commitment_cost

    def draw(self, amount_usd: float) -> float:
        """Drawdown from credit facility.

        Returns actual amount drawn in USD.
        """
        if not self._facility.is_available or amount_usd <= 0.0:
            return 0.0

        max_possible = min(amount_usd, self.available_capacity_usd)
        if self._facility.max_draw_usd > 0.0:
            max_possible = min(max_possible, self._facility.max_draw_usd)

        if max_possible < self._facility.min_draw_usd:
            return 0.0

        new_utilized = self._facility.utilized_amount_usd + max_possible
        self._facility = CreditFacility(
            facility_id=self._facility.facility_id,
            is_available=self._facility.is_available,
            committed_limit_usd=self._facility.committed_limit_usd,
            utilized_amount_usd=new_utilized,
            apr=self._facility.apr,
            fee_rate=self._facility.fee_rate,
            min_draw_usd=self._facility.min_draw_usd,
            max_draw_usd=self._facility.max_draw_usd,
            permitted_draw_rules=self._facility.permitted_draw_rules,
        )
        return max_possible

    def repay(self, amount_usd: float) -> float:
        """Repay outstanding credit facility balance.

        Returns actual principal repaid in USD.
        """
        if amount_usd <= 0.0 or self._facility.utilized_amount_usd <= 0.0:
            return 0.0

        actual_repay = min(amount_usd, self._facility.utilized_amount_usd)
        new_utilized = self._facility.utilized_amount_usd - actual_repay

        self._facility = CreditFacility(
            facility_id=self._facility.facility_id,
            is_available=self._facility.is_available,
            committed_limit_usd=self._facility.committed_limit_usd,
            utilized_amount_usd=new_utilized,
            apr=self._facility.apr,
            fee_rate=self._facility.fee_rate,
            min_draw_usd=self._facility.min_draw_usd,
            max_draw_usd=self._facility.max_draw_usd,
            permitted_draw_rules=self._facility.permitted_draw_rules,
        )
        return actual_repay
