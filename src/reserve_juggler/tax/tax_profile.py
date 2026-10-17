"""
Tax profile and jurisdiction configuration models.

Supports progressive brackets, multi-jurisdiction rules, asset classifications,
holding period distinctions, and versioned audit tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class IncomeType(str, Enum):
    """Taxable income classification."""

    INTEREST = "interest"
    DIVIDEND = "dividend"
    CAPITAL_GAIN = "capital_gain"
    TAX_EXEMPT = "tax_exempt"


@dataclass(frozen=True)
class TaxBracket:
    """Progressive tax bracket."""

    lower_threshold: float
    upper_threshold: Optional[float]  # None indicates infinity
    rate: float  # Decimal rate (e.g. 0.21 for 21.0%)

    def __post_init__(self) -> None:
        if self.upper_threshold is not None and self.upper_threshold <= self.lower_threshold:
            raise ValueError(
                f"upper_threshold ({self.upper_threshold}) must exceed lower_threshold ({self.lower_threshold})"
            )
        if not (0.0 <= self.rate <= 1.0):
            raise ValueError(f"Tax rate must be between 0.0 and 1.0, got {self.rate}")


@dataclass(frozen=True)
class InstrumentTaxClassification:
    """Instrument-specific tax classification."""

    asset_id: str
    income_type: IncomeType
    is_qualified_dividend: bool = False
    is_exempt: bool = False
    notes: str = ""


@dataclass(frozen=True)
class TaxJurisdictionRules:
    """Versioned statutory rules for a legal jurisdiction and tax year."""

    jurisdiction_code: str  # e.g. 'US_FED_CORP', 'US_FED_INDIV', 'UK_CORP'
    tax_year: int
    rule_version: str  # e.g. '2026.1-US-CORP'
    ordinary_income_brackets: list[TaxBracket]
    qualified_dividend_brackets: list[TaxBracket]
    long_term_capital_gains_brackets: list[TaxBracket]
    long_term_holding_period_days: float = 365.0
    capital_loss_offset_limit: float = 3000.0  # Limit for offsetting against ordinary income
    financing_cost_deductible: bool = True
    transaction_fees_deductible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction_code": self.jurisdiction_code,
            "tax_year": self.tax_year,
            "rule_version": self.rule_version,
            "long_term_holding_period_days": self.long_term_holding_period_days,
            "capital_loss_offset_limit": self.capital_loss_offset_limit,
            "financing_cost_deductible": self.financing_cost_deductible,
            "transaction_fees_deductible": self.transaction_fees_deductible,
        }


@dataclass
class TaxProfile:
    """Tax profile for a specific corporate entity, account, or fund."""

    entity_id: str
    jurisdiction_rules: TaxJurisdictionRules
    accumulated_ordinary_income: float = 0.0
    accumulated_capital_gains: float = 0.0
    accumulated_capital_losses: float = 0.0
    instrument_classifications: dict[str, InstrumentTaxClassification] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_classification(self, asset_id: str) -> Optional[InstrumentTaxClassification]:
        """Look up tax classification for a given asset."""
        return self.instrument_classifications.get(asset_id)

    def register_classification(self, classification: InstrumentTaxClassification) -> None:
        """Register or update an asset tax classification."""
        self.instrument_classifications[classification.asset_id] = classification


def create_default_corporate_profile(entity_id: str = "TREASURY_CORP_01") -> TaxProfile:
    """Create a default 2026 corporate tax profile with progressive brackets."""
    rules = TaxJurisdictionRules(
        jurisdiction_code="US_FED_CORP",
        tax_year=2026,
        rule_version="2026.1-US-CORP",
        ordinary_income_brackets=[
            TaxBracket(lower_threshold=0.0, upper_threshold=50000.0, rate=0.15),
            TaxBracket(lower_threshold=50000.0, upper_threshold=200000.0, rate=0.21),
            TaxBracket(lower_threshold=200000.0, upper_threshold=None, rate=0.25),
        ],
        qualified_dividend_brackets=[
            TaxBracket(lower_threshold=0.0, upper_threshold=100000.0, rate=0.10),
            TaxBracket(lower_threshold=100000.0, upper_threshold=None, rate=0.15),
        ],
        long_term_capital_gains_brackets=[
            TaxBracket(lower_threshold=0.0, upper_threshold=100000.0, rate=0.15),
            TaxBracket(lower_threshold=100000.0, upper_threshold=None, rate=0.20),
        ],
        long_term_holding_period_days=365.0,
        capital_loss_offset_limit=10000.0,
        financing_cost_deductible=True,
        transaction_fees_deductible=True,
    )
    return TaxProfile(entity_id=entity_id, jurisdiction_rules=rules)
