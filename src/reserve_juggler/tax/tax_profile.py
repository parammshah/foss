"""
Tax profile and jurisdiction configuration models.

Supports progressive brackets, statutory corporate rules (IRS Form 1120),
synthetic demo configurations, asset classifications, and audit tracking.
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
    """Versioned statutory or synthetic rules for a legal jurisdiction and tax year."""

    jurisdiction_code: str  # e.g. 'US_FED_CORP_STATUTORY', 'SYNTHETIC_DEMO'
    tax_year: int
    rule_version: str  # e.g. '2026.1-US-CORP-STATUTORY'
    ordinary_income_brackets: list[TaxBracket]
    qualified_dividend_brackets: list[TaxBracket]
    long_term_capital_gains_brackets: list[TaxBracket]
    long_term_holding_period_days: float = 365.0
    capital_loss_offset_limit: float = 0.0  # Limit for offsetting against ordinary income (0 for US corps)
    financing_cost_deductible: bool = True
    transaction_fees_deductible: bool = True
    is_synthetic: bool = False
    statutory_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction_code": self.jurisdiction_code,
            "tax_year": self.tax_year,
            "rule_version": self.rule_version,
            "long_term_holding_period_days": self.long_term_holding_period_days,
            "capital_loss_offset_limit": self.capital_loss_offset_limit,
            "financing_cost_deductible": self.financing_cost_deductible,
            "transaction_fees_deductible": self.transaction_fees_deductible,
            "is_synthetic": self.is_synthetic,
            "statutory_citations": list(self.statutory_citations),
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

    def validate(self) -> list[str]:
        """Validate completeness, bracket order, rate bounds, and accounting invariants.

        Returns an empty list if valid, or a list of descriptive validation errors.
        """
        errors: list[str] = []
        if not self.entity_id or not self.entity_id.strip():
            errors.append("TaxProfile entity_id must not be empty.")
        rules = self.jurisdiction_rules
        if not rules:
            errors.append("TaxProfile jurisdiction_rules must not be None.")
            return errors
        if not rules.jurisdiction_code or not rules.jurisdiction_code.strip():
            errors.append("TaxJurisdictionRules jurisdiction_code must not be empty.")
        if not (1900 <= rules.tax_year <= 2100):
            errors.append(f"TaxJurisdictionRules tax_year ({rules.tax_year}) outside valid range [1900, 2100].")
        if not rules.ordinary_income_brackets:
            errors.append("TaxJurisdictionRules ordinary_income_brackets must not be empty.")
        else:
            for idx, b in enumerate(rules.ordinary_income_brackets):
                if not (0.0 <= b.rate <= 1.0):
                    errors.append(f"Ordinary income bracket {idx} rate ({b.rate}) outside valid range [0.0, 1.0].")
                if b.upper_threshold is not None and b.upper_threshold <= b.lower_threshold:
                    errors.append(f"Ordinary income bracket {idx} upper_threshold ({b.upper_threshold}) <= lower_threshold ({b.lower_threshold}).")
        if self.accumulated_ordinary_income < 0.0:
            errors.append(f"accumulated_ordinary_income ({self.accumulated_ordinary_income}) cannot be negative.")
        if self.accumulated_capital_gains < 0.0:
            errors.append(f"accumulated_capital_gains ({self.accumulated_capital_gains}) cannot be negative.")
        if self.accumulated_capital_losses < 0.0:
            errors.append(f"accumulated_capital_losses ({self.accumulated_capital_losses}) cannot be negative.")
        return errors

    def is_applicable_to(
        self, entity_id: Optional[str] = None, jurisdiction: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Verify that the tax profile matches the required corporate entity and legal jurisdiction.

        Statutory rules are not universally applicable to all legal entities or foreign jurisdictions.
        """
        if entity_id and self.entity_id and entity_id.strip().upper() != self.entity_id.strip().upper():
            return (
                False,
                f"Entity mismatch: Profile entity '{self.entity_id}' does not match target entity '{entity_id}'.",
            )
        if jurisdiction and self.jurisdiction_rules.jurisdiction_code:
            if jurisdiction.strip().upper() != self.jurisdiction_rules.jurisdiction_code.strip().upper():
                return (
                    False,
                    f"Jurisdiction mismatch: Profile rules '{self.jurisdiction_rules.jurisdiction_code}' "
                    f"do not match target jurisdiction '{jurisdiction}'. Statutory rules are jurisdiction-specific.",
                )
        return True, None


def create_statutory_us_corp_profile(
    entity_id: str = "US_CORP_01", tax_year: int = 2026
) -> TaxProfile:
    """Create a validated US Federal Corporate Tax Profile based on IRS Form 1120 statutory rules.

    Statutory Basis:
    - IRC § 11: Flat 21.0% corporate income tax rate.
    - IRC § 1211(a): Corporations may deduct capital losses ONLY to the extent of capital gains.
      No ordinary income deduction permitted ($0 offset limit). Net capital losses carry
      back 3 years and forward 5 years (IRC § 1212(a)).
    - IRC § 1222: Corporate net capital gains are taxed at the same flat 21.0% rate as ordinary income.
    - IRC § 243: Dividends-Received Deduction (DRD). Assuming <20% ownership, 50% DRD results in
      an effective federal tax rate of 10.5% (21% * 50%).
    - IRC § 163(j): Business interest expense deduction limits apply to large corporations.
    """
    rules = TaxJurisdictionRules(
        jurisdiction_code="US_FED_CORP_STATUTORY",
        tax_year=tax_year,
        rule_version=f"{tax_year}.1-US-CORP-STATUTORY",
        ordinary_income_brackets=[
            TaxBracket(lower_threshold=0.0, upper_threshold=None, rate=0.21),
        ],
        qualified_dividend_brackets=[
            # 50% DRD under IRC § 243 results in 10.5% effective tax rate on dividends
            TaxBracket(lower_threshold=0.0, upper_threshold=None, rate=0.105),
        ],
        long_term_capital_gains_brackets=[
            # Corporations pay standard 21% on net capital gains (no preferential rate)
            TaxBracket(lower_threshold=0.0, upper_threshold=None, rate=0.21),
        ],
        long_term_holding_period_days=365.0,
        capital_loss_offset_limit=0.0,  # Strict IRC § 1211(a): $0 offset against ordinary income
        financing_cost_deductible=True,
        transaction_fees_deductible=True,
        is_synthetic=False,
        statutory_citations=[
            "IRC § 11 (21% corporate income tax rate)",
            "IRC § 1211(a) (capital losses allowed only to extent of capital gains)",
            "IRC § 243 (50% dividends-received deduction)",
            "IRS Form 1120 Instructions (Corporate Income Tax)",
        ],
    )
    profile = TaxProfile(entity_id=entity_id, jurisdiction_rules=rules)
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="OVERNIGHT_REPO",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="Overnight general collateral repo interest (IRC § 61(a)(4))",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="SOVEREIGN_2Y",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="US Treasury note coupon interest (federal taxable)",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="EQUITY_INDEX",
            income_type=IncomeType.DIVIDEND,
            is_qualified_dividend=True,
            notes="Corporate dividend distributions eligible for 50% DRD (IRC § 243)",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="T1_REPO_PORTFOLIO",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="Tier 1 repo portfolio interest income",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="T2_UST_2Y_PORTFOLIO",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="Tier 2 US Treasury benchmark bond interest income",
        )
    )
    return profile


def create_synthetic_demo_profile(entity_id: str = "SYNTHETIC_DEMO_01") -> TaxProfile:
    """Create a multi-tier progressive synthetic profile for testing non-linear progressive arithmetic.

    Clearly labeled as SYNTHETIC. Must not be confused with statutory corporate rules.
    """
    rules = TaxJurisdictionRules(
        jurisdiction_code="SYNTHETIC_DEMO_PROGRESSIVE",
        tax_year=2026,
        rule_version="2026.1-SYNTHETIC-DEMO",
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
        capital_loss_offset_limit=3000.0,
        financing_cost_deductible=True,
        transaction_fees_deductible=True,
        is_synthetic=True,
        statutory_citations=["SYNTHETIC DEMO ONLY - NOT A STATUTORY TAX JURISDICTION"],
    )
    profile = TaxProfile(entity_id=entity_id, jurisdiction_rules=rules)
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="OVERNIGHT_REPO",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="Synthetic repo demo interest",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="SOVEREIGN_2Y",
            income_type=IncomeType.INTEREST,
            is_exempt=False,
            notes="Synthetic bond demo interest",
        )
    )
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="EQUITY_INDEX",
            income_type=IncomeType.DIVIDEND,
            is_qualified_dividend=True,
            notes="Synthetic dividend demo",
        )
    )
    return profile


def create_default_corporate_profile(entity_id: str = "TREASURY_CORP_01") -> TaxProfile:
    """Backward-compatible default corporate profile (delegates to synthetic demo).

    NOTE: For operational and production decision cycles, configure create_statutory_us_corp_profile.
    """
    return create_synthetic_demo_profile(entity_id=entity_id)
