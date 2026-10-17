"""
Tax optimization, progressive brackets, and after-tax net yield calculation.
"""

from reserve_juggler.tax.tax_engine import (
    TaxAdjustedValueResult,
    calculate_progressive_tax,
    evaluate_tax_adjusted_value,
)
from reserve_juggler.tax.tax_profile import (
    IncomeType,
    InstrumentTaxClassification,
    TaxBracket,
    TaxJurisdictionRules,
    TaxProfile,
    create_default_corporate_profile,
)

__all__ = [
    "evaluate_tax_adjusted_value",
    "calculate_progressive_tax",
    "TaxAdjustedValueResult",
    "TaxProfile",
    "TaxBracket",
    "TaxJurisdictionRules",
    "InstrumentTaxClassification",
    "IncomeType",
    "create_default_corporate_profile",
]
