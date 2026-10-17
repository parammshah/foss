"""
Tests for Tax-Aware Net Yield Optimization and progressive taxation.
"""

import pytest

from reserve_juggler.tax.tax_engine import (
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


def test_progressive_brackets_no_flat_distortion():
    """Verify progressive bracket calculation without flat-rate distortion.

    Brackets:
    0 - $50,000: 15%
    $50,000 - $200,000: 21%
    $200,000+: 25%

    For $70,000 income:
    Bracket 1 ($0 to $50k): $50,000 * 0.15 = $7,500
    Bracket 2 ($50k to $70k): $20,000 * 0.21 = $4,200
    Total tax = $11,700.
    A flat 21% calculation would incorrectly produce $14,700.
    """
    brackets = [
        TaxBracket(lower_threshold=0.0, upper_threshold=50000.0, rate=0.15),
        TaxBracket(lower_threshold=50000.0, upper_threshold=200000.0, rate=0.21),
        TaxBracket(lower_threshold=200000.0, upper_threshold=None, rate=0.25),
    ]

    total_tax, marginal_rate = calculate_progressive_tax(
        taxable_amount=70000.0, brackets=brackets, prior_taxable_income=0.0
    )

    assert round(total_tax, 2) == 11700.0
    assert marginal_rate == 0.21


def test_progressive_tax_incremental_prior_income():
    """Verify marginal taxation on incremental income given existing prior taxable income."""
    brackets = [
        TaxBracket(lower_threshold=0.0, upper_threshold=50000.0, rate=0.15),
        TaxBracket(lower_threshold=50000.0, upper_threshold=200000.0, rate=0.21),
        TaxBracket(lower_threshold=200000.0, upper_threshold=None, rate=0.25),
    ]

    # Prior income already filled bracket 1 ($50k). An incremental $20k is taxed entirely in bracket 2 at 21%
    total_tax, marginal_rate = calculate_progressive_tax(
        taxable_amount=20000.0, brackets=brackets, prior_taxable_income=50000.0
    )

    assert round(total_tax, 2) == 4200.0
    assert marginal_rate == 0.21


def test_tax_engine_financing_cost_deductibility():
    """Verify bridge financing costs reduce taxable interest income."""
    profile = create_default_corporate_profile()

    # Gross interest $100k, bridge financing cost $10k
    result = evaluate_tax_adjusted_value(
        gross_revenue=100000.0,
        financing_costs=10000.0,
        transaction_fees=0.0,
        tax_profile=profile,
        explicit_income_type=IncomeType.INTEREST,
    )

    # Net taxable interest = $90,000
    # First $50k @ 15% = $7,500; Next $40k @ 21% = $8,400; Total tax = $15,900
    assert round(result.tax_liability, 2) == 15900.0
    assert round(result.net_pre_tax_value, 2) == 90000.0
    assert round(result.after_tax_net_value, 2) == 74100.0
    assert result.tax_shield > 0.0


def test_holding_period_short_vs_long_term_capital_gains():
    """Verify long-term holding period qualifies for lower capital gains rate."""
    profile = create_default_corporate_profile()

    # Short-term (<365 days): 180 days -> taxed at ordinary rates (15% on first $50k, 21% on next $50k)
    st_result = evaluate_tax_adjusted_value(
        gross_revenue=100000.0,
        cost_basis=0.0,
        holding_days=180.0,
        tax_profile=profile,
        explicit_income_type=IncomeType.CAPITAL_GAIN,
    )

    # Long-term (>=365 days): 400 days -> taxed at preferential LTCG brackets (15% flat up to $100k)
    lt_result = evaluate_tax_adjusted_value(
        gross_revenue=100000.0,
        cost_basis=0.0,
        holding_days=400.0,
        tax_profile=profile,
        explicit_income_type=IncomeType.CAPITAL_GAIN,
    )

    # Long-term capital gains tax should be strictly lower than short-term ordinary tax
    assert lt_result.tax_liability < st_result.tax_liability
    assert lt_result.after_tax_net_value > st_result.after_tax_net_value


def test_missing_tax_classification_warning():
    """Verify missing tax classifications are flagged and do not silently rank."""
    profile = create_default_corporate_profile()

    result = evaluate_tax_adjusted_value(
        gross_revenue=50000.0,
        instrument_id="UNCLASSIFIED_EXOTIC_DERIVATIVE",
        tax_profile=profile,
    )

    assert result.has_missing_classification is True
    assert any("Missing tax classification" in w for w in result.warnings)
