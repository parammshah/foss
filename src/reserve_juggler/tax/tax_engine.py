"""
Tax-Aware Operations Research and Net Yield Evaluation Engine.

Implements true progressive tax bracket evaluations, multi-jurisdiction rules,
holding period logic, fee deductibility, and loss offsets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from reserve_juggler.tax.tax_profile import (
    IncomeType,
    InstrumentTaxClassification,
    TaxBracket,
    TaxJurisdictionRules,
    TaxProfile,
    create_default_corporate_profile,
    create_statutory_us_corp_profile,
    create_synthetic_demo_profile,
)


@dataclass(frozen=True)
class TaxAdjustedValueResult:
    """Result of evaluate_tax_adjusted_value."""

    gross_revenue: float
    financing_costs: float
    transaction_fees: float
    liquidation_losses: float
    net_pre_tax_value: float
    tax_liability: float
    tax_shield: float
    after_tax_net_value: float
    effective_tax_rate: float
    marginal_tax_rate: float
    jurisdiction_version: str
    has_missing_classification: bool
    ranking_is_provisional: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross_revenue": round(self.gross_revenue, 4),
            "financing_costs": round(self.financing_costs, 4),
            "transaction_fees": round(self.transaction_fees, 4),
            "liquidation_losses": round(self.liquidation_losses, 4),
            "net_pre_tax_value": round(self.net_pre_tax_value, 4),
            "tax_liability": round(self.tax_liability, 4),
            "tax_shield": round(self.tax_shield, 4),
            "after_tax_net_value": round(self.after_tax_net_value, 4),
            "effective_tax_rate": round(self.effective_tax_rate, 4),
            "marginal_tax_rate": round(self.marginal_tax_rate, 4),
            "jurisdiction_version": self.jurisdiction_version,
            "has_missing_classification": self.has_missing_classification,
            "ranking_is_provisional": self.ranking_is_provisional,
            "warnings": list(self.warnings),
        }


def calculate_progressive_tax(
    taxable_amount: float,
    brackets: Sequence[TaxBracket],
    prior_taxable_income: float = 0.0,
) -> tuple[float, float]:
    """Calculate progressive tax on incremental income over progressive brackets.

    Does NOT treat a higher marginal bracket as taxing all prior income.
    Returns:
        (total_tax_on_increment, top_marginal_rate_applied)
    """
    if taxable_amount <= 0.0 or not brackets:
        top_rate = brackets[0].rate if brackets else 0.0
        return 0.0, top_rate

    start_income = max(0.0, prior_taxable_income)
    end_income = start_income + taxable_amount

    total_tax = 0.0
    highest_rate_applied = 0.0

    # Sort brackets by lower threshold
    sorted_brackets = sorted(brackets, key=lambda b: b.lower_threshold)

    for b in sorted_brackets:
        b_lower = b.lower_threshold
        b_upper = float("inf") if b.upper_threshold is None else b.upper_threshold

        # Determine overlap of [start_income, end_income] with [b_lower, b_upper]
        overlap_start = max(start_income, b_lower)
        overlap_end = min(end_income, b_upper)

        if overlap_end > overlap_start:
            taxable_slice = overlap_end - overlap_start
            slice_tax = taxable_slice * b.rate
            total_tax += slice_tax
            highest_rate_applied = max(highest_rate_applied, b.rate)

    return total_tax, highest_rate_applied


def evaluate_tax_adjusted_value(
    gross_revenue: float,
    financing_costs: float = 0.0,
    transaction_fees: float = 0.0,
    liquidation_losses: float = 0.0,
    holding_days: float = 0.0,
    instrument_id: str = "DEFAULT",
    tax_profile: Optional[TaxProfile] = None,
    explicit_income_type: Optional[IncomeType] = None,
    cost_basis: Optional[float] = None,
) -> TaxAdjustedValueResult:
    """Evaluate expected after-tax net value after financing costs, fees, and liquidity losses.

    Definitions of gross_revenue:
    - For INTEREST: gross_revenue is interest/coupon income earned.
    - For DIVIDEND: gross_revenue is dividend distribution received.
    - For CAPITAL_GAIN: gross_revenue is gross sale proceeds (or capital gain if cost_basis is None).
      When cost_basis is provided, capital gain/loss = gross_revenue - cost_basis.

    Accounts for:
    - Progressive marginal tax brackets (without flat-rate distortion).
    - Interest vs. dividend vs. capital gain classifications.
    - Short-term (<365 days) vs. long-term (>=365 days) capital gains holding rules.
    - Loss-offset mechanisms against existing gains and statutory limits.
    - Deductibility of bridge financing costs and transaction fees.
    - Missing classification flagging to prevent unverified rankings.
    """
    warnings: list[str] = []
    has_missing_classification = False
    ranking_is_provisional = False

    # Use default corporate profile if none provided, but flag warning
    if tax_profile is None:
        tax_profile = create_default_corporate_profile()
        warnings.append(
            "No TaxProfile provided; defaulted to synthetic demo profile. "
            "For production tax calculations, configure create_statutory_us_corp_profile."
        )
        ranking_is_provisional = True

    rules = tax_profile.jurisdiction_rules
    classification = tax_profile.get_classification(instrument_id)

    if classification is None and explicit_income_type is None:
        has_missing_classification = True
        ranking_is_provisional = True
        warnings.append(
            f"WARNING: Missing tax classification for asset '{instrument_id}'. "
            f"Assuming ordinary interest income; ranking must be verified by audit."
        )
        resolved_income_type = IncomeType.INTEREST
        is_exempt = False
        is_qualified = False
    elif explicit_income_type is not None:
        resolved_income_type = explicit_income_type
        if classification is not None:
            # Preserve existing classification attributes unless explicitly noted, and log auditable note
            is_exempt = classification.is_exempt
            is_qualified = classification.is_qualified_dividend
            warnings.append(
                f"Auditable Override: Instrument '{instrument_id}' income type overridden to {explicit_income_type.value} "
                f"while preserving classification attributes (exempt={is_exempt}, qualified={is_qualified})."
            )
        else:
            is_exempt = False
            is_qualified = False
    else:
        assert classification is not None
        resolved_income_type = classification.income_type
        is_exempt = classification.is_exempt
        is_qualified = classification.is_qualified_dividend

    # 1. Pre-tax Net Cash Value
    # Cash generated minus explicit out-of-pocket expenses
    net_pre_tax_value = gross_revenue - financing_costs - transaction_fees - liquidation_losses

    # 2. Tax Shields from Deductible Expenses
    deductible_expenses = 0.0
    if rules.financing_cost_deductible:
        deductible_expenses += financing_costs
    if rules.transaction_fees_deductible:
        deductible_expenses += transaction_fees

    tax_liability = 0.0
    tax_shield = 0.0
    marginal_rate = 0.0
    excess_shield = 0.0

    if is_exempt:
        # Fully tax-exempt instrument (e.g. municipal paper)
        tax_liability = 0.0
        tax_shield = 0.0
        marginal_rate = 0.0
        excess_shield = 0.0
    elif resolved_income_type == IncomeType.INTEREST:
        # Interest income: ordinary brackets
        # Deductible expenses reduce taxable interest income
        taxable_interest = max(0.0, gross_revenue - deductible_expenses)
        tax_liability, marginal_rate = calculate_progressive_tax(
            taxable_interest,
            rules.ordinary_income_brackets,
            prior_taxable_income=tax_profile.accumulated_ordinary_income,
        )
        # Tax shield on excess deduction against other income
        excess_deduction = max(0.0, deductible_expenses - gross_revenue)
        excess_shield, _ = calculate_progressive_tax(
            excess_deduction,
            rules.ordinary_income_brackets,
            prior_taxable_income=tax_profile.accumulated_ordinary_income,
        )
        tax_shield = deductible_expenses * marginal_rate

    elif resolved_income_type == IncomeType.DIVIDEND:
        # Dividend income: qualified brackets or ordinary brackets
        brackets = rules.qualified_dividend_brackets if is_qualified else rules.ordinary_income_brackets
        tax_liability, marginal_rate = calculate_progressive_tax(
            gross_revenue,
            brackets,
            prior_taxable_income=tax_profile.accumulated_ordinary_income,
        )
        excess_shield, _ = calculate_progressive_tax(
            deductible_expenses,
            rules.ordinary_income_brackets,
            prior_taxable_income=tax_profile.accumulated_ordinary_income,
        )
        tax_shield = excess_shield

    elif resolved_income_type == IncomeType.CAPITAL_GAIN:
        # Capital gains / losses
        gain_or_loss = gross_revenue - (cost_basis if cost_basis is not None else 0.0)
        net_capital_gain = gain_or_loss - transaction_fees - liquidation_losses

        is_long_term = holding_days >= rules.long_term_holding_period_days

        if net_capital_gain >= 0.0:
            accumulated_losses = tax_profile.accumulated_capital_losses
            offset_amount = min(net_capital_gain, accumulated_losses)
            taxable_gain = net_capital_gain - offset_amount

            brackets = (
                rules.long_term_capital_gains_brackets
                if is_long_term
                else rules.ordinary_income_brackets
            )
            tax_liability, marginal_rate = calculate_progressive_tax(
                taxable_gain,
                brackets,
                prior_taxable_income=tax_profile.accumulated_capital_gains,
            )
            excess_shield = 0.0
            tax_shield = offset_amount * marginal_rate
        else:
            tax_liability = 0.0
            loss_amount = abs(net_capital_gain)
            offset_against_gains = min(loss_amount, tax_profile.accumulated_capital_gains)
            offset_against_ordinary = min(
                loss_amount - offset_against_gains, rules.capital_loss_offset_limit
            )

            shield_gains, _ = calculate_progressive_tax(
                offset_against_gains,
                rules.long_term_capital_gains_brackets,
                prior_taxable_income=0.0,
            )
            shield_ord, marginal_rate = calculate_progressive_tax(
                offset_against_ordinary,
                rules.ordinary_income_brackets,
                prior_taxable_income=0.0,
            )
            tax_shield = shield_gains + shield_ord
            excess_shield = tax_shield

    # After-tax net value: net pre-tax cash minus taxes paid on this income, plus tax shield on excess deductions
    after_tax_net_value = net_pre_tax_value - tax_liability + excess_shield

    effective_tax_rate = (
        (tax_liability / gross_revenue) if gross_revenue > 0.0 else 0.0
    )

    return TaxAdjustedValueResult(
        gross_revenue=gross_revenue,
        financing_costs=financing_costs,
        transaction_fees=transaction_fees,
        liquidation_losses=liquidation_losses,
        net_pre_tax_value=net_pre_tax_value,
        tax_liability=tax_liability,
        tax_shield=tax_shield,
        after_tax_net_value=after_tax_net_value,
        effective_tax_rate=effective_tax_rate,
        marginal_tax_rate=marginal_rate,
        jurisdiction_version=rules.rule_version,
        has_missing_classification=has_missing_classification,
        ranking_is_provisional=ranking_is_provisional,
        warnings=warnings,
    )
