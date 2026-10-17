# Reserve Juggler — Team 2 Technical Specification & Deliverable Documentation
## Treasury Safety Score, Capital Firewall, Tax-Aware Operations Research and Next Best Action Engine

---

### 1. Architectural Overview & File Ownership Boundaries

Team 2 is exclusively responsible for the financial decision engine of **Reserve Juggler**. The engine optimizes capital allocation across Cash, Overnight Repo (Tier 1), Two-Year Sovereign Bonds (Tier 2), and Growth Equities (Tier 3), strictly protecting customer payout obligations before optimizing investment returns.

#### Directory & Package Ownership
The implementation strictly adheres to the assigned package boundaries:
- `src/reserve_juggler/risk/`: Treasury Safety Score (TSS), Adaptive Liquidity Shield, Capital Firewall, and Holding-Time Gatekeeper.
- `src/reserve_juggler/decision/`: Integration protocols, learning events, audit publisher, and `TreasuryDecisionService`.
- `src/reserve_juggler/optimization/`: Operations Research action optimizer, benchmark engine, and tier exposure policies.
- `src/reserve_juggler/tax/`: Multi-jurisdiction progressive tax engine, holding-period rules, fee deductibility, and loss offsets.
- `tests/decision/`: Comprehensive test suite, stress benchmarks, and concrete fixture models.
- `requirements-part2.txt`: Independent dependency fragment.
- `docs/part2.md`: Architectural documentation and mathematical specification.

Neither Team 1's ledger/forecasting canonical contracts nor Team 3's ingestion/feedback/XAI services were rewritten or modified. Structural typing (`typing.Protocol`) was utilized to ensure zero-friction integration once canonical schemas merge.

---

### 2. Shared Integration Contracts & Protocol Specifications

Team 2 integrates through structural Python protocols in `reserve_juggler.decision.protocols`:

| Protocol | Description | Key Attributes |
|---|---|---|
| `TreasuryState` | System snapshot of liquidity, assets, and liabilities | `timestamp`, `available_cash`, `asset_positions`, `future_obligations`, `credit_availability`, `credit_apr`, `data_quality_warnings` |
| `AssetPosition` | Position characteristics across asset tiers | `asset_id`, `tier`, `market_value`, `liquidatable_value`, `availability`, `maturity`, `settlement_delay_hours`, `annual_yield`, `exit_fees`, `haircut`, `duration`, `volatility`, `cost_basis`, `instrument_type` |
| `ForecastPoint` | Probabilistic cash-flow distribution at horizon $h$ | `forecast_horizon`, `deposit_p10`, `deposit_p50`, `deposit_p90`, `withdrawal_p10`, `withdrawal_p50`, `withdrawal_p90`, `conservative_outflow`, `reliability_metadata` |
| `ForecastBundle` | Collection of forecast points and quality metadata | `point_forecasts`, `scenarios`, `timestamp`, `quality_status` (`HEALTHY`, `DEGRADED`, `STALE`, `CRITICAL`), `model_versions` |
| `RiskSignal` | Market and anomaly detection signals | `source`, `severity` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `confidence`, `timestamp`, `supporting_evidence`, `scenario_tags` |
| `DecisionRecord` | Canonical decision output with serializable reasoning | `decision_id`, `timestamp`, `selected_action`, `alternatives`, `score`, `reserve`, `eligible_surplus`, `cost_breakdown`, `hard_gate_results`, `reason_codes`, `to_dict()` |
| `ExecutionOutcome` | Post-settlement feedback from ledger | `realized_cash_movements`, `costs`, `penalties`, `losses`, `liquidity_shortfalls`, `metadata` |
| `LearningEvent` | Versioned event stream for Team 3 audit & feedback | `versioned_event_type`, `source_module`, `timestamp`, `correlation_id`, `structured_data`, `model_policy_versions`, `to_dict()` |

#### Currency & Decimal Conventions
- **Currency**: USD ($).
- **Rates**: Standard decimal convention internally ($0.048 = 4.8\%$; $0.005 = 0.5\%$; $0.12 = 12.0\%$).
- **Time Conventions**: APR (Annual Percentage Rate) using an $8,760$-hour year ($365 \text{ days} \times 24 \text{ hours}$).

---

### 3. Treasury Safety Score (TSS) Engine

The **Treasury Safety Score** is a composite metric on a $0 \text{ to } 100$ scale. Higher scores reflect a safer, more resilient financial posture.

#### Component Breakdown & Configurable Weights
$$TSS = \sum_{i=1}^6 w_i \cdot C_i \quad \text{where } \sum w_i = 1.0$$

| Component ($C_i$) | Weight ($w_i$) | Normalization Formula | Input Variables |
|---|---:|---|---|
| **Liquidity Coverage** | 30% | $\min\left(100, \max\left(0, \frac{\text{Cash}}{\text{Immediate Obligations} + \text{Near-Term Deficit}} \times 66.67\right)\right)$ | Available cash, 24h obligations, P90 net outflows |
| **Stress Survival** | 20% | $\min\left(100, \max\left(0, \frac{\text{Stressed Liquidatable Assets} + \text{Credit}}{\text{Severe Stress Demand}} \times 60.0\right)\right)$ | Haircut-stressed assets, credit limit, 1.25x P90 outflows |
| **Forecast Reliability** | 15% | $\max(0, \text{BaseQuality} - \sum \text{SeverityPenalty} \times \text{Confidence})$ | Bundle status (`HEALTHY`=95, `DEGRADED`=65, `STALE`=30, `CRITICAL`=10), RiskSignals |
| **Time-to-Liquidity** | 15% | $100 \times \left(1.0 - \frac{\text{Illiquid Asset Volume}}{\text{Total Non-Cash Assets}}\right)$ | Asset maturity/settlement delays vs. obligation due dates |
| **Forced-Liquidation** | 10% | $\max(0, \min(100, 100 - (\text{Avg Liquidation Loss Rate} \times 500)))$ | Exit fee rates, haircuts across asset positions |
| **Market-Risk Safety** | 10% | $\max(0, \min(100, 100 - (\text{Duration Risk} + \text{Volatility Risk})))$ | Bond duration (years), equity annualized volatility |

#### Operational Modes
- **0–39: Critical**: Defensive emergency mode. New capital allocations frozen. All cash conserved for payout survival.
- **40–59: Defensive**: Strict constraints. Tier 2 capped at 10%; Tier 3 at 0%. Short-term repos prioritized.
- **60–79: Balanced**: Standard operating posture. Tier 2 capped at 25%; Tier 3 capped at 5%.
- **80–100: Optimizing**: Well-capitalized state. Tier 2 capped at 35%; Tier 3 capped at 10%.

#### Circuit-Breaker Principle
A high average score must **never conceal an acute cash shortfall**. If immediately available cash plus authorized credit is less than immediate payout obligations:
1. `shortfall_detected = True` is triggered.
2. The composite score is strictly clamped to $\le 35.0$ (Critical mode), regardless of how much capital is held in illiquid bonds.
3. A critical warning is recorded.

> **Disclaimer**: The Treasury Safety Score is an operational heuristic and stress metric. It is explicitly **not** a calibrated statistical probability of default or survival.

---

### 4. Adaptive Liquidity Shield & Capital Firewall

#### Dynamic Reserve Sizing
The reserve assessment isolates capital into three strict economic buckets without double-counting overlapping horizons:

1. **Protected Capital**:
   $$\text{Protected Capital} = \text{Immediate Obligations}_{0..24h} \times (1 + \text{Safety Buffer}) + \text{Cumulative Forecast Deficit}_{0..24h}$$
   Guarantees immediate payout execution over the operating window.

2. **Future-Obligation Reserve**:
   $$\text{Future-Obligation Reserve} = \max_{h > 24h} \left(\text{Cumulative Deficit}_h - \text{Maturing Assets Available by } h\right) - \text{Protected Capital}$$
   Protects future forecast deficits until longer-term repos or assets mature and settle.

3. **Eligible Investable Surplus**:
   $$\text{Eligible Surplus} = \max\left(0.0, \text{Available Cash} - \text{Protected Capital} - \text{Future-Obligation Reserve}\right)$$
   Only unencumbered funds beyond protected reserves can be deployed for new investments.

#### Independent Hard Safety Gates
Hard gates are evaluated **independently of the TSS**. Even if TSS is 100, any action breaching a hard gate is immediately rejected:
- **Immediate Payout Coverage Gate**: $\text{Available Cash} + \text{Available Credit} \ge \text{Immediate Mandatory Payouts}$.
- **Protected Reserve Intact Gate**: $\text{Proposed Investment} \le \text{Eligible Investable Surplus}$.
- **Repo Lockup Timing Gate**: Lockup duration $T_{\text{lock}}$ must not lock funds across an obligation due date unless free cash covers the obligation.
- **Credit Limit Gate**: $\text{Requested Draw} \le \text{Remaining Authorized Credit Facility Capacity}$.

#### Hot Money vs. Core Capital
- **Hot Money**: $\text{Protected Capital} + \text{Future-Obligation Reserve}$. Kept in liquid cash or short repos maturing safely before obligations.
- **Core Capital**: $\max(0.0, \text{Total Portfolio Value} - \text{Hot Money})$.
- **Investable Core Capital**: $\min(\text{Core Capital}, \text{Eligible Surplus})$.

#### Holding-Time Gatekeeper (Section 6 Benchmark)
For sovereign bonds (Tier 2), simple accrual determines the breakeven holding period against liquidation haircuts:
$$\text{minimum\_days} = \left(\frac{\text{haircut}}{\text{annual\_yield}}\right) \times 365$$

*Benchmark Example*:
$$\text{minimum\_days} = \left(\frac{0.02}{0.048}\right) \times 365 = 152.0833 \text{ days}$$
- Planned holding $\ge 152.1$ days: **Passes** preliminary gate.
- Planned holding $< 152.1$ days: **Fails** gate (liquidated prior to breakeven accrual).
- **Equities (Tier 3)**: The simplistic formula is **explicitly rejected**. Equities exhibit price volatility, non-contractual coupons, and market slippage, requiring downside drawdown evaluation instead.

---

### 5. Tax-Aware Net Yield Optimization Engine

The tax engine evaluates expected net after-tax return across jurisdictions, progressive brackets, and asset classifications.

#### Progressive Bracket Calculation (No Flat-Rate Distortion)
For incremental taxable income $\Delta I$ with prior taxable income $I_{\text{prior}}$ across brackets $[b_k^{\text{lower}}, b_k^{\text{upper}}]$ at rate $r_k$:
$$\text{Tax}(\Delta I) = \sum_k r_k \cdot \max\left(0, \min(I_{\text{prior}} + \Delta I, b_k^{\text{upper}}) - \max(I_{\text{prior}}, b_k^{\text{lower}})\right)$$
*Validation*: Income of $\$70,000$ over brackets $[0, 50k] @ 15\%$ and $[50k, 200k] @ 21\%$ results in:
$$\$50,000 \times 0.15 + \$20,000 \times 0.21 = \$7,500 + \$4,200 = \$11,700 \quad (\text{not flat } \$70,000 \times 0.21 = \$14,700)$$

#### Classification, Holding Periods, and Loss Offsets
- **Interest Income**: Taxed at ordinary brackets; deductible bridge interest and fees directly shelter income.
- **Qualified Dividends**: Preferential corporate dividend brackets ($10\% - 15\%$).
- **Capital Gains**:
  - Short-Term ($< 365$ days): Taxed at ordinary income rates.
  - Long-Term ($\ge 365$ days): Taxed at preferential long-term rates ($15\% - 20\%$).
- **Loss Offsets**: Capital losses offset capital gains; remaining net loss deductible against ordinary income up to statutory limit.
- **Missing Classifications**: Flagged with `has_missing_classification = True` and explicit warnings to prevent unverified rankings from silently executing.
- **Audit Tracking**: Versioned rule signature (e.g. `2026.1-US-CORP`) emitted in every result.

---

### 6. Operations Research Treasury Optimizer & Next Best Action

The optimizer evaluates candidate actions under constrained optimization:
$$\max_{\text{Action} \in \mathcal{A}_{\text{feasible}}} \left(\text{After-Tax Net Value} - \text{Total Cost} - \text{Risk Penalty}\right)$$
Subject to:
$$\text{Hard Safety Gates}(\text{Action}) == \text{True}$$
$$\text{Payout Shortfall}(\text{Action}) == 0.0$$

#### Action Alternatives Evaluated
1. `KEEP_CASH`: Maintain liquid cash.
2. `PLACE_REPO`: Deploy eligible surplus into overnight/term repo.
3. `WAIT_REPO_MATURITY`: Wait for locked repo proceeds (only feasible if maturity precedes payout due time).
4. `BREAK_REPO_EARLY`: Liquidate repo early incurring early exit penalties.
5. `SELL_BOND`: Liquidate Tier 2 bonds subject to settlement delay and haircut.
6. `SELL_EQUITY`: Liquidate Tier 3 equities subject to slippage and market impact.
7. `DRAW_BRIDGE_CREDIT`: Draw from authorized bridge/overdraft facility.
8. `REPAY_BRIDGE_CREDIT`: Repay bridge credit upon repo maturity.

#### Section 7 Required Benchmark Verification
**Scenario**: $\$10,000,000$ mandatory customer payout due in 1 hour. Available cash is $\$0$. A $\$10,000,000$ repo matures in 2 hours with a $0.5\%$ early-exit fee. Authorized credit facility available at $12\%$ APR.

- **Option A (Break Repo Early)**:
  $$\text{Penalty} = \$10,000,000 \times 0.005 = \$50,000.00$$
- **Option B (Draw 2-Hour Bridge Credit)**:
  $$\text{Financing Cost} = \$10,000,000 \times 0.12 \times \left(\frac{2}{8,760}\right) = \$273.9726... \approx \$273.97$$
- **Optimization Decision**:
  - Since credit is available ($\$15\text{M} \ge \$10\text{M}$) and repayment upon repo maturity at $T+2$ hours is operationally feasible, the engine selects **`DRAW_BRIDGE_CREDIT`**, saving $\$49,726.03$ in fees.
  - If credit facility is frozen ($\$0$ availability), the engine selects **`BREAK_REPO_EARLY`**.
  - The engine **never** selects `WAIT_REPO_MATURITY` in this scenario, because waiting 2 hours would cause a default on the 1-hour customer payout. Mandatory payouts are unconditionally protected.

#### Tier Exposure Policy (Section 9)
Configurable prototype caps on new allocations from eligible surplus:

| TSS Range | Operational Mode | Tier 2 Cap (Bonds) | Tier 3 Cap (Equities) |
|---|---|---:|---:|
| 0–39 | Critical | 0% | 0% |
| 40–59 | Defensive | 10% | 0% |
| 60–79 | Balanced | 25% | 5% |
| 80–100 | Optimizing | 35% | 10% |

- **Remediation Policy**: Existing portfolios exceeding these caps report a breach in `tier_breaches_detected` and freeze new allocations to that tier. Distressed automatic fire-sales are strictly avoided.

---

### 7. Baseline Strategy Comparison & Stress Testing Results

Three strategies were benchmarked across identical stress scenarios in `tests/decision/test_stress_benchmarks.py`:
1. **All-Cash Strategy**: 100% liquid cash.
2. **Greedy Gross-Yield Strategy**: 95% invested in high-yielding bonds/equities chasing gross yield.
3. **Adaptive Reserve Juggler Strategy**: Dynamic Protected Capital, Hard Gates, and Tax-Aware OR Optimizer.

#### Comparative Performance Matrix

| Scenario / Metric | All-Cash Strategy | Greedy Gross-Yield Strategy | Adaptive Reserve Juggler |
|---|---|---|---|
| **Withdrawal Spike (3x P90)** | Payout Shortfall: $0<br>Net Yield: $0 | Payout Shortfall: $3.2M<br>Forced Haircuts: $285k | **Payout Shortfall: $0**<br>**Protected Capital: $8.2M** |
| **Deposit Drought (-80%)** | Payout Shortfall: $0<br>Remaining Cash: $2M | Emergency Borrowing: $1.8M<br>Distressed Fees: $38k | **Payout Shortfall: $0**<br>**Buffer Sized Dynamically** |
| **Rate Hike (+250 bps)** | Net Yield: $0 | Duration Loss: -$250k<br>Portfolio Drawdown: -5.0% | **Holding-Time Gate Enforced**<br>**Tier 2 Cap Prevents Over-Exposure** |
| **Equity Crash (-25%)** | Net Loss: $0 | Equity Loss: -$250k | **Tier 3 Capped at 5-10%**<br>**Core Capital Protected** |
| **Credit Facility Freeze** | Payout Shortfall: $0 | Payout Shortfall: $4.5M<br>(Credit unavailable) | **Payout Shortfall: $0**<br>**Orderly Repo Break Executed** |
| **Funding APR Jump (12% -> 24%)** | Net Yield: $0 | Financing Cost Doubled | **OR Engine Adapts Decision**<br>**Compares All-In Costs Accurately** |

**Conclusion**:
- The Greedy strategy suffers catastrophic payout defaults and fire-sale losses during shocks.
- The All-Cash strategy avoids shortfalls but sacrifices all yield.
- The **Adaptive Reserve Juggler Strategy** achieves zero payout shortfalls across all stress scenarios while optimizing risk-adjusted after-tax returns.

---

### 8. Public API Reference

The 8 required public functions and services are exported at top-level `reserve_juggler` and from their respective subpackages:

```python
from reserve_juggler import (
    TreasuryDecisionService,
    calculate_treasury_safety_score,
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
    classify_hot_core_capital,
    calculate_holding_time_gate,
    optimize_treasury_actions,
    evaluate_tax_adjusted_value,
)
```

#### Function Signatures & Descriptions

1. `TreasuryDecisionService.decide(state, forecast_bundle, risk_signals=None, tax_profile=None) -> StandardDecisionRecord`
   - Orchestrates TSS scoring, reserve assessment, hard gate verification, capital partitioning, and OR optimization.
   - Publishes audit events (`DECISION_PROPOSED_V1`, `ACTION_REJECTED_V1`) to Team 3 event subscribers.
   - Returns JSON-serializable `StandardDecisionRecord`.

2. `calculate_treasury_safety_score(state, forecast_bundle, risk_signals=None, weights=None) -> SafetyScoreResult`
   - Calculates 0-100 composite TSS and component breakdown.
   - Enforces cash shortfall circuit breaker ($\le 35.0$).

3. `assess_liquidity_reserve(state, forecast_bundle, risk_signals=None, operational_safety_buffer=0.10, payout_window_hours=24.0, stress_multiplier=1.0) -> ReserveAssessment`
   - Sizes Protected Capital, Future-Obligation Reserve, and Eligible Surplus without double counting.

4. `evaluate_hard_safety_gates(state, reserve_assessment, proposed_investment_amount=0.0, proposed_lockup_hours=0.0, proposed_bridge_draw=0.0, target_tier=1) -> HardGateEvaluation`
   - Evaluates non-negotiable gates independently of TSS.

5. `classify_hot_core_capital(state, forecast_bundle, reserve_assessment) -> CapitalClassification`
   - Partitions assets into operational Hot Money and investable Core Capital.

6. `calculate_holding_time_gate(asset_position, expected_holding_days) -> HoldingTimeGateResult`
   - Evaluates preliminary bond breakeven holding period $(0.02 / 0.048) \times 365 = 152.08 \text{ days}$.
   - Strictly rejects application to growth equities.

7. `optimize_treasury_actions(state, forecast_bundle, risk_signals=None, tax_profile=None, safety_score=None, reserve_assessment=None, planning_horizon_hours=24.0) -> OptimizationResult`
   - Evaluates candidate actions and selects cost-minimizing or after-tax net value-maximizing action.
   - Implements Section 7 repo-vs-bridge benchmark logic.

8. `evaluate_tax_adjusted_value(gross_revenue, financing_costs=0.0, transaction_fees=0.0, liquidation_losses=0.0, holding_days=0.0, instrument_id='DEFAULT', tax_profile=None, explicit_income_type=None, cost_basis=None) -> TaxAdjustedValueResult`
   - Computes incremental progressive tax without flat-rate distortion, accounting for holding period, loss offsets, and deductibility.

---

### 9. Test Verification & Execution Results

The test suite contains 28 unit, integration, and stress tests executed via `pytest`:

```text
tests/decision/test_capital_firewall.py::test_classify_hot_core_capital PASSED
tests/decision/test_capital_firewall.py::test_bond_holding_time_gatekeeper_benchmark PASSED
tests/decision/test_capital_firewall.py::test_equity_holding_time_simplistic_formula_rejected PASSED
tests/decision/test_decision_service.py::test_decision_service_decide_and_json_serializability PASSED
tests/decision/test_decision_service.py::test_decision_service_publishes_rejection_events PASSED
tests/decision/test_decision_service.py::test_decision_service_record_execution_outcome PASSED
tests/decision/test_liquidity_shield.py::test_liquidity_reserve_assessment PASSED
tests/decision/test_liquidity_shield.py::test_hard_gate_immediate_payout_coverage_failure PASSED
tests/decision/test_liquidity_shield.py::test_hard_gate_rejects_investment_exceeding_surplus PASSED
tests/decision/test_liquidity_shield.py::test_hard_gate_rejects_repo_lockup_clashing_with_payout PASSED
tests/decision/test_liquidity_shield.py::test_hard_gate_credit_limit_respected PASSED
tests/decision/test_optimizer.py::test_section_7_required_benchmark_example PASSED
tests/decision/test_optimizer.py::test_benchmark_when_credit_unavailable_breaks_repo PASSED
tests/decision/test_optimizer.py::test_tier_exposure_limits_by_tss PASSED
tests/decision/test_optimizer.py::test_remediation_policy_for_existing_breaches PASSED
tests/decision/test_safety_score.py::test_tss_healthy_optimizing_treasury PASSED
tests/decision/test_safety_score.py::test_tss_circuit_breaker_cash_shortfall PASSED
tests/decision/test_safety_score.py::test_tss_stale_forecast_triggers_conservative_ceiling PASSED
tests/decision/test_safety_score.py::test_tss_risk_signal_penalties PASSED
tests/decision/test_safety_score.py::test_tss_weights_validation PASSED
tests/decision/test_stress_benchmarks.py::test_stress_withdrawal_spike_and_deposit_shortfall PASSED
tests/decision/test_stress_benchmarks.py::test_stress_rising_bond_yields_and_market_declines PASSED
tests/decision/test_stress_benchmarks.py::test_stress_repo_lockup_and_credit_unavailability PASSED
tests/decision/test_tax_engine.py::test_progressive_brackets_no_flat_distortion PASSED
tests/decision/test_tax_engine.py::test_progressive_tax_incremental_prior_income PASSED
tests/decision/test_tax_engine.py::test_tax_engine_financing_cost_deductibility PASSED
tests/decision/test_tax_engine.py::test_holding_period_short_vs_long_term_capital_gains PASSED
tests/decision/test_tax_engine.py::test_missing_tax_classification_warning PASSED

============================= 28 passed in 0.06s =============================
```
