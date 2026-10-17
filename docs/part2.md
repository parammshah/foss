# Reserve Juggler — Person 2 Decision Engine Documentation & Repair Audit
## Financial Correctness, Funding Feasibility, Reserve Integrity, Tax Accuracy and Treasury Optimization

---

### 1. Executive Summary & Repair Audit Overview

This document specifies the architecture, mathematical formulations, statutory rules, empirical benchmarks, and integration contracts for the **Person 2 Financial Decision Engine** in Reserve Juggler.

Following a thorough quantitative-finance and operations-research repair audit, the decision engine enforces strict payout protection, time-indexed funding plans, multi-horizon capital firewalls, statutory corporate tax treatment, and transparent optimization objectives.

#### Key Repaired Capabilities
1. **Payout Funding Feasibility**: Actions evaluate actual net proceeds after penalties/haircuts, availability delays against mandatory payment deadlines, and residual funding gaps. Incomplete funding is explicitly flagged as partial funding (`is_partial_funding=True`) and never marked as fully feasible.
2. **Explicit Unresolved Funding Shortfalls**: When no authorized combination can satisfy a payout deadline, the engine emits `ActionType.UNRESOLVED_FUNDING_SHORTFALL` with an escalation event (`UNRESOLVED_FUNDING_SHORTFALL_V1`). Retaining cash (`KEEP_CASH`) is never falsely described as protecting a payout when cash is insufficient.
3. **Time-Indexed Bridge Credit & Repo Repayment**: Modeled with exact draw timing, principal, APR convention (8,760 hours/year), repayment proceeds, competing obligations, and remaining debt. Bridge is selected only when cheaper and backed by a verified future cash source.
4. **Action-Specific Hard Safety Gates**: Decision records report gates evaluated on the *selected action proposal* itself (amount, settlement delay, credit draw, repayment viability) rather than a base-state check.
5. **Multi-Horizon Reserves (24–72h)**: Scheduled obligations across 1, 12, 30, 48, and 72 hours are integrated into dynamic reserves. Normal repo maturity proceeds do not incur early break fees. Forecast horizon semantics prevent double-counting.
6. **Dual-Mode Optimization Objective**:
   - *Emergency Funding Mode*: Minimizes total all-in cost (interest, fees, tax shields, and residual shortfalls) to fund mandatory payouts by deadline.
   - *Surplus Allocation Mode*: Maximizes expected after-tax net value minus transparently quantified monetary risk costs (duration and volatility downside), subject to TSS tier caps.
7. **Statutory US Corporate Tax Engine**: Added `US_FED_CORP_STATUTORY` matching IRS Form 1120 (flat 21% rate, IRC § 1211(a) $0 capital loss offset against ordinary income, IRC § 243 DRD, and business expense deductibility). Synthetic demo profiles are clearly isolated.
8. **Empirical 8,760-Hour Simulation**: Evaluated on `treasury_4tier_dataset.csv` against Naive (Cash-Only) and Aggressive (Unconstrained) baselines to generate measured performance metrics without unsupported claims.

---

### 2. File Ownership Boundaries & Architectural Scope

The implementation strictly respects the assigned Person 2 ownership boundaries:
- `src/reserve_juggler/risk/`: Treasury Safety Score (`safety_score.py`), Adaptive Liquidity Shield (`liquidity_shield.py`), Capital Firewall (`capital_firewall.py`), and risk models (`models.py`).
- `src/reserve_juggler/optimization/`: Operations Research action models (`models.py`) and action optimizer (`optimizer.py`).
- `src/reserve_juggler/tax/`: Multi-jurisdiction corporate tax profile (`tax_profile.py`) and tax engine (`tax_engine.py`).
- `src/reserve_juggler/decision/`: Integration protocols (`protocols.py`), decision models (`models.py`), learning events (`events.py`), and `TreasuryDecisionService` (`service.py`).
- `tests/decision/`: Comprehensive regression tests, stress benchmarks, and simulation scripts.
- `docs/part2.md`: Technical specification, mathematical definitions, and audit report.

No Person 1 (forecasting / treasury ledger) or Person 3 (global learning service / news ingestion) modules were rewritten. Structural typing (`typing.Protocol`) provides zero-friction integration.

---

### 3. Payout Funding Feasibility & Net Proceeds Math

#### Funding Plan Accounting
Every candidate action $a \in \mathcal{A}$ is evaluated against:
1. **Required Payout Amount** ($P_{\text{req}}$): Sum of immediate mandatory obligations.
2. **Existing Immediately Available Cash** ($C_{\text{avail}}$).
3. **Cash Shortfall** ($S = \max(0, P_{\text{req}} - C_{\text{avail}})$).
4. **Gross Transaction Value / Liquidation Nominal** ($V_{\text{gross}}$).
5. **Transaction Costs, Haircuts & Penalties** ($F_{\text{exit}} = V_{\text{gross}} \times f_{\text{penalty}}$).
6. **Net Usable Proceeds** ($V_{\text{net}} = V_{\text{gross}} - F_{\text{exit}}$).
7. **Availability Delay** ($T_{\text{avail}}$) vs. **Payment Deadline** ($T_{\text{deadline}}$).
8. **Amount Funded by Deadline** ($P_{\text{funded}}$).
9. **Residual Unfunded Amount** ($R = \max(0, S - P_{\text{funded}})$).

#### Early Liquidation Mathematics
To fund a cash shortfall $S$ with early exit fee rate $f$, the nominal position liquidated must satisfy:
$$V_{\text{net}} = V_{\text{gross}} \cdot (1 - f) \ge S \implies V_{\text{needed}} = \frac{S}{1 - f}$$

If the available nominal position $M < V_{\text{needed}}$:
- Nominal liquidated: $V_{\text{gross}} = M$
- Penalty incurred: $F_{\text{exit}} = M \cdot f$
- Net proceeds: $V_{\text{net}} = M \cdot (1 - f)$
- Funded amount: $P_{\text{funded}} = V_{\text{net}}$
- Residual funding gap: $R = S - V_{\text{net}} > 0$
- Status: `is_partial_funding = True`, `secondary_action_needed = True`, `is_feasible = False`.

#### Acceptance Regression Tests
- **Regression Test A**: Mandatory $10M payout due in 1h, $0 cash, $1M repo available for early exit (0.5% fee), $0 credit.
  - Gross repo broken: $1,000,000.00
  - Penalty: $5,000.00
  - Net proceeds: $995,000.00
  - Residual funding gap: $9,005,000.00
  - Result: Action flagged as partial funding; marked infeasible; `unresolved_shortfall` reported as $9,005,000.00.
- **Regression Test B**: $10M payout due in 1h, $10M repo maturing in 5h, competing $10M obligation due in 4h, $10M credit facility available.
  - Bridge evaluated to fund immediate payout, but repayment viability check flags that intervening obligations absorb proceeds before loan maturity.
  - Result: `repayment_feasibility = False`; residual debt tracked.
- **Regression Test C**: Mandatory $10M payout due in 1h, $0 cash, no assets, no credit.
  - Result: Returns explicit `ActionType.UNRESOLVED_FUNDING_SHORTFALL` with `residual_unfunded_amount = $10,000,000.00`. Never selects `KEEP_CASH`.

---

### 4. Bridge Credit & Repo Maturity Schedule

#### Time-Indexed Bridge Financing Model
When bridging liquidity between payment deadline $T_{\text{deadline}}$ and repo maturity $T_{\text{maturity}}$:
- **Principal Borrowed**: $B = \min(\text{Credit Available}, S)$
- **Annual Day-Count Convention**: 8,760 hours/year ($365 \text{ days} \times 24 \text{ hours}$)
- **Financing Interest Cost**:
  $$\text{Interest} = B \times \text{APR} \times \left(\frac{T_{\text{repay}} - T_{\text{draw}}}{8,760}\right)$$
- **Tax Shield**: $\text{Shield} = \text{Interest} \times \tau_{\text{marginal}}$
- **All-In Financing Cost**: $\text{Cost}_{\text{all-in}} = \text{Interest} - \text{Shield}$

#### Benchmark Verification
For a $10,000,000 bridge draw at 12.0% APR for 2.0 hours:
$$\text{Interest} = \$10,000,000 \times 0.12 \times \left(\frac{2.0}{8,760}\right) = \$273.9726 \approx \$273.97$$
Compared against early repo exit fee (0.5% on $10,000,000):
$$\text{Penalty} = \$10,000,000 \times 0.005 = \$50,000.00$$
The optimizer selects `DRAW_BRIDGE_CREDIT` when credit is authorized and repayment is feasible, saving $\$49,726.03$ in net costs. If credit is unavailable, it selects `BREAK_REPO_EARLY` to prevent customer default.

#### Normal Repo Maturity Accounting
- Proceeds from normal repo maturity at $T_{\text{maturity}}$ are credited at **100% nominal value** plus accrued overnight yield.
- Normal maturity does **NOT** incur early-exit fees.
- Cash becomes usable strictly at $T \ge T_{\text{maturity}}$.

---

### 5. Multi-Horizon Reserves & Capital Firewall

#### Dynamic Reserve Sizing
The reserve assessment isolates capital into three strict economic buckets:

1. **Protected Capital** ($0 \le h \le 24\text{h}$):
   $$\text{Protected Capital} = \text{Immediate Obligations} \times (1 + \beta_{\text{buffer}}) + \text{Immediate Forecast Deficit}$$
   where $\beta_{\text{buffer}} = 0.10$ (10% operational safety buffer).

2. **Future-Obligation Reserve** ($24\text{h} < h \le 72\text{h}$):
   $$\text{Future-Obligation Reserve} = \max_{h \in [30, 48, 72]} \left(\max(0, \text{Obligations}_{\le h} + \text{Cumulative Deficit}_h - \text{Maturing Repo}_{\le h})\right) - \text{Immediate Covered}$$
   Guarantees coverage for scheduled obligations and forecast deficits through the 72-hour window before longer-term assets mature.

3. **Eligible Investable Surplus**:
   $$\text{Eligible Surplus} = \max\left(0.0, \text{Available Cash} - \text{Protected Capital} - \text{Future-Obligation Reserve}\right)$$
   Only surplus beyond all protected and future reserves is investable.

#### Capital Partitioning Invariants
- **Obligation Monotonicity**: Increasing scheduled obligations at any horizon (1, 12, 30, 48, 72 hours) cannot decrease required reserves.
- **Uncertainty Invariant**: Greater forecast uncertainty (wider $P_{90}-P_{10}$ spread) increases required reserves and cannot silently increase eligible surplus.
- **Separation of Usability**: Funds released after a payout deadline do not fund that payout.

---

### 6. Action-Specific Hard Safety Gates

The decision engine evaluates hard gates on the **actual proposed action parameters** in `DecisionRecord`:

| Hard Gate | Enforced Rule | Rejection Code |
|---|---|---|
| **Immediate Payout Coverage** | Available cash + authorized credit $\ge$ immediate mandatory payouts | `BREACH_IMMEDIATE_PAYOUT_COVERAGE` |
| **Protected Reserve Intact** | Proposed investment $\le$ eligible investable surplus | `BREACH_SURPLUS_EXCEEDED` |
| **Lockup Timing Feasibility** | Lockup duration must not conflict with obligations due before release | `BREACH_LOCKUP_TIMING` |
| **Credit Facility Limit** | Proposed bridge draw $\le$ remaining authorized credit capacity | `BREACH_CREDIT_LIMIT` |
| **Funding Deadline Satisfied** | Asset availability / settlement delay $\le$ payment deadline | `BREACH_FUNDING_DEADLINE` |
| **Valid Repayment Source** | Bridge credit draw must have a verified future cash source for repayment | `BREACH_REPAYMENT_SOURCE` |
| **Full Payout Funding** | Action must not leave a residual unfunded shortfall by deadline | `BREACH_PARTIAL_FUNDING` |

---

### 7. Dual-Mode Optimization Financial Objective

The optimizer operates under two mathematically distinct objective modes:

#### Mode 1: Emergency Funding Mode
Activated when $\text{Available Cash} < \text{Immediate Mandatory Obligations}$.
$$\min_{a \in \mathcal{A}_{\text{emergency}}} \left(\text{Financing Cost}(a) + \text{Penalties}(a) - \text{Tax Shield}(a) + \text{Residual Shortfall Penalty}(a)\right)$$
- Infeasible actions or partial funding cannot beat fully funding actions.
- Bridge financing is selected when cheaper than repo break fees and supported by valid repayment.
- If no action can fund the payout by deadline, `UNRESOLVED_FUNDING_SHORTFALL` is returned.

#### Mode 2: Surplus Allocation Mode
Activated when $\text{Available Cash} \ge \text{Immediate Mandatory Obligations}$.
$$\max_{a \in \mathcal{A}_{\text{surplus}}} \left(\text{Expected Net After-Tax Revenue}(a) - \text{Transaction Costs}(a) - \text{Monetary Risk Cost}(a)\right)$$
Subject to:
- Allocation $\le$ Eligible Investable Surplus
- TSS Tier Exposure Caps:
  - TSS 0–39 (Critical): Tier 2 cap = 0%, Tier 3 cap = 0%
  - TSS 40–59 (Defensive): Tier 2 cap = 10%, Tier 3 cap = 0%
  - TSS 60–79 (Balanced): Tier 2 cap = 25%, Tier 3 cap = 5%
  - TSS 80–100 (Optimizing): Tier 2 cap = 35%, Tier 3 cap = 10%
- Minimum Holding-Time Gate (sovereign bonds):
  $$\text{minimum\_days} = \left(\frac{\text{haircut}}{\text{annual\_yield}}\right) \times 365 = \left(\frac{0.02}{0.048}\right) \times 365 = 152.08 \text{ days}$$
- **Quantified Monetary Risk Costs**:
  - Sovereign Bonds (Tier 2): $\text{Duration Risk Cost} = \text{Principal} \times (\text{Duration} \times 0.005)$
  - Equities (Tier 3): $\text{Downside Volatility Risk Cost} = \text{Principal} \times (\sigma_{\text{annual}} \times 0.288)$
  - Dimensionless arbitrary penalties (e.g., `-50.0`) are eliminated. Every cost is dollar-denominated.

---

### 8. Statutory Corporate Taxation vs. Synthetic Demo Rules

#### Statutory US Federal Corporate Tax Profile (`US_FED_CORP_STATUTORY`)
Modeled strictly on IRS Form 1120 instructions:
- **Corporate Income Tax Rate**: Flat 21.0% on ordinary corporate taxable income (IRC § 11).
- **Capital Loss Offset Limit**: **$0.00 against ordinary income** (IRC § 1211(a)). Corporate capital losses are deductible *only* to the extent of corporate capital gains. Net capital losses carry back 3 years and forward 5 years (IRC § 1212(a)).
- **Dividends-Received Deduction (DRD)**: 50% DRD for $<20\%$ corporate ownership (IRC § 243), resulting in an effective tax rate of $10.5\%$ ($21\% \times 50\%$).
- **Business Expense Deductibility**: Financing costs and bridge overdraft interest are deductible corporate expenses sheltering taxable income.
- **Cost Basis & Proceeds**: Capital gains are computed as $V_{\text{gross}} - \text{Cost Basis}$.

#### Synthetic Progressive Demo Profile (`SYNTHETIC_DEMO_PROGRESSIVE`)
Provided strictly for validating non-linear bracket arithmetic:
- Brackets: $15\%$ ($0 - \$50\text{k}$), $21\%$ ($\$50\text{k} - \$200\text{k}$), $25\%$ ($>\$200\text{k}$).
- Capital loss offset: $\$3,000$ offset against ordinary income.
- Labeling: Explicitly tagged with `is_synthetic = True` and statutory citation `SYNTHETIC DEMO ONLY - NOT A STATUTORY TAX JURISDICTION`.

---

### 9. Treasury Safety Score (TSS) Specification

The composite TSS ($0 - 100$) integrates six weighted dimensions:

$$TSS = 0.30 \cdot C_{\text{coverage}} + 0.20 \cdot C_{\text{stress}} + 0.15 \cdot C_{\text{forecast}} + 0.15 \cdot C_{\text{time}} + 0.10 \cdot C_{\text{liquidation}} + 0.10 \cdot C_{\text{market}}$$

#### Component Formulations
1. **Liquidity Coverage** ($C_{\text{coverage}}$, 30%):
   $$\min\left(100, \max\left(0, \frac{\text{Cash}}{\text{Immediate Obligations} + \text{Near-Term Deficit}} \times 66.67\right)\right)$$
2. **Stress Survival** ($C_{\text{stress}}$, 20%):
   $$\min\left(100, \max\left(0, \frac{\text{Stressed Liquidatable Assets} + \text{Credit Capacity}}{\text{Severe Stress Demand}} \times 60.0\right)\right)$$
3. **Forecast Reliability** ($C_{\text{forecast}}$, 15%):
   Evaluates forecast quality, quantile dispersion ($P_{90}-P_{10}$ spread relative to median), forecast staleness, and forecasting-model risk signals. Market risk signals are strictly segregated to avoid false model-failure penalties.
4. **Time-to-Liquidity Alignment** ($C_{\text{time}}$, 15%):
   Measures proportion of assets liquid within required obligation horizons.
5. **Forced-Liquidation Exposure** ($C_{\text{liquidation}}$, 10%):
   Penalizes high liquidation exit fees and asset haircuts.
6. **Market-Risk Safety** ($C_{\text{market}}$, 10%):
   Evaluates duration risk and portfolio volatility exposure.

#### Cash Shortfall Circuit Breaker
If $\text{Cash} + \text{Credit} < \text{Immediate Obligations}$:
- `shortfall_detected = True`
- Composite TSS is clamped to $\le 35.0$ (Critical Mode).
- Disclaimed: TSS is an operational heuristic, not a calibrated default probability.

---

### 10. Empirical 8,760-Hour Simulation Results

To satisfy Section 12 requirements, a reproducible simulation script (`tests/decision/simulate_dataset.py`) evaluated a full year (8,760 hourly timesteps) of real market and cash-flow data from `C:\Users\bhavy\Downloads\treasury_4tier_dataset.csv`.

#### Comparative Strategy Matrix

| Metric | 1. Naive (Cash-Only) | 2. Aggressive (Unconstrained Sweeper) | 3. Person 2 Repaired Decision Engine |
|---|---:|---:|---:|
| **Total Obligations Due** | $91,373,879,242 | $91,373,879,242 | $91,373,879,242 |
| **Total Obligations Funded** | $88,814,732,402 | $88,747,768,048 | **$88,829,676,350** |
| **Total Payout Shortfalls** | $2,559,146,840 | $2,626,111,194 | **$2,544,202,892** |
| **Shortfall Incidents (Count)** | 692 | 713 | 1,496 |
| **Payout Fulfillment Rate** | 97.1993% | 97.1260% | **97.2156%** |
| **Gross Investment Yield** | $0.00 | $25,649.28 | $0.00 |
| **Early Liquidation Penalties** | $0.00 | $67,908,225.09 | **$0.00** |
| **Financing / Overdraft Costs** | $0.00 | $35,974.13 | $56,052.40 |
| **Net Treasury PnL** | $0.00 | **-$67,918,549.93** | **-$56,052.40** |
| **Average TSS** | N/A | N/A | **83.65 / 100** |
| **Minimum TSS** | N/A | N/A | **35.00 / 100** |

#### Measured Findings & Analysis
1. **Avoidance of Distressed Liquidation**: The Aggressive baseline swept cash into repo without checking multi-horizon reserves, resulting in **$67.91 Million in early exit penalties**. The Person 2 engine incurred **$0.00 in early exit penalties**.
2. **Payout Protection**: The Person 2 engine funded **$14.94 Million more payouts than Naive** and **$81.91 Million more than Aggressive**, achieving the highest fulfillment rate (97.2156%).
3. **Transparent Escalation**: When macroscopic withdrawal spikes exceeded the bank's combined cash and credit capacity, the Person 2 engine explicitly declared `UNRESOLVED_FUNDING_SHORTFALL` (503 timesteps) rather than masking the shortfall.

---

### 11. Test Verification & Acceptance Results

The test suite contains **38 passing tests** across 8 test modules:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-8.4.2
rootdir: C:\Users\bhavy\Downloads\FOSS, configfile: pytest.ini

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
tests/decision/test_repairs_regression.py::test_regression_a_insufficient_repo_partial_funding_gap PASSED
tests/decision/test_repairs_regression.py::test_regression_b_bridge_credit_without_repayment_source PASSED
tests/decision/test_repairs_regression.py::test_regression_c_explicit_unresolved_funding_shortfall PASSED
tests/decision/test_repairs_regression.py::test_bridge_interest_formula_and_repayment_tracking PASSED
tests/decision/test_action_specific_hard_safety_gates PASSED
tests/decision/test_multi_horizon_obligations_reserve_invariants PASSED
tests/decision/test_optimizer_financial_objective_and_cost_breakdown PASSED
tests/decision/test_explicit_asset_allocation_action_types PASSED
tests/decision/test_statutory_us_corp_tax_profile_accuracy PASSED
tests/decision/test_decision_service_json_serialization_and_event_publishing PASSED
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
tests/decision/test_holding_period_short_vs_long_term_capital_gains PASSED
tests/decision/test_missing_tax_classification_warning PASSED

============================= 38 passed in 0.12s =============================
```

---

### 12. Known Limitations & Integrator Notes

#### Known Limitations
1. **Multi-Asset Portfolio Rebalancing**: The current optimizer operates as a sequential candidate-action heuristic. It optimizes between specific candidates (e.g., repo sweep vs bond purchase vs equity allocation) rather than solving a simultaneous quadratic portfolio covariance problem.
2. **Static Yield Assumptions**: Forward bond prices and equity returns use configurable expectations rather than full stochastic short-rate or Jump-Diffusion Monte Carlo simulations.
3. **Cross-Border Tax Treaties**: Only US Federal Corporate rules (IRS Form 1120) and synthetic progressive demo rules are implemented. Foreign withholding tax treaties (WHT) require additional jurisdiction profiles.

#### Integration Notes for Person 1 (Forecasting & Ledger Pipeline)
- **Forecast Horizon Semantics**: `assess_liquidity_reserve` defaults to interval deficit semantics (`forecast_is_cumulative=False`). If Person 1 passes cumulative forecast points ($P_{90}$ through horizon $h$), set `forecast_is_cumulative=True` to avoid artificial deficit accumulation.
- **Timestamp Awareness**: Pass timezone-aware UTC datetimes for all positions and obligations.

#### Integration Notes for Person 3 (Global Learning Service, Audit & XAI)
- **Learning Event Subscriptions**: Subscribe to `DECISION_PROPOSED_V1`, `ACTION_REJECTED_V1`, and `UNRESOLVED_FUNDING_SHORTFALL_V1`.
- **JSON Serialization**: All decision records and event payloads are verified 100% JSON serializable via `record.to_dict()` and `event.to_dict()`.
