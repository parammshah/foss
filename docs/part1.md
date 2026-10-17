# Reserve Juggler — Data Foundation, Treasury Ledger, Simulation and Hybrid Forecasting
**Team 1 Final Architecture & Engineering Deliverable**

---

## 1. Executive Summary & Deliverables

Team 1 has implemented the foundational layer of **Reserve Juggler**, a quantitative treasury management and simulation system for a digital bank handling ~$20 million in hourly deposits, withdrawals, and merchant settlements.

### Scope & Modules Exclusively Implemented
- `src/reserve_juggler/contracts.py`: Canonical shared integration schemas and injectable `EventPublisher`.
- `src/reserve_juggler/data/`: Data schema, validator, preprocessor, feature engineering with zero lookahead leakage, and realistic 8,760-hour benchmark dataset generator.
- `src/reserve_juggler/ledger/`: Hourly double-entry treasury ledger, Tier 0-3 asset mechanics, mark-to-market vs. liquidatable valuations, and credit facility manager.
- `src/reserve_juggler/forecasting/`: Statistical baselines, CPU-friendly ML (Ridge regression), Operations Research (OR) constrained ensemble combination, and quantile/cumulative scenario reconciler.
- `src/reserve_juggler/simulation/`: Hourly operations simulation runner, action simulator, and execution outcome recorder.
- `src/reserve_juggler/evaluation/`: MAE, WAPE, Pinball Quantile Loss, PICP interval coverage, and chronological rolling-origin backtesting.
- `tests/core/`: 34 comprehensive unit and integration tests (100% pass rate).
- `config/treasury_config.yaml`: Fully parameterized treasury configuration.
- `requirements-part1.txt`: Explicit dependency manifest.

---

## 2. Canonical Integration Contracts (`contracts.py`)

All schemas inherit from Pydantic `BaseModel` (`frozen=True`) with automated UTC enforcement and JSON serialization.

### Financial Representation Standards
- **Currency**: USD, represented as 64-bit float.
- **Rates & Percentages**: Normalized decimal fractions (e.g., $4.8\%$ is represented as `0.048`; $0.5\%$ is `0.005`).
- **Timestamps**: UTC-aware `datetime` objects (`datetime.timezone.utc`).

### Contract Catalog

| Schema Name | Purpose | Key Attributes |
|---|---|---|
| `TreasuryState` | Point-in-time financial snapshot | `timestamp`, `operational_cash_usd`, `asset_positions`, `obligations`, `credit_facility`, `warnings`, `total_liquidatable_value_usd`, `net_liquidity_usd` |
| `AssetPosition` | Holding in an asset tier | `asset_id`, `tier` (0..3), `units`, `unit_price_usd`, `market_value_usd`, `liquidatable_value_usd`, `annual_yield_or_return`, `early_exit_fee_rate`, `liquidation_haircut_rate` |
| `Obligation` | Scheduled payout liability | `obligation_id`, `timestamp`, `amount_usd`, `obligation_type`, `is_settled`, `priority` |
| `CreditFacility` | Committed emergency liquidity line | `facility_id`, `is_available`, `committed_limit_usd`, `utilized_amount_usd`, `apr`, `fee_rate`, `available_capacity_usd` |
| `ForecastPoint` | Probabilistic multi-quantile prediction | `as_of_timestamp`, `target_timestamp`, `horizon_hours`, `deposit_p10/p50/p90`, `withdrawal_p10/p50/p90`, `net_outflow_p50/p95/p99`, `cumulative_net_outflow_p95/p99` |
| `ForecastBundle` | Multi-horizon forecast deliverable | `forecast_timestamp`, `points`, `scenario_references`, `data_quality_status`, `model_versions`, `evaluation_metadata` |
| `RiskSignal` | Incoming market or liquidity alert | `signal_id`, `source`, `category`, `severity` [0..1], `confidence` [0..1], `evidence_references`, `stress_scenario_tags` |
| `DecisionRecord` | Optimizer decision output (Team 2) | `decision_id`, `timestamp`, `status`, `selected_action`, `protected_reserve_usd`, `eligible_surplus_usd`, `cost_breakdown` |
| `ExecutionOutcome` | Outcome of executing an action | `outcome_id`, `decision_id`, `execution_status`, `realized_cash_movements`, `costs_usd`, `penalties_usd`, `losses_usd`, `shortfall_usd` |
| `LearningEvent` | Telemetry payload (Team 3) | `event_id`, `event_type`, `source_module`, `event_timestamp`, `recording_timestamp`, `payload`, `model_version`, `schema_version` |

### Stable Integration Interfaces

```python
# 1. State Inspection
state: TreasuryState = ledger.get_state(as_of_timestamp: datetime)

# 2. Action Simulation
new_state, outcome = ledger.simulate_action(
    state: TreasuryState,
    action: Union[Dict[str, Any], TreasuryAction],
    forecast_bundle: Optional[ForecastBundle] = None
)

# 3. Forecast Generation
bundle: ForecastBundle = forecast_service.forecast(
    as_of_timestamp: datetime,
    horizons_hours: Optional[List[int]] = [1, 6, 24, 48, 72]
)
```

---

## 3. Dataset Processing & Integrity Validation

### Dataset Schema Fields
The challenge dataset contains 8,760 hourly rows (365 days):
- `timestamp`: UTC ISO-8601 string.
- `deposits_usd`: Customer deposit inflows ($\ge 0$).
- `withdrawals_usd`: Customer withdrawal outflows ($\ge 0$).
- `net_flow_usd`: Reconciled net flow ($= \text{deposits} - \text{withdrawals}$).
- `repo_sweep_window_open`: Binary flag (1 at 17:00, 0 otherwise).
- `repo_sweep_mature`: Binary flag (1 at 08:00, 0 otherwise).
- `tier1_repo_rate_apy`: Decimal rate (e.g. 0.043).
- `tier2_bond_yield_pct`: Decimal yield (e.g. 0.048).
- `tier2_bond_price_index`: Bond price index (normalized around 100.0).
- `tier2_bond_duration_years`: Effective duration ($\approx 1.9$ years).
- `tier3_stock_index_price`: Equity index price ($\approx 5000.0$).
- `tier3_stock_volatility_pct`: Annualized volatility ($\approx 0.18$).
- `penalty_rate_tier1_early_exit`: Repo early-exit penalty ($0.005$).
- `haircut_rate_tier2_emergency`: Sovereign bond liquidation haircut ($0.02$).
- `slippage_rate_tier3_emergency`: Equity emergency liquidation slippage ($0.15$).
- `market_hours_open`: Financial market trading hours (weekdays 14:00-21:00 UTC).

### Strict Leakage-Free Feature Engineering
Features at time step $t$ are derived **strictly** from observations up to $t-1$:
1. **Lags**: $x_{t-1}, x_{t-2}, x_{t-3}, x_{t-6}, x_{t-12}, x_{t-24}, x_{t-48}, x_{t-72}, x_{t-168}$.
2. **Rolling Statistics**: Rolling mean, std, min, max over $[6, 24, 72, 168]$ hours computed on `series.shift(1)` so row $t$ is never included in the statistics for row $t$.
3. **Calendar Features**: Sinusoidal hour/day cyclical embeddings ($\sin, \cos$).
4. **Mathematical Verification**: `FeatureEngineer.assert_no_lookahead_leakage()` perturbs $x_T$ by $+\$100\text{M}$ and asserts that all features for $t \le T$ remain strictly invariant.

---

## 4. Treasury Ledger & Asset Tier Mechanics

### Double-Entry Balance Sheet Identity
At all hourly intervals $t$:
$$\text{Total Assets} = \text{Operational Cash (Tier 0)} + \sum_{i=1}^3 \text{Marked-to-Market Asset Holdings}_i + \text{Receivables}$$
$$\text{Total Liabilities} = \text{Customer Deposit Liabilities} + \text{Credit Facility Drawn} + \text{Payables}$$
$$\text{Bank Equity} = \text{Total Assets} - \text{Total Liabilities}$$

### Asset Tiers & Liquidation Value Equations

$$\text{Liquidatable Value} = \begin{cases} 
\text{Market Value} & \text{Tier 0 (Cash)} \\
\text{Market Value} \cdot (1 - \text{early\_exit\_fee}) & \text{Tier 1 (Repo, if broken before 08:00)} \\
\text{Market Value} + \text{Accrued Interest} & \text{Tier 1 (Repo, at 08:00 maturity)} \\
\text{Market Value} \cdot (1 - 0.02) & \text{Tier 2 (Sovereign Bonds, emergency)} \\
\text{Market Value} \cdot (1 - 0.15) & \text{Tier 3 (Equities, emergency)}
\end{cases}$$

---

## 5. Hybrid ML + Operations Research (OR) Forecasting

### Role Distinction
- **Machine Learning (ML)**: Learns complex non-linear multivariate relationships from historical features.
- **Operations Research (OR)**: Does not conjure predictions out of thin air; rather, it combines validated candidate forecasts under rigorous financial constraints and reconciles multi-horizon risk scenarios.

### Mathematical Formulation of OR Constrained Ensemble
Given validation predictions from $M$ models $\hat{Y} \in \mathbb{R}^{N \times M}$ and ground truth $y \in \mathbb{R}^N$:
$$\min_{w \in \mathbb{R}^M} \frac{1}{2} \left\| \frac{y}{\sigma} - \frac{\hat{Y}}{\sigma} w \right\|_2^2 + \frac{\gamma}{2} \left\| w - \frac{1}{M}\mathbf{1} \right\|_2^2$$
$$\text{subject to } w_i \ge 0 \quad \forall i=1,\dots,M, \quad \sum_{i=1}^M w_i = 1.0$$
- Solved via **Sequential Least Squares Programming (SLSQP)**.
- Guaranteed fallback via **Euclidean Simplex Projection** $\Pi_{\Delta}(v)$.

### OR Scenario Reconciliation
- Ensures **gross flow non-negativity**: $\hat{D} \ge 0, \hat{W} \ge 0$.
- Preserves **net flow identity**: $\text{Net Outflow} = \hat{W} - \hat{D}$.
- Enforces **monotonic quantiles**: $P10 \le P50 \le P90 \le P95 \le P99$.
- Multi-hour cumulative shock buffers account for empirical serial autocorrelation:
  $$\text{CumNet}_H^{(95)} = H \cdot \text{Net}_{50} + z_{0.95} \cdot \sigma_{\text{net}} \sqrt{H} \sqrt{1 + 2\hat{\rho}_1}$$

---

## 6. Empirical Benchmark Results (8,760-Hour Evaluation)

Evaluated via chronological rolling-origin backtesting across 10 weekly test origins on the full year dataset:

| Target Variable | Horizon | Recent Baseline MAE | Seasonal Baseline MAE | Ridge ML MAE | OR Ensemble MAE | Top OR Model Weight |
|---|---|---|---|---|---|---|
| **Deposits** | **1-Hour** | $2,595,818 (14.6%) | $1,103,776 (6.2%) | **$1,066,937 (6.0%)** | $1,125,237 (6.3%) | Seasonal (97.3%) |
| **Deposits** | **6-Hour** | $8,676,611 (132.7%) | $1,046,536 (16.0%) | $1,407,525 (21.5%) | **$1,032,665 (15.8%)** | Seasonal (78.9%), ML (18.6%) |
| **Deposits** | **24-Hour** | $3,781,406 (19.9%) | **$1,054,649 (5.5%)** | $1,550,359 (8.2%) | $1,069,147 (5.6%) | Seasonal (93.5%) |
| **Withdrawals** | **1-Hour** | $1,801,746 (10.8%) | $1,250,418 (7.5%) | $1,495,151 (8.9%) | **$1,231,625 (7.4%)** | Seasonal (97.4%) |
| **Withdrawals** | **6-Hour** | $9,259,075 (144.6%) | $780,177 (12.2%) | $977,159 (15.3%) | **$777,071 (12.1%)** | Seasonal (97.3%) |
| **Withdrawals** | **24-Hour** | $4,096,906 (20.7%) | **$1,645,634 (8.3%)** | $2,414,687 (12.2%) | $1,655,252 (8.4%) | Seasonal (83.6%), ML (13.8%) |

*Key Takeaway*: The OR Constrained Ensemble consistently avoids the massive error blowups of recent-average baselines at multi-hour horizons (e.g. at 6h, Recent MAE was $\$8.7\text{M}$ while OR Ensemble achieved $\$1.03\text{M}$), beating all individual submodels at 6h horizons.

---

## 7. System-Wide Learning Integration (Team 3 Interface)

An injectable `EventPublisher` protocol is provided in `contracts.py`:
- `NoOpEventPublisher`: Default zero-overhead publisher for standalone execution and testing.
- `InMemoryEventPublisher`: In-memory event sink supporting synchronous callback subscription.

### Emitted Learning Events

| Event Type | Source Module | Payload Contents |
|---|---|---|
| `dataset_validated` | `data.validator` | Status, issue count, missing hours, row count |
| `data_gap_imputed` | `data.preprocessor` | Imputed timestamp list, imputation method |
| `forecast_generated` | `forecasting.service` | As-of time, horizons, model version |
| `forecast_evaluated` | `evaluation.backtest` | Actual vs. predicted, submodel breakdown, error metrics |
| `repo_matured` | `ledger` | Asset ID, principal, interest credited to cash |
| `liquidity_shortfall` | `ledger` | Deficit USD, emergency credit drawn |
| `simulation_step_completed`| `simulation.engine` | Step index, cash, liquidatable value, shortfall |

---

## 8. Verification & Test Execution

The entire test suite consists of 34 unit tests:
```bash
# Run complete test suite
.venv/bin/pytest tests/core/ -v

# Run rolling-origin backtest benchmark
.venv/bin/python -c '
import pandas as pd
from reserve_juggler.evaluation import RollingOriginBacktest
df = pd.read_csv("data/challenge_dataset_8760.csv")
report = RollingOriginBacktest().run(df, horizons=[1, 6, 24])
print(report.summary())
'
```

---

## 9. Assumptions and Hand-off Notes for Teams 2 & 3

1. **Unverified External Data**: The challenge brief outlined 10 columns; we extended the schema to include `net_flow_usd`, duration, penalty columns, and market hours. If an external production CSV is provided by the organizers, pass it through `DatasetPreprocessor.preprocess()`, which automatically normalizes unit percentages (e.g. $4.8\% \rightarrow 0.048$) and populates default assumptions.
2. **Mandatory Payout Reserves**: Team 2's optimizer should query `state.upcoming_obligations_usd` and lock `protected_reserve_usd` at least equal to $P95$ net outflow buffer before allocating surplus into Tiers 1-3.
3. **Emergency Liquidation Ordering**: We recommend liquidating Tier 1 (0.5% break penalty) first, Tier 2 (2% bond haircut) second, and Tier 3 (15% equity slippage) last.
