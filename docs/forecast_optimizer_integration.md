# Reserve Juggler — Forecast & Optimizer Integration Guide (Team 1 to Team 2)

**Document Version**: 2.0 (Team 2 Integration Ready)  
**Producing Team**: Team 1 (Data Foundation, Forecasting & Financial Simulation)  
**Consuming Team**: Team 2 (Treasury Optimization & Asset-Tier Allocation)  
**Canonical Schema Source**: `reserve_juggler.contracts.ForecastBundle`, `reserve_juggler.contracts.ForecastPoint`

---

## 1. Executive Summary & Architecture Boundary

The Reserve Juggler quantitative architecture enforces a strict separation of concerns between forecast estimation and financial allocation:

1. **Team 1 (Forecasting Engine)** provides **estimates, probability distributions, quantile shocks, empirical calibration, and reliability metadata**.
   - The forecasting engine **never** decides that money is safe to sweep or invest.
   - Forecasts are statistical estimates based on historical patterns, not contractual guarantees of future customer cash flows.
2. **Team 2 (Treasury Optimizer)** is exclusively responsible for **portfolio allocation, liquidity sizing, asset sweeps, and emergency funding decisions**.
   - The optimizer **must independently enforce hard liquidity and legal payout constraints**.
   - The optimizer must consume uncertainty quantiles conservatively and never override balance-sheet safety rules based solely on favorable point predictions.

---

## 2. Financial Units, Semantics & Accounting Integrity

To prevent fatal accounting discrepancies, the optimizer must adhere strictly to these unit and sign conventions:

| Field / Quantity | Units | Valid Domain | Accounting Semantics |
| :--- | :--- | :--- | :--- |
| `deposits_usd` | USD ($) | $[0, +\infty)$ | Gross customer/merchant inflows entering Tier 0 cash. |
| `withdrawals_usd` | USD ($) | $[0, +\infty)$ | Gross customer payout and settlement outflows leaving Tier 0 cash. |
| `net_flow_usd` | USD ($) | $(-\infty, +\infty)$ | Net flow = $\text{Deposits} - \text{Withdrawals}$. Positive = cash surplus; Negative = cash deficit. |
| `net_outflow_p50/p95/p99` | USD ($) | $(-\infty, +\infty)$ | Net outflow = $\text{Withdrawals} - \text{Deposits}$. Positive = net cash drain; Negative = net surplus. |
| Interest Rates & APY | Decimal | $[0.0, 1.0]$ | Ingestion boundary normalized (e.g., $4.29\%$ is stored as `0.0429`). |
| Haircuts & Slippage | Decimal | $[0.0, 1.0]$ | Decimal fractions (e.g., Repo early penalty `0.005`, Bond haircut `0.02`, Equity slippage `0.15`). |
| Price Indices | Points | $[0, +\infty)$ | Sovereign bond index ($\sim 100$) and Equity index ($\sim 5000$). **Not dollar portfolio balances.** |

> [!CAUTION]
> **Zero Double-Counting Rule**:
> Do NOT combine `net_outflow_p50` with individual `withdrawal_p50` or `deposit_p50` in the same constraint equation. Choose either:
> 1. Component gross flows: $\Delta \text{Cash} = \text{Deposit} - \text{Withdrawal}$, OR
> 2. Unified net outflow: $\Delta \text{Cash} = -(\text{Net Outflow})$.

---

## 3. Forecast Contract Specifications

Team 1 delivers forecasts encapsulated in a canonical `ForecastBundle` containing a series of `ForecastPoint` objects for requested horizons ($h \in \{1, 6, 24, 48, 72\}$ hours).

### `ForecastPoint` Field Reference

Each `ForecastPoint` exposes multi-level quantile distributions and reliability metadata:

```python
class ForecastPoint(BaseModel):
    as_of_timestamp: datetime            # Forecast origin (UTC)
    target_timestamp: datetime          # Prediction horizon target (UTC)
    horizon_hours: int                  # Lead time in hours (1, 6, 24, 48, 72)

    # Point Expectations (USD)
    deposit_point_forecast: float       # Expected gross deposit inflow
    withdrawal_point_forecast: float    # Expected gross withdrawal outflow

    # Deposits Quantile Distribution (USD)
    deposit_p025: float                 # 2.5% conservative lower tail (95% interval)
    deposit_p10: float                  # 10% conservative lower tail (80% interval)
    deposit_p25: float                  # 25% lower quartile (50% interval)
    deposit_p50: float                  # 50% median expected deposits
    deposit_p75: float                  # 75% upper quartile
    deposit_p90: float                  # 90% optimistic upper tail
    deposit_p975: float                 # 97.5% optimistic upper tail

    # Withdrawals Quantile Distribution (USD)
    withdrawal_p025: float              # 2.5% optimistic lower tail
    withdrawal_p10: float               # 10% optimistic lower tail
    withdrawal_p25: float               # 25% lower quartile
    withdrawal_p50: float               # 50% median expected withdrawals
    withdrawal_p75: float               # 75% adverse upper quartile
    withdrawal_p90: float               # 90% adverse upper tail (80% interval)
    withdrawal_p975: float              # 97.5% extreme adverse tail (95% interval)

    # Coherent Net Outflow Shocks: (Withdrawals - Deposits)
    net_outflow_p50: float              # Median single-hour net cash outflow
    net_outflow_p95: float              # Adverse 95% single-hour net outflow shock
    net_outflow_p99: float              # Extreme stress 99% single-hour net outflow shock
    cumulative_net_outflow_p95: float   # Cumulative net outflow buffer from t=0 to target (P95)
    cumulative_net_outflow_p99: float   # Cumulative net outflow buffer from t=0 to target (P99)

    # Calibration & Reliability Metadata
    calibration_sample_size: int        # N historical residuals used in uncertainty calibration
    training_cutoff_timestamp: datetime # Latest training data timestamp observed
    reliability_score: float            # Out-of-sample confidence [0.0 to 1.0]
    reliability_tier: str               # "HIGH", "MEDIUM", "LOW", or "UNCALIBRATED"
    reliability_rationale: str          # Grounded justification for tier
    is_fallback: bool                   # True if conservative degraded fallback was activated
    model_version: str                  # Submodel identification
```

---

## 4. Empirical Uncertainty Calibration & Optimizer Guidelines

An independent 60-day chronological rolling-origin backtest on the actual challenge dataset revealed the following out-of-sample coverage behavior:

| Nominal Interval | Target Coverage | Observed Empirical Coverage | Mean Interval Width (Withdrawals) |
| :--- | :--- | :--- | :--- |
| **50% Interval** ($P_{25} - P_{75}$) | **50.0%** | **28.3% – 35.0%** | $\sim \$1.1\text{M}$ |
| **80% Interval** ($P_{10} - P_{90}$) | **80.0%** | **50.0% – 63.3%** | $\sim \$2.3\text{M} - \$2.6\text{M}$ |
| **95% Interval** ($P_{02.5} - P_{97.5}$) | **95.0%** | **70.0% – 88.3%** | $\sim \$4.2\text{M} - \$4.8\text{M}$ |

### Mandatory Optimization Sizing Rules
1. **Never Size Liquidity on Point Forecasts**:
   Median predictions (`deposit_p50`, `withdrawal_p50`) represent the center of mass. Sizing liquidity solely against $P_{50}$ guarantees a $\sim 50\%$ probability of underfunding customer withdrawals.
2. **Use P90 or P95 for Discretionary Sweeps**:
   When deciding how much operational cash can safely be swept into Tier 1 Repo or Tier 2 Bonds, Team 2 must compute investable surplus using conservative outflow bounds:
   $$\text{Max Sweepable Surplus} = \text{Current Cash} - \text{Mandatory Reserve} - \text{Withdrawal}_{P90} + \text{Deposit}_{P10}$$
   If `reliability_tier` is `MEDIUM` or `LOW`, substitute `Withdrawal_p975` and set `Deposit_p025 = 0.0`.
3. **Multi-Hour Shock Buffers**:
   For multi-day planning ($h=24, 48, 72$), utilize `cumulative_net_outflow_p95`. This term accounts for multi-hour serial correlation ($\rho_1 \approx 0.28$), scaling higher than simple $\sqrt{h}$ random-walk assumptions.

---

## 5. Downstream Optimizer Hard Constraints & Safety Boundaries

The optimization formulation must incorporate the following physical, market, and institutional rules as **hard mathematical constraints** (never soft penalties):

### A. Immediate Legal Payout Obligations
- Any scheduled obligation in `TreasuryState.upcoming_obligations` with priority 1 (customer payouts, merchant batch settlements) must be funded **with 100% certainty** from immediately available Tier 0 operational cash.
- Settlement cannot rely on anticipated future deposit inflows.

### B. Tier 0 Operational Cash Minimum ($C_{\min}$)
- Operational cash must never fall below the configured policy floor (e.g., $\$10,000,000$ or $10\%$ of peak daily volume).
- Cash below $C_{\min}$ triggers immediate halt of all Tier 1, 2, and 3 sweeps.

### C. Tier 1 Repo Sweep Windows & Settlement Timing
- **Entry Window**: Repo investments can only be executed when `repo_sweep_window_open == True` (daily 17:00–18:00 UTC).
- **Maturity Window**: Overnight repo matures and pays principal plus interest into operational cash at 08:00 UTC (`repo_sweep_mature == True`).
- **Illiquidity Lock**: Between 18:00 UTC and 08:00 UTC, Tier 1 repo is illiquid. Breaking repo prior to 08:00 incurs a **0.50% early redemption penalty** (`tier1_early_redemption_haircut_pct = 0.0050`).

### D. Tier 2 Sovereign Bonds Liquidation Mechanics
- Clean price index ($\sim 100.0$) fluctuates with benchmark yield ($\sim 4.8\%$).
- Secondary market emergency liquidation incurs a **2.00% liquidation haircut** (`tier2_bond_liquidation_haircut_pct = 0.0200`).
- Settlement delay: Secondary bond sales require a settlement delay (typically $T+1$ or 2 hours in simulation); cash is not instantaneous.

### E. Tier 3 Equities Liquidation Mechanics
- Equity index ($\sim 5000$) exhibits annualized volatility ($\sim 18\%$).
- Emergency equity liquidation incurs a **15.00% market impact / slippage penalty** (`tier3_stock_liquidation_haircut_pct = 0.1500`).
- Tier 3 assets must be the absolute last tier liquidated before drawing emergency credit.

### F. Emergency Credit Line & Overdraft Facility
- Committed credit facility limit $L_{\text{credit}}$.
- Uncommitted borrowing or overdraft incurs an aggressive penalty of **12.00% APR** (`emergency_overdraft_penalty_apr_pct = 0.1200`).
- Overdraft must be treated as a distress state, not an arbitrage funding mechanism.

---

## 6. Stale Data, Fallbacks & Data-Quality Handling

When data feeds are interrupted or market feeds freeze:
1. `ForecastBundle.data_quality_status == DataQualityStatus.DEGRADED` (or `WARNING`).
2. `ForecastBundle.is_fallback_bundle == True`.
3. `ForecastPoint.is_fallback == True` and `reliability_tier == "UNCALIBRATED"`.

### Fallback Policy for Team 2:
```
IF is_fallback_bundle == True:
    1. Freeze all outbound asset purchases across Tier 1, Tier 2, and Tier 3.
    2. Size operational cash buffer using withdrawal_p975 + 25% contingency buffer.
    3. Assume zero new deposit inflows (Deposit = $0.00).
    4. Issue high-priority alert to treasury operations.
```

---

## 7. Concrete Integration Code Example

Below is a complete, runnable example demonstrating how Team 2 should consume a `ForecastBundle` and enforce conservative liquidity boundaries:

```python
from datetime import datetime, timezone
from reserve_juggler.contracts import ForecastBundle, DataQualityStatus

def optimize_tier1_repo_sweep(
    current_cash_usd: float,
    mandatory_reserve_usd: float,
    scheduled_obligations_usd: float,
    forecast_bundle: ForecastBundle,
    repo_window_open: bool,
) -> float:
    """Computes the safe dollar amount to sweep into overnight repo at 17:00 UTC."""
    # 1. Check Sweep Window Timing
    if not repo_window_open:
        return 0.0  # Cannot sweep outside 17:00-18:00 UTC window

    # 2. Check Data Quality & Fallback Status
    if forecast_bundle.is_fallback_bundle or forecast_bundle.data_quality_status != DataQualityStatus.OK:
        # Data is degraded or in fallback mode: strictly preserve cash
        return 0.0

    # 3. Retrieve 1-hour and overnight (15-hour to 08:00 maturity) forecast points
    pt_1h = forecast_bundle.get_point(horizon_hours=1)
    pt_24h = forecast_bundle.get_point(horizon_hours=24)

    if pt_1h is None or pt_24h is None:
        return 0.0

    # 4. Determine Conservative Outflow Requirements
    # If reliability is high, use P90; if medium or low, use extreme P95/P97.5
    if pt_1h.reliability_tier == "HIGH":
        expected_withdrawal_1h = pt_1h.withdrawal_p90
        expected_deposit_1h = pt_1h.deposit_p10
        overnight_stress_outflow = pt_24h.cumulative_net_outflow_p95
    else:
        # Conservative elevated buffer under lower reliability
        expected_withdrawal_1h = pt_1h.withdrawal_p975 or (pt_1h.withdrawal_p90 * 1.25)
        expected_deposit_1h = 0.0  # Disregard incoming deposits
        overnight_stress_outflow = pt_24h.cumulative_net_outflow_p99

    # 5. Calculate Immediate Liquidity Needs until Next Morning (08:00 UTC)
    immediate_cash_needed = (
        mandatory_reserve_usd
        + scheduled_obligations_usd
        + max(0.0, expected_withdrawal_1h - expected_deposit_1h)
        + max(0.0, overnight_stress_outflow)
    )

    # 6. Compute Safe Investable Surplus
    available_surplus = current_cash_usd - immediate_cash_needed

    return max(0.0, available_surplus)
```
