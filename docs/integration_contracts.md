# Reserve Juggler — Cross-Team Integration Contracts

## Canonical Single Source of Truth: `reserve_juggler.contracts`

All cross-team interactions must adhere strictly to the typed contracts defined in `reserve_juggler.contracts`.

### 1. Team 1 Inbound Contracts (Ledger & Data/Forecasting)
- **`TreasuryState`**:
  - `timestamp: str` (ISO 8601 UTC)
  - `available_cash: float`
  - `asset_positions: List[AssetPosition]`
  - `obligations: List[Obligation]`
  - `authorized_credit_availability: float`
  - `protected_reserve_floor: Optional[float] = None` (Must be explicitly configured; never defaults to 0.0)
  - `core_capital_pct: float = 0.70`
  - `hot_capital_pct: float = 0.30`
  - `source_status: ServiceStatus`

- **`ForecastBundle` & `ForecastPoint`**:
  - `horizon: str` ("1h", "6h", "24h", "48h", "72h")
  - `target_timestamp: str` (Future timestamp of observation window)
  - `expected_inflow: float`, `expected_outflow: float`, `expected_net_flow: float`
  - `uncertainty_p05`, `uncertainty_p50`, `uncertainty_p90`, `uncertainty_p95`, `uncertainty_p99`
  - `confidence_score: float`

### 2. Team 2 Outbound/Inbound Contracts (Treasury Decision & Risk)
- **`DecisionRecord`**:
  - `decision_id: str`
  - `selected_action: str`
  - `target_asset: Optional[str]`
  - `amount: float`
  - `treasury_safety_score: Optional[float] = None` (Explicitly None if unsupplied)
  - `hard_gate_outcomes: List[HardGateOutcome]` (Status must be `GateStatus.PASS`, `FAIL`, `UNKNOWN`, or `NOT_EVALUATED`)
  - `approval_status: str`
  - `estimated_costs: Dict[str, float]`

- **`ExecutionOutcome`**:
  - `outcome_id: str`
  - `realized_amount: float`
  - `realized_costs: Dict[str, float]`
  - `settlement_status: str` ("SETTLED", "SIMULATED_SETTLED", "CANCELLED", "REJECTED", "PENDING")

### 3. Team 3 Outbound Intelligence (Macro-Panic Sentinel)
- **`RiskSignal`**:
  - `signal_id: str`
  - `category: str` (MONETARY_POLICY, BANKING_STRESS, MARKET_SHOCK, etc.)
  - `severity: str` (LOW, MEDIUM, HIGH, CRITICAL, LOW_CONFIDENCE)
  - `confidence: float` (0.0 to 1.0)
  - `relevance_to_treasury: float` (Calculated dynamically against active portfolio asset classes)
  - `warnings: List[str]` (Data quality flags, invalid timestamp notices)

### 4. Telemetry Loop (`LearningEvent`)
- All components publish auditable events with `event_id`, `versioned_event_type`, `timestamp`, `correlation_id`, `payload`, and `associated_versions`.
