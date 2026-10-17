# Reserve Juggler — Team 3 Architecture & Integration Guide
## Macro-Panic Sentinel, Global Self-Learning, Explainability, Human Oversight, and Dashboard

**Branch:** `feature/team-3-intelligence-oversight`  
**Schema & API Version:** `1.0.0`

---

## 1. Executive Summary & Architecture Overview

Reserve Juggler is an enterprise autonomous treasury management platform designed to predict liquidity demand, safeguard mandatory regulatory reserves, select financially feasible actions, and optimize investment yields across cash, repo, sovereign bonds, and equities.

Team 3 delivers the intelligence, oversight, and human-facing layers of the system:
1. **Macro-Panic Sentinel (`events/`):** External financial-event monitoring, deduplication, and market-risk interpretation.
2. **System-Wide Self-Learning Feedback Loop (`learning/`):** End-to-end telemetry across all subsystems, prediction-to-outcome matching, drift detection, and safety-gated candidate update promotion/rollback.
3. **Explainable AI & Human Oversight (`explainability/`):** Grounded mathematical decision rationale, alternative cost comparison, and role-based authorization workflows.
4. **Interactive Dashboard (`dashboard/`):** Streamlit application implementing all 8 specified pages with glassmorphism design, real-time stress testing, and governance auditing.
5. **Cross-Team Facade & Adapters (`integration/`):** Clean integration layer exposing unified services to Teams 1 and 2 without altering their private financial logic.

```mermaid
graph TD
    subgraph External_Feeds["External Intelligence Feeds"]
        RSS[Authorized Central Bank / News RSS]
        SYN[Synthetic Event Lab Mode]
    end

    subgraph Team_3_Events["Macro-Panic Sentinel (Team 3)"]
        DEDUP[Event Deduplicator & Echo Clustering]
        INTERP[Event Risk Interpreter]
        SENTINEL[MacroPanicSentinel Service]
    end

    subgraph Cross_Team_Contracts["Shared Cross-Team Contracts"]
        T_STATE[TreasuryState (Team 1)]
        F_BUNDLE[ForecastBundle (Team 1)]
        R_SIGNAL[RiskSignal (Team 3)]
        D_RECORD[DecisionRecord (Team 2)]
        E_OUTCOME[ExecutionOutcome (Team 2 / Simulated)]
        L_EVENT[LearningEvent (System-Wide)]
    end

    subgraph Team_3_Explainability["Explainability & Human Gate (Team 3)"]
        EXPLAIN[DecisionExplanationService]
        WORKFLOW[Role-Based Approval / Override Engine]
    end

    subgraph Team_3_Learning["Global Self-Learning Loop (Team 3)"]
        STORE[(Append-Only Event Store)]
        MATCHER[Outcome Matcher]
        EVALUATOR[System Performance Evaluator & Drift]
        GOV[Safety Invariants Validator & Rollback Registry]
    end

    subgraph UI_Layer["Dashboard & Demo (Team 3)"]
        DASH[Streamlit Executive Dashboard (8 Pages)]
        FACADE[ApplicationFacade API]
    end

    RSS --> DEDUP
    SYN --> DEDUP
    DEDUP --> INTERP
    T_STATE --> INTERP
    INTERP --> SENTINEL
    SENTINEL --> R_SIGNAL

    R_SIGNAL --> D_RECORD
    T_STATE --> D_RECORD
    F_BUNDLE --> D_RECORD

    D_RECORD --> EXPLAIN
    EXPLAIN --> WORKFLOW
    WORKFLOW --> E_OUTCOME

    L_EVENT --> STORE
    D_RECORD -. Telemetry .-> L_EVENT
    F_BUNDLE -. Telemetry .-> L_EVENT
    E_OUTCOME -. Telemetry .-> L_EVENT
    WORKFLOW -. Telemetry .-> L_EVENT

    STORE --> MATCHER
    MATCHER --> EVALUATOR
    EVALUATOR --> GOV

    FACADE --> DASH
```

---

## 2. File Ownership & Boundary Compliance

As mandated by project rules, Team 3 exclusively owns:
- `src/reserve_juggler/events/`
- `src/reserve_juggler/learning/`
- `src/reserve_juggler/explainability/`
- `src/reserve_juggler/dashboard/`
- `src/reserve_juggler/integration/`
- `tests/integration/`

Team 3 **does not modify** Team 1's canonical data/ledger/forecasting modules or Team 2's risk/decision/optimization/tax modules. Instead, the `integration/adapters.py` wrapper intercepts telemetry and standardizes protocols.

---

## 3. Public Service Interfaces & Contracts

### 3.1 Macro-Panic Sentinel
- **Endpoint:** `MacroPanicSentinel.evaluate(as_of_timestamp: str, treasury_state: Optional[TreasuryState]) -> SentinelEvaluationResult`
- **Output:**
  - `active_signals`: List of `RiskSignal` records.
  - `aggregate_macro_stress_score`: Float `[0.0, 100.0]`.
  - `recommended_stress_scenarios`: Tags e.g., `["DEPOSIT_RUN_SIM", "REPO_SQUEEZE_100BP", "RATES_UP_50BP"]`.
  - `mode`: `"LIVE"` or `"SYNTHETIC_LAB"`.
  - `system_status`: `"OK"` or `"DEGRADED_OFFLINE"`.

### 3.2 System Learning Service
- **Endpoint 1:** `LearningService.record(event: LearningEvent) -> None`
  - Records versioned inputs, outputs, decisions, or execution results.
- **Endpoint 2:** `LearningService.evaluate_system_performance(start: Optional[str], end: Optional[str]) -> SystemPerformanceReport`
  - Computes cross-lifecycle metrics: MAPE, RMSE, interval coverage %, cost slippage in basis points, shortfall counts, drift detection flag, and system health status.
- **Endpoint 3:** `LearningService.propose_update(...) -> CandidateUpdate`
  - Validates candidate models or parameters against hard safety invariants.
- **Endpoint 4:** `LearningService.promote_candidate(candidate_id, approver, approver_role, justification) -> (bool, str)`
  - Requires human authorization with `HEAD_OF_TREASURY` or `RISK_OFFICER` role.
- **Endpoint 5:** `LearningService.rollback_component(component, target_version, author, approver_role, justification) -> (bool, str)`
  - Reverts component to prior approved version in the audit registry.

### 3.3 Decision Explanation Service
- **Endpoint:** `DecisionExplanationService.explain(decision_record: DecisionRecord, forecast_bundle: Optional[ForecastBundle], treasury_state: Optional[TreasuryState]) -> DecisionExplanation`
  - Produces grounded, non-hallucinatory explanations containing cost breakdowns, hard safety gate statuses, alternative cost deltas, and timing/settlement constraints.
- **Workflow Endpoint:** `DecisionExplanationService.execute_human_workflow(...)`
  - Enforces role-based permissions (`AUDITOR`, `TREASURY_ANALYST`, `RISK_OFFICER`, `HEAD_OF_TREASURY`) and executes simulated transactions.

### 3.4 Application Facade
- **Endpoint:** `ApplicationFacade.get_dashboard_snapshot(as_of_timestamp: Optional[str] = None) -> DashboardSnapshot`
  - Provides a single versioned snapshot consumed by UI and API callers.

---

## 4. Macro-Panic Sentinel & Event-Risk Interpretation

### Modular Ingestion & Fault Tolerance
The Sentinel supports:
- **Synthetic Mode (`SyntheticEventSource`):** Provides realistic scenario injections (`NORMAL`, `BANKING_STRESS`, `RATE_HIKE_SHOCK`, `SOVEREIGN_DEBT_DISRUPTION`) enabling full testing without live network dependencies.
- **Authorized RSS/API Mode (`RSSFeedSource`):** Robust parser with rate-limiting, secure environment variable token support (`*_API_KEY`), configurable timeouts, and graceful offline fallback.
- **Deduplication Engine (`EventDeduplicator`):** Token-based Jaccard similarity clustering identifies original source announcements versus syndicated echo coverage, preventing double-counting while measuring reporting velocity.

### Event-Risk Interpretation
- **Relevance to Treasury Holdings:** Calculates exposure weights against the actual portfolio (e.g., rate shocks directly weight against 2Y and 10Y sovereign bonds; repo spreads weight against repo holdings).
- **Separation of Sentiment from Objective Risk:** High negative sentiment on unverified reports does not trigger critical asset liquidation. Alerts instead widen uncertainty assumptions and flag recommended stress tests.
- **Confidence Calibration:** Accounts for source credibility (e.g., Central Bank = 0.98, news wire = 0.90, social rumor = 0.35) and applies exponential decay to stale reports (>48h).

---

## 5. System-Wide Self-Learning & Safety Invariants

### Complete Lifecycle Telemetry
The feedback loop spans:
1. Data Ingestion & Quality -> 2. ML / OR Multi-Horizon Forecasts -> 3. Forecast Interval Calibration -> 4. Treasury Safety Score & Reserves -> 5. News & Event Alerts -> 6. Hot/Core Capital Classification -> 7. Asset Eligibility & Settlement Constraints -> 8. Tax-Adjusted Return Calculations -> 9. Next Best Action & Alternatives -> 10. Simulated or Realized Outcomes -> 11. Human Approvals, Overrides & Modifications.

### Permitted Candidate Changes
- Forecast recalibration and model weights.
- Validated ensemble combining weights.
- Scenario-generation stress parameters.
- Event-detection thresholds.
- Validated risk-model calibration.
- Non-mandatory optimizer parameters inside approved safe ranges.
- Decision-explanation and escalation policies.

### Strict Negative Constraints (Enforced by `SafetyInvariantValidator`)
Autonomous machine learning is **strictly prohibited** from:
1. **Lowering or removing mandatory liquidity reserves** below the statutory floor (`MandatoryReserveFloorGate`).
2. **Bypassing credit limits or asset eligibility criteria**.
3. **Modifying statutory tax rules or schedules**.
4. **Disabling hard safety constraints**.
5. **Self-approving high-risk policy updates without human authorization**.
6. **Promoting candidates that increase liquidity shortfalls** under stress testing (even if predictive accuracy improves).
7. **Fabricating improvement when data is insufficient**; reports `INCONCLUSIVE` when sample size is below threshold.

---

## 6. Streamlit Dashboard Architecture

The dashboard (`src/reserve_juggler/dashboard/app.py`) provides 8 integrated pages:

1. **🏛️ Overview:** Top metric tiles (Operating Cash, Protected Reserve Floor, Eligible Surplus, Treasury Safety Score), active Next Best Action summary, Macro-Panic Sentinel status, and interactive Core vs Hot Capital distribution charts.
2. **📈 Forecasts and Reliability:** Multi-horizon cash flow trajectory (1h, 6h, 24h, 48h, 72h) with 90% uncertainty intervals (`[p05, p95]`), actual-versus-predicted error tracking, MAPE/RMSE metrics, and calibration drift indicators.
3. **💼 Portfolio:** Multi-asset holding cards across cash, overnight repo, 30-day repo, 2Y/10Y sovereign bonds, and equities. Interactive comparison of Marked-to-Market value versus accessible intraday liquidity after haircut discounts and settlement latency (`T+0`, `T+1`, `T+2`).
4. **🎯 Next Best Action:** Selected allocation action and rejected alternatives table with cost deltas. Complete grounded mathematical rationale. Interactive human-in-the-loop oversight form supporting approval, modification, and override with role authentication.
5. **🔬 Stress Lab:** Interactive slider-based stress simulator testing simultaneous deposit withdrawal spikes, sovereign bond yield jumps (+bps), repo settlement delays, and bridge-credit freezes against the reserve floor with waterfall liquidity visualization.
6. **📡 Macro-Panic Sentinel:** Live/synthetic feed intelligence, active risk signals table, credibility ratings, exposure relevance %, and an interactive custom event injection panel.
7. **🧠 System Learning:** Lifecycle health gauges, active model/policy version matrix, candidate updates in shadow mode, candidate promotion controls, and audited version rollback tools.
8. **📜 Decision Log:** Comprehensive chronological audit trail of algorithmic recommendations, human authorizations, simulated execution receipts, and regulatory governance disclaimers.

---

## 7. Startup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Dependencies installed via `requirements-part3.txt`:
  ```bash
  pip install -r requirements-part3.txt
  ```

### Running the Dashboard
```bash
# Set PYTHONPATH to include src
$env:PYTHONPATH="src"   # PowerShell
# export PYTHONPATH=src # Linux/macOS

# Launch Streamlit app
streamlit run src/reserve_juggler/dashboard/app.py
```

### Running the Test Suite
```bash
$env:PYTHONPATH="src"
python -m pytest tests/integration -v -p no:cacheprovider
```

---

## 8. Measured Test Results

The test suite contains 18 comprehensive integration tests with 100% pass rate:

| Test File | Test Case | Target Capability | Result |
| :--- | :--- | :--- | :--- |
| `test_sentinel_and_events.py` | `test_sentinel_evaluates_synthetic_mode_successfully` | Offline synthetic event mode execution | **PASSED** |
| `test_sentinel_and_events.py` | `test_event_deduplication_and_clustering` | Headline deduplication & echo clustering | **PASSED** |
| `test_sentinel_and_events.py` | `test_event_risk_interpretation_schema_and_relevance` | RiskSignal contract & exposure relevance | **PASSED** |
| `test_sentinel_and_events.py` | `test_stale_event_confidence_decay` | Confidence decay on stale reports | **PASSED** |
| `test_sentinel_and_events.py` | `test_rss_feed_failure_resilience` | Network fault tolerance & graceful fallback | **PASSED** |
| `test_system_learning.py` | `test_event_store_persistence_and_correlation` | LearningEvent store indexing by correlation ID | **PASSED** |
| `test_system_learning.py` | `test_predictions_matched_to_subsequent_outcomes` | OutcomeMatcher cost & shortfall pairing | **PASSED** |
| `test_system_learning.py` | `test_safety_invariants_reject_mandatory_reserve_reduction` | Invariant: reject reserve floor lowering | **PASSED** |
| `test_system_learning.py` | `test_safety_invariants_reject_candidate_increasing_shortfall` | Invariant: reject candidate with shortfalls | **PASSED** |
| `test_system_learning.py` | `test_safety_invariants_reject_insufficient_sample_size` | Invariant: reject fabricated improvements | **PASSED** |
| `test_system_learning.py` | `test_candidate_promotion_and_authorized_rollback` | Role authorization & version rollback | **PASSED** |
| `test_explainability_and_oversight.py` | `test_decision_explanation_grounded_completeness` | Mathematical grounding without hallucination | **PASSED** |
| `test_explainability_and_oversight.py` | `test_human_workflow_role_permissions` | Role permissions (`AUDITOR`, `ANALYST`, etc.) | **PASSED** |
| `test_explainability_and_oversight.py` | `test_human_workflow_modification` | Amount modification before simulated execution | **PASSED** |
| `test_end_to_end_facade.py` | `test_end_to_end_event_to_decision_to_learning_trace` | Full event-to-decision-to-learning trace | **PASSED** |
| `test_end_to_end_facade.py` | `test_facade_scenario_switching` | Operational scenario switching | **PASSED** |
| `test_end_to_end_facade.py` | `test_stress_lab_waterfall_and_shortfall_calculation` | Shock simulator & waterfall calculation | **PASSED** |
| `test_dashboard_views.py` | `test_dashboard_snapshot_schema` | UI snapshot completeness & schema integrity | **PASSED** |

---

## 9. Known Limitations & Regulatory Governance Notice

1. **Simulated Execution Default:** For demonstration and risk security, the human authorization workflow connects strictly to a simulated execution adapter. It does not route live settlement instructions to real SWIFT, Fedwire, or banking accounts.
2. **External Network Policy:** External RSS and news feeds are disabled by default unless explicitly configured with valid credentials in environment variables (`*_API_KEY`). The system operates autonomously using calibrated synthetic scenarios.
3. **Statutory Compliance Disclaimer:** This platform provides internal risk governance, mathematical explainability, and operational intelligence. It does **not** constitute legal regulatory certification under Basel III, Dodd-Frank, SEC Rule 15c3-1, or FINRA requirements. Institutional deployment requires formal regulatory review and jurisdiction-specific validation.
