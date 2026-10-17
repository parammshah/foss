# Reserve Juggler Team 3 — System Architecture

## Overview
Team 3 provides the **Intelligence, Governance, Explainability, and Oversight** layer of Reserve Juggler.

```
                          ┌───────────────────────────┐
                          │ External Intelligence Feeds│
                          │   (Fed, ECB, News, Wires) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │      Macro-Panic Sentinel     │
                        │ (Deduplication, Interp, Risk) │
                        └───────────────┬───────────────┘
                                        │ Risk Signals
                                        ▼
┌──────────────────────┐  State & Forecasts  ┌──────────────────────────────┐
│ Team 1 (Forecasting  ├────────────────────►│    Application Facade        │
│ & Ledger Core)       │                     │  (Central Orchestration API) │
└──────────────────────┘                     └──────────────┬───────────────┘
                                                            │
┌──────────────────────┐  Optimized Decision                │
│ Team 2 (Decisions,   ├────────────────────────────────────┤
│ Risk & Policy Core)  │                                    │
└──────────────────────┘                                    ▼
                                             ┌──────────────────────────────┐
                                             │   Decision Explainability    │
                                             │    & Human Oversight API     │
                                             └──────────────┬───────────────┘
                                                            │ Actions & Audits
                                                            ▼
                                             ┌──────────────────────────────┐
                                             │    Global Self-Learning      │
                                             │ (Matching, Metrics, Gov, DB) │
                                             └──────────────┬───────────────┘
                                                            │
                                                            ▼
                                             ┌──────────────────────────────┐
                                             │      Replaceable Client      │
                                             │   (legacy_streamlit / React) │
                                             └──────────────────────────────┘
```

## Key Subsystems

1. **Macro-Panic Sentinel (`reserve_juggler.events`)**:
   - RFC 2822 / ISO 8601 publication timestamp normalization (UTC).
   - Bounded event deduplication with TTL cache.
   - Reporting velocity ($v$) and acceleration ($a = \Delta v / \Delta t$).
   - Portfolio exposure relevance calculation.
   - Separation of news sentiment from actual financial severity.

2. **System-Wide Self-Learning Engine (`reserve_juggler.learning`)**:
   - Durable SQLite event and governance store.
   - Idempotent prediction-to-outcome matching.
   - Statistical metrics: WAPE, MAE, empirical prediction interval coverage, KS-test two-sample drift detection.
   - Candidate evaluation engine running genuine holdout comparisons and executed multi-asset stress testing.
   - Typed runtime registry and transactional candidate promotion with automatic rollback upon persistence failure.

3. **Explainability & Human Oversight (`reserve_juggler.explainability`)**:
   - Structured decision rationale tree and cost breakdowns.
   - Grounded hard-gate evaluation without false `PASS` defaults.
   - Action-specific revalidation (reserve floor, credit line limits, asset availability).
   - Zero phantom transactions for rejected/cancelled proposals.

4. **Stress Lab Simulation (`reserve_juggler.integration.stress_engine`)**:
   - Multi-asset double-entry balance sheet reconciliation without double-counting cash or principal.
   - Execution of repo liquidity delay and credit freeze constraints.
   - Comparative benchmarks: All-Cash, Greedy Gross-Yield, Adaptive Reserve Juggler.
