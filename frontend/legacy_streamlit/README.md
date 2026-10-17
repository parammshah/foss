# Reserve Juggler — Legacy Streamlit Frontend

This directory contains the legacy prototype Streamlit user interface for Reserve Juggler (Team 3).

## Architecture & Decoupling Notice

The core financial, event monitoring, explainability, and self-learning engines of Reserve Juggler are isolated within the Python package `reserve_juggler` under `src/`.

**The frontend is decoupled from the backend:**
- All communication with the backend passes through the `ApplicationFacade` (`reserve_juggler.integration.facade.ApplicationFacade`) or typed domain contracts (`reserve_juggler.contracts`).
- No business logic, safety enforcement, persistence, or optimization algorithms reside in this directory.
- A future frontend (React, Next.js, FastAPI web app, or desktop client) can replace this directory entirely without modifying any code inside `src/reserve_juggler/`.

## Running the Legacy Interface

### Prerequisites
Install the frontend requirements:
```bash
pip install -r requirements-frontend.txt
```

### Launch Command
From the project root:
```bash
streamlit run frontend/legacy_streamlit/app.py
```

### Configuration Environment Variables
- `RESERVE_JUGGLER_MODE`: Set to `SYNTHETIC_DEMO` (default) for illustrative simulation, or `PRODUCTION` to connect to canonical external services.
- `RESERVE_JUGGLER_DB_PATH`: Path to SQLite event store (defaults to `var/reserve_juggler.db` or `reserve_juggler.db`).

## Page Navigation Overview
1. **Overview**: Executive liquidity summary, Future-Protected Capital Firewall status, Hot/Core Capital breakdown.
2. **Forecasts & Reliability**: Cash inflow/outflow predictions, WAPE, interval coverage, and chronological drift.
3. **Portfolio**: Multi-asset holdings across Cash, Repo, Sovereign Bonds, and Equities.
4. **Next Best Action**: Recommended liquidity decisions, cost breakdown, and authorized human override actions.
5. **Stress Lab**: Multi-asset comparative stress tests across All-Cash, Greedy Gross-Yield, and Adaptive Reserve Juggler.
6. **Macro-Panic Sentinel**: Real-time financial intelligence feed, corroborated signals, and macro stress score.
7. **System Learning**: Governance lifecycle, candidate model validation reports, and audited promotion/rollback.
8. **Decision Log**: Complete immutable audit trail of past treasury decisions and human workflows.
