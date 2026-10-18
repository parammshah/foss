#!/usr/bin/env python3
"""
Reserve Juggler — Unified API & FOSS UI Server Launcher.
Launches the FastAPI backend and serves the FOSS UI on http://localhost:8089.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Reserve Juggler Integrated API & Platform.")
    parser.add_argument("--port", "-p", type=int, default=8089, help="Port to bind the server (default: 8089)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "src"

    if not src_path.exists():
        sys.stderr.write(f"Error: src directory not found at {src_path}\n")
        return 1

    # Ensure src and repo_root are in sys.path and PYTHONPATH
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(repo_root))

    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    os.environ["RESERVE_JUGGLER_MODE"] = "SYNTHETIC_DEMO"
    os.environ["RESERVE_JUGGLER_USE_SYNTHETIC"] = "true"

    try:
        import uvicorn
        from reserve_juggler.api_server import app

        print("================================================================================")
        print("          RESERVE JUGGLER: UNIFIED BACKEND API & FOSS UI SERVER                 ")
        print("================================================================================")
        print(f"Platform Command UI:  http://localhost:{args.port}/platform.html")
        print(f"Landing Page:         http://localhost:{args.port}/index.html")
        print(f"Swagger API Docs:     http://localhost:{args.port}/docs")
        print(f"ReDoc Specification:  http://localhost:{args.port}/redoc")
        print("--------------------------------------------------------------------------------")
        print("Subsystems Connected:")
        print("  * Team 1: Time-Series Data, Ledger & Conformal Quantile Forecasting")
        print("  * Team 2: Convex Optimizer, Statutory Tax Engine & Asset Allocation")
        print("  * Team 3: Macro-Panic Sentinel, Explainability Rationale & Self-Learning")
        print("================================================================================")

        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        return 0
    except Exception as e:
        sys.stderr.write(f"Failed to start server: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
