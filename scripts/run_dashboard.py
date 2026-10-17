#!/usr/bin/env python3
"""
Reserve Juggler — Dashboard Entry Point Script.
Launches the integrated Streamlit user interface located at frontend/legacy_streamlit/app.py.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Reserve Juggler Treasury Management Dashboard.")
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to bind the Streamlit server (default: 8050)")
    parser.add_argument("--host", type=str, default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode without opening browser")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "frontend" / "legacy_streamlit" / "app.py"

    if not app_path.exists():
        sys.stderr.write(f"Error: Dashboard app not found at {app_path}\n")
        return 1

    # Ensure repository root and src are in PYTHONPATH
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{src_path}:{existing_pythonpath}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
    ]
    if args.headless:
        cmd.append("--server.headless=true")

    print(f"Launching Reserve Juggler Dashboard on http://{args.host}:{args.port}...")
    try:
        import subprocess
        return subprocess.run(cmd, env=env, cwd=str(repo_root)).returncode
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
