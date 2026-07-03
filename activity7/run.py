"""
run.py — Entry point for Activity 7.
Runs quadrant.py and displays results cleanly.
Usage: python run.py
"""

import subprocess
import sys
import os


def main():
    print("=" * 55)
    print("  Activity 7 — Qdrant Chunking Strategies Demo")
    print("=" * 55)

    # Check .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        print("\n  .env found ✓")
    else:
        print("\n  WARNING: .env not found. Using defaults (localhost:6333)")

    print("\n  Running quadrant.py...\n")
    print("-" * 55)

    result = subprocess.run(
        [sys.executable, "quadrant.py"],
        cwd=os.path.dirname(__file__),
        text=True
    )

    print("-" * 55)
    if result.returncode == 0:
        print("\n  Done! Activity 7 completed successfully.")
    else:
        print(f"\n  Error (exit code {result.returncode}). Check output above.")


if __name__ == "__main__":
    main()
