"""Run the Phase 5 host-side verification suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> int:
    print("$ " + " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-plant", action="store_true", help="also run the MATLAB plant vector check")
    args = parser.parse_args()

    checks = [
        [sys.executable, "-m", "unittest", "discover", "-s", "pi", "-p", "test_*.py"],
    ]
    if args.include_plant:
        checks.append([sys.executable, "pi/test_plant_parity.py"])
    for command in checks:
        if run_command(command):
            return 1
    print("phase5_verification=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
