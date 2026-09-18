"""
Batch-runs dipf_controller.py once per seed, from --start to --end
(inclusive), each as its own subprocess so every run gets a clean
TraCI/SUMO connection. No manual per-run intervention needed.

Each run writes to results/results_<seed>/ (trajectory.csv, metrics.json) --
see save_results() in dipf_controller.py. This script does not touch
config.yaml; it overrides the seed via the DIPF_SEED environment variable,
which dipf_controller.py reads in preference to config.yaml's
simulation.seed when set.

Usage:
    python3 run_seeds.py                # seeds 0..99
    python3 run_seeds.py --start 0 --end 9
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="first seed, inclusive (default 0)")
    parser.add_argument("--end", type=int, default=99, help="last seed, inclusive (default 99)")
    args = parser.parse_args()

    seeds = range(args.start, args.end + 1)
    total = len(seeds)

    failed = []

    for i, seed in enumerate(seeds, start=1):
        print("=" * 90)
        print(f"[{i}/{total}] Running seed={seed}")
        print("=" * 90)

        env = os.environ.copy()
        env["DIPF_SEED"] = str(seed)

        result = subprocess.run(
            [sys.executable, str(CONTROLLER)],
            cwd=str(BASE_DIR),
            env=env,
        )

        if result.returncode != 0:
            print(f"[WARNING] seed={seed} exited with code {result.returncode}")
            failed.append(seed)

    print("\n" + "=" * 90)
    print(f"BATCH COMPLETE: {total - len(failed)}/{total} seeds finished cleanly")
    if failed:
        print(f"Failed seeds: {failed}")
    print("=" * 90)


if __name__ == "__main__":
    main()
