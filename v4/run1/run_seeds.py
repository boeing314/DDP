"""
Batch-runs dipf_controller.py once per seed, from --start to --end
(inclusive), in parallel across multiple CPU cores. Each seed is its own
subprocess (clean TraCI/SUMO connection per run), submitted to a
ProcessPoolExecutor pool sized by --n-jobs.

Unlike optimize.py, this does NOT search over parameters -- it runs the
CURRENT config.yaml's dipf.* values against a batch of seeds. Useful as a
quick baseline/sanity check (e.g. before and after an optimize.py run) to
see the average cost C and success rate for a specific parameter set.

Each run writes to results/results_<seed>/ (trajectory.csv, metrics.json) --
see save_results() in dipf_controller.py. This script does not touch
config.yaml; it overrides the seed via the DIPF_SEED environment variable.

Usage:
    python3 run_seeds.py                       # seeds 0..99, n_jobs=cpu_count()-1
    python3 run_seeds.py --start 0 --end 19 --n-jobs 8
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"


def run_one_seed(seed):
    env = os.environ.copy()
    env["DIPF_SEED"] = str(seed)

    result = subprocess.run(
        [sys.executable, str(CONTROLLER)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    metrics_file = BASE_DIR / "results" / f"results_{seed}" / "metrics.json"
    status, c = None, None
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            m = json.load(f)
        status, c = m.get("status"), m.get("C")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return seed, result.returncode, status, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="first seed, inclusive (default 0)")
    parser.add_argument("--end", type=int, default=99, help="last seed, inclusive (default 99)")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() - 3),
                         help="parallel SUMO instances (default: cpu_count()-1)")
    args = parser.parse_args()

    seeds = list(range(args.start, args.end + 1))
    total = len(seeds)

    print("=" * 90)
    print(f"Running {total} seeds ({args.start}..{args.end}) across {args.n_jobs} parallel workers")
    print("=" * 90)

    failed = []
    results = {}

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(run_one_seed, seed): seed for seed in seeds}
        done = 0
        for fut in as_completed(futures):
            seed, returncode, status, c = fut.result()
            done += 1
            results[seed] = (status, c)

            if returncode != 0:
                failed.append(seed)
                print(f"[{done}/{total}] seed={seed:<4} FAILED (exit code {returncode})")
            else:
                c_str = f"{c:.4f}" if c is not None else "n/a"
                print(f"[{done}/{total}] seed={seed:<4} status={status:<10} C={c_str}")

    costs = [c for (_, c) in results.values() if c is not None]
    successes = sum(1 for (status, _) in results.values() if status == "success")

    print("\n" + "=" * 90)
    print(f"BATCH COMPLETE: {total - len(failed)}/{total} seeds finished cleanly")
    if failed:
        print(f"Failed seeds: {sorted(failed)}")
    print(f"Success rate: {successes}/{total} ({100 * successes / total:.1f}%)")
    if costs:
        print(f"Average C   : {sum(costs) / len(costs):.4f}  (over {len(costs)} runs with a valid C)")
    print("=" * 90)


if __name__ == "__main__":
    main()
