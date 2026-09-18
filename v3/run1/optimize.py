"""
Bayesian optimization driver for the Pure-DIPF controller's obstacle/lane
shaping parameters (v3).

Tunes:
    alpha, tau_y_base, sx, delta_lane, lambda_obs

to minimize the average episode cost C (see dipf_controller.compute_episode_cost)
across a FIXED batch of seeds -- default 0..99, per the project's request to
optimize over the average C across 100 seeds.

tau_x_base is held fixed at its config.yaml value (not tuned). All other
dipf.* parameters (sy, w_dest, w_obs, w_lane, dv_max_front,
obs_radius_ahead/behind, lateral_band, grad_eps, y_margin) are left exactly
as config.yaml has them.

Design notes:
  - Uses Optuna's TPE sampler (Bayesian-ish, sample-efficient, tolerant of a
    noisy objective) rather than grid/random search -- 5 continuous
    dimensions with an expensive-to-evaluate objective is squarely in BO's
    comfort zone.
  - Common random numbers (CRN): every trial is evaluated against the exact
    same fixed seed list. This isolates differences in the measured
    objective to real differences between parameter vectors rather than
    random traffic variation, which is what lets the optimizer converge
    with a reasonable number of trials.
  - Search space is log-uniform for all 5 parameters, including lambda_obs
    over (0, 10000): that range spans 7 orders of magnitude, and lambda_obs
    is a multiplicative gain on the obstacle-avoidance term, so a factor-of-
    10 change (e.g. 1 -> 10) matters just as much near the bottom of the
    range as near the top. A linear search would spend almost its entire
    budget above 1000 and barely sample the 0-10 region, even though that's
    where the currently calibrated value (lambda_obs=1) lives. Log-uniform
    can't represent exactly 0, so the lower bound is nudged to 1e-3
    (effectively "obstacle term off") -- if lambda_obs=0 itself needs to be
    tested, run it as one manual trial outside the sweep.
  - Each trial's N seed runs are executed as independent dipf_controller.py
    subprocesses (matching the existing DIPF_SEED pattern), in parallel via
    a process pool, each pointed at its own DIPF_OUTPUT_DIR so concurrent
    seeds/trials never collide writing to the same results_<seed>/ path.
    That scratch directory is deleted after its metrics.json is read, so a
    100-seed x many-trial sweep doesn't leave thousands of result folders
    on disk -- the full optimization history is kept instead in
    optuna_study.db (resumable) and optimization_log.jsonl (one line per
    trial, human-readable).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import optuna

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"

# (env_var_name, low, high) -- all sampled log-uniform. Bounds per the
# project's stated search ranges; lower bound nudged off 0 since a log
# scale can't represent exactly 0 (and 0 isn't a meaningful value for any
# of these multiplicative gain/scale parameters anyway).
PARAM_SPACE = {
    "alpha":       ("DIPF_ALPHA",       1e-3, 100.0),
    "tau_y_base":  ("DIPF_TAU_Y_BASE",  1e-3, 10000.0),
    "sx":          ("DIPF_SX",          1e-3, 10000.0),
    "delta_lane":  ("DIPF_DELTA_LANE",  1e-3, 10000.0),
    "lambda_obs":  ("DIPF_LAMBDA_OBS",  1e-3, 10000.0),
}


def run_one_seed(params, seed, scratch_root):
    """Runs one dipf_controller.py subprocess for one seed with the given
    parameter overrides, returns that episode's C (float, NaN on failure)."""
    out_dir = scratch_root / f"seed_{seed}"

    env = os.environ.copy()
    env["DIPF_SEED"] = str(seed)
    env["DIPF_OUTPUT_DIR"] = str(out_dir)
    for pname, (env_name, _, _) in PARAM_SPACE.items():
        env[env_name] = repr(params[pname])

    result = subprocess.run(
        [sys.executable, str(CONTROLLER)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        return float("nan")

    metrics_file = out_dir / f"results_{seed}" / "metrics.json"
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("C", float("nan")))
    except (FileNotFoundError, json.JSONDecodeError):
        return float("nan")


def evaluate(params, seeds, n_jobs, trial_scratch_dir):
    """Runs all seeds for one parameter vector, returns (mean_C, per_seed_C)."""
    costs = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(run_one_seed, params, seed, trial_scratch_dir): seed
            for seed in seeds
        }
        for fut in as_completed(futures):
            costs.append(fut.result())

    valid = [c for c in costs if c == c]  # drop NaNs (c==c is False for NaN)
    mean_c = sum(valid) / len(valid) if valid else float("inf")
    return mean_c, costs


def make_objective(seeds, n_jobs, scratch_base, log_path):
    def objective(trial):
        params = {
            pname: trial.suggest_float(pname, low, high, log=True)
            for pname, (_, low, high) in PARAM_SPACE.items()
        }

        trial_scratch_dir = scratch_base / f"trial_{trial.number}"
        try:
            mean_c, per_seed = evaluate(params, seeds, n_jobs, trial_scratch_dir)
        finally:
            shutil.rmtree(trial_scratch_dir, ignore_errors=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trial": trial.number,
                "params": params,
                "mean_C": mean_c,
                "n_seeds": len(seeds),
                "n_failed": sum(1 for c in per_seed if c != c),
            }) + "\n")

        return mean_c

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100,
                         help="number of Optuna trials (default 60)")
    parser.add_argument("--n-seeds", type=int, default=100,
                         help="seeds per trial, 0..n_seeds-1 (default 100)")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() - 3),
                         help="parallel SUMO instances per trial (default: cpu_count-1)")
    parser.add_argument("--study-name", type=str, default="dipf_v3_opt")
    parser.add_argument("--storage", type=str,
                         default=f"sqlite:///{BASE_DIR / 'optuna_study.db'}",
                         help="Optuna storage URL (sqlite by default -- resumable)")
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    log_path = BASE_DIR / "optimization_log.jsonl"

    scratch_base = Path(tempfile.mkdtemp(prefix="dipf_opt_", dir=str(BASE_DIR)))

    print("=" * 90)
    print("DIPF v3 parameter optimization")
    print(f"Trials     : {args.n_trials}")
    print(f"Seeds/trial: {args.n_seeds} (0..{args.n_seeds - 1}, fixed across all trials)")
    print(f"Parallelism: {args.n_jobs} concurrent SUMO instances")
    print(f"Search space (log-uniform):")
    for pname, (_, low, high) in PARAM_SPACE.items():
        print(f"  {pname:<12} [{low}, {high}]")
    print(f"Study      : {args.study_name} @ {args.storage}")
    print(f"Log        : {log_path}")
    print("=" * 90)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    try:
        study.optimize(
            make_objective(seeds, args.n_jobs, scratch_base, log_path),
            n_trials=args.n_trials,
        )
    finally:
        shutil.rmtree(scratch_base, ignore_errors=True)

    print("\n" + "=" * 90)
    print("OPTIMIZATION COMPLETE")
    print(f"Best mean_C : {study.best_value:.4f}")
    print("Best params :")
    for k, v in study.best_params.items():
        print(f"  {k:<12} = {v:.6g}")
    print("=" * 90)

    best_file = BASE_DIR / "best_params.json"
    with best_file.open("w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\nBest parameters written to: {best_file}")


if __name__ == "__main__":
    main()
