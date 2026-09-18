"""
Differential Evolution (DE) optimization driver for the Pure-DIPF
controller's obstacle/lane shaping parameters (v4).

Implements "Algorithm 1: Scenario-Specific Parameter Optimization" as given:

    1. Initialize population {theta_i}_{i=1}^{Np} of parameter vectors
       (here: Latin Hypercube Sampling, in log-space -- see design note
       below).
    2. For each generation, for each candidate theta_i:
         Mutation (DE/best/2):
             v_i = theta_best + F*(theta_r1 - theta_r2 + theta_r3 - theta_r4)
             with r1,r2,r3,r4 four distinct population indices, none equal
             to i or to the current best.
         Crossover (exponential):
             u_i starts as a copy of theta_i. Pick a random start index j.
             Copy v_i's value into u_i at (j+k) mod D for k=0,1,2,...,
             stopping as soon as a random draw is >= CR, or after D
             dimensions have been copied.
         Evaluate J(u_i): run every seed in the fixed seed list through
             dipf_controller.py with u_i's parameters and average the
             resulting cost C -- this *is* the pseudocode's inner
             "for each test case s / for each time step" simulate-and-score
             loop; dipf_controller.py's ego_control_step() +
             compute_episode_cost() already implement exactly that
             (generate the potential field, take -grad(U) as the force,
             integrate position, accumulate a per-scenario cost, average
             over test cases).
         Selection: theta_i <- u_i iff J(u_i) < J(theta_i), else unchanged.
             Each population member's cost is cached from when it was last
             (re)computed, so an unchanged parent is never re-simulated.
    3. Return theta* = the population member with the lowest cached cost.

Tunes: alpha, tau_y_base, sx, delta_lane, lambda_obs.
tau_x_base is held fixed at its config.yaml value (not tuned). All other
dipf.* parameters are left exactly as config.yaml has them.

Design notes / deviations from the bare pseudocode (which leaves these
unspecified):
  - Log-space parameterization: these 5 parameters span up to 7 orders of
    magnitude (per the project's stated bounds), and DE's difference-vector
    mutation is not scale-invariant -- in linear space, the largest-range
    parameters (tau_y_base/sx/delta_lane/lambda_obs, up to 10000) would
    dominate every mutation step while alpha (0-100) barely moves. All DE
    arithmetic (mutation, crossover, the LHS initialization) is therefore
    done on log10(theta); parameters are only exponentiated back to real
    values right before being handed to the simulator. This mirrors the
    log-uniform search space already used in v3's Bayesian optimizer.
    lambda_obs in particular is a multiplicative gain on the obstacle-
    avoidance term, so a factor-of-10 change matters just as much near the
    bottom of its (0, 10000) range as near the top -- linear sampling would
    spend almost its entire budget above 1000 and barely touch the 0-10
    region where the currently calibrated value (lambda_obs=1) lives.
  - Bound handling: trial vectors are clipped back into [log(lo), log(hi)]
    after mutation, since raw DE mutation can wander outside the box and
    the pseudocode doesn't otherwise constrain v_i.
  - "Test cases" = seeds: a fixed seed list (common random numbers) is
    reused for every evaluation throughout the whole run, so differences in
    measured cost reflect real parameter differences rather than random
    traffic variation between evaluations.
  - Checkpointing: DE has no notion of "resume" in the bare pseudocode, but
    a full run here can be very expensive (see the cost note below), so
    the population/cost state is checkpointed to de_checkpoint.npz after
    the initial population and after every generation; --resume continues
    from that file instead of reinitializing.

Cost note: unlike v3's Bayesian search (~100 trials x 100 seeds = 10,000
episodes), DE evaluates Np NEW trial vectors per generation, each costing
n_seeds episodes. Total cost is roughly

    n_seeds * pop_size * (1 + n_generations)

episodes (the "+1" is the one-time initial-population evaluation). With the
defaults below (pop_size=15, n_generations=20, n_seeds=100) that is already
~31,500 episode runs -- scale --pop-size / --n-generations / --n-seeds down
for a faster, noisier search.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"

PARAM_NAMES = ["alpha", "tau_y_base", "sx", "delta_lane", "lambda_obs"]
ENV_NAMES = {
    "alpha": "DIPF_ALPHA",
    "tau_y_base": "DIPF_TAU_Y_BASE",
    "sx": "DIPF_SX",
    "delta_lane": "DIPF_DELTA_LANE",
    "lambda_obs": "DIPF_LAMBDA_OBS",
}
# (low, high), per the project's stated search ranges. Lower bound nudged
# off 0 since log-space can't represent exactly 0 (and 0 isn't a meaningful
# value for any of these multiplicative gain/scale parameters anyway).
BOUNDS = {
    "alpha": (1e-3, 100.0),
    "tau_y_base": (1e-3, 10000.0),
    "sx": (1e-3, 10000.0),
    "delta_lane": (1e-3, 10000.0),
    "lambda_obs": (1e-3, 10000.0),
}
D = len(PARAM_NAMES)
LOG_LO = np.array([np.log10(BOUNDS[p][0]) for p in PARAM_NAMES])
LOG_HI = np.array([np.log10(BOUNDS[p][1]) for p in PARAM_NAMES])


# =============================================================================
# Parameter <-> log-space helpers
# =============================================================================

def log_vec_to_params(log_vec):
    return {p: float(10.0 ** log_vec[i]) for i, p in enumerate(PARAM_NAMES)}


def clip_log(vec):
    return np.clip(vec, LOG_LO, LOG_HI)


def sample_lhs_log(n_samples, rng):
    """Latin Hypercube Sampling in log-space -- one stratified, independently
    permuted stratum per dimension, per standard LHS construction."""
    u = np.empty((n_samples, D))
    for d in range(D):
        perm = rng.permutation(n_samples)
        u[:, d] = (perm + rng.random(n_samples)) / n_samples
    return LOG_LO + u * (LOG_HI - LOG_LO)


# =============================================================================
# Scenario evaluation (the pseudocode's "for each test case s" loop)
# =============================================================================

COST_COMPONENTS = ("C", "C_collision", "C_dist", "C_time")


def run_one_seed(params, seed, scratch_root):
    """Runs one dipf_controller.py subprocess for one seed with the given
    parameter overrides; returns that episode's full cost breakdown (all
    components NaN on failure)."""
    out_dir = scratch_root / f"seed_{seed}"

    env = os.environ.copy()
    env["DIPF_SEED"] = str(seed)
    env["DIPF_OUTPUT_DIR"] = str(out_dir)
    for pname, env_name in ENV_NAMES.items():
        env[env_name] = repr(params[pname])

    result = subprocess.run(
        [sys.executable, str(CONTROLLER)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    nan_result = {k: float("nan") for k in COST_COMPONENTS}

    if result.returncode != 0:
        return nan_result

    metrics_file = out_dir / f"results_{seed}" / "metrics.json"
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            m = json.load(f)
        return {k: float(m.get(k, float("nan"))) for k in COST_COMPONENTS}
    except (FileNotFoundError, json.JSONDecodeError):
        return nan_result


def evaluate(params, seeds, n_jobs, scratch_base, tag):
    """
    J(u_i): average cost over every test case (seed) -- pseudocode lines
    14-26 (Initialize total cost / for each test case / Average Cost).
    Also averages the C_collision/C_dist/C_time components that make up C,
    for reporting the split-wise cost breakdown alongside the total.
    """
    trial_scratch = scratch_base / tag
    try:
        per_seed = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(run_one_seed, params, seed, trial_scratch): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                per_seed.append(fut.result())
    finally:
        shutil.rmtree(trial_scratch, ignore_errors=True)

    means = {}
    for k in COST_COMPONENTS:
        valid = [r[k] for r in per_seed if r[k] == r[k]]  # drop NaNs
        means[k] = (sum(valid) / len(valid)) if valid else float("inf")

    n_failed = sum(1 for r in per_seed if r["C"] != r["C"])
    return means, n_failed


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(path, generation, log_pop, costs, breakdowns, rng):
    np.savez(
        path,
        generation=generation,
        log_pop=log_pop,
        costs=costs,
        cost_collision=breakdowns["C_collision"],
        cost_dist=breakdowns["C_dist"],
        cost_time=breakdowns["C_time"],
    )

    # np.savez can't cleanly round-trip a Generator's state dict; store it
    # alongside as plain JSON so --resume also reproduces the optimizer's
    # own random draws exactly (mutation/crossover/LHS), not just the
    # population.
    state_path = path.with_suffix(".rngstate.json")
    state = rng.bit_generator.state
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)


def load_checkpoint(path, rng):
    data = np.load(path)
    generation = int(data["generation"])
    log_pop = data["log_pop"]
    costs = data["costs"]
    breakdowns = {
        "C_collision": data["cost_collision"],
        "C_dist": data["cost_dist"],
        "C_time": data["cost_time"],
    }

    state_path = path.with_suffix(".rngstate.json")
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state["state"]["state"] = int(state["state"]["state"])
        state["state"]["inc"] = int(state["state"]["inc"])
        rng.bit_generator.state = state

    return generation, log_pop, costs, breakdowns


# =============================================================================
# DE core: mutation (DE/best/2) + exponential crossover + greedy selection
# =============================================================================

def mutate_best_2(log_pop, i, best_idx, F, rng):
    excluded = {i, best_idx}
    candidates = [k for k in range(len(log_pop)) if k not in excluded]
    r1, r2, r3, r4 = rng.choice(candidates, size=4, replace=False)
    v_i = log_pop[best_idx] + F * (log_pop[r1] - log_pop[r2] + log_pop[r3] - log_pop[r4])
    return clip_log(v_i)


def exponential_crossover(theta_i, v_i, CR, rng):
    u_i = theta_i.copy()
    j = int(rng.integers(0, D))
    k = 0
    while True:
        idx = (j + k) % D
        u_i[idx] = v_i[idx]
        k += 1
        if rng.random() >= CR or k == D:
            break
    return u_i


def log_eval(log_path, **kwargs):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kwargs) + "\n")


def format_breakdown(means):
    return (f"[collision={means['C_collision']:.4f} "
            f"dist={means['C_dist']:.4f} "
            f"time={means['C_time']:.4f}]")


def differential_evolution(pop_size, n_generations, F, CR, seeds, n_jobs,
                            scratch_base, log_path, checkpoint_path, rng,
                            resume):
    if pop_size < 6:
        raise ValueError(
            "pop_size must be >= 6 for DE/best/2 (needs i, best, and 4 "
            "other distinct population members)"
        )

    start_gen = 0
    if resume and checkpoint_path.exists():
        start_gen, log_pop, costs, breakdowns = load_checkpoint(checkpoint_path, rng)
        print(f"Resumed from checkpoint at generation {start_gen} "
              f"(best C so far: {np.min(costs):.4f})")
    else:
        log_pop = sample_lhs_log(pop_size, rng)
        costs = np.full(pop_size, np.inf)
        breakdowns = {k: np.full(pop_size, np.inf) for k in ("C_collision", "C_dist", "C_time")}

        print(f"Evaluating initial population ({pop_size} members)...")
        for i in range(pop_size):
            params = log_vec_to_params(log_pop[i])
            t0 = time.time()
            means, n_failed = evaluate(params, seeds, n_jobs, scratch_base, f"init_{i}")
            costs[i] = means["C"]
            for k in ("C_collision", "C_dist", "C_time"):
                breakdowns[k][i] = means[k]
            log_eval(log_path, generation=0, member=i, params=params,
                      cost=means["C"], cost_collision=means["C_collision"],
                      cost_dist=means["C_dist"], cost_time=means["C_time"],
                      accepted=True, n_failed=n_failed, is_initial=True,
                      seconds=round(time.time() - t0, 1))
            print(f"  init pop[{i}] C={means['C']:.4f} {format_breakdown(means)}  "
                  f"({time.time() - t0:.1f}s, {n_failed}/{len(seeds)} seeds failed)")

        save_checkpoint(checkpoint_path, 0, log_pop, costs, breakdowns, rng)

    best_idx = int(np.argmin(costs))

    for gen in range(start_gen + 1, n_generations + 1):
        print(f"\n=== Generation {gen}/{n_generations} "
              f"(best so far: C={costs[best_idx]:.4f}, member {best_idx}) ===")

        for i in range(pop_size):
            v_i = mutate_best_2(log_pop, i, best_idx, F, rng)
            u_i = exponential_crossover(log_pop[i], v_i, CR, rng)

            params = log_vec_to_params(u_i)
            t0 = time.time()
            means, n_failed = evaluate(params, seeds, n_jobs, scratch_base, f"g{gen}_i{i}")
            elapsed = time.time() - t0
            trial_cost = means["C"]

            parent_cost = costs[i]
            accepted = bool(trial_cost < parent_cost)

            log_eval(log_path, generation=gen, member=i, params=params,
                      cost=trial_cost, cost_collision=means["C_collision"],
                      cost_dist=means["C_dist"], cost_time=means["C_time"],
                      accepted=accepted, n_failed=n_failed,
                      is_initial=False, parent_cost=float(parent_cost),
                      seconds=round(elapsed, 1))

            if accepted:
                log_pop[i] = u_i
                costs[i] = trial_cost
                for k in ("C_collision", "C_dist", "C_time"):
                    breakdowns[k][i] = means[k]
                print(f"  cand[{i}] C={trial_cost:.4f} {format_breakdown(means)} "
                      f"< parent {parent_cost:.4f} -> ACCEPTED  ({elapsed:.1f}s)")
            else:
                print(f"  cand[{i}] C={trial_cost:.4f} {format_breakdown(means)} "
                      f">= parent {parent_cost:.4f} -> rejected  ({elapsed:.1f}s)")

        best_idx = int(np.argmin(costs))
        save_checkpoint(checkpoint_path, gen, log_pop, costs, breakdowns, rng)

    return log_pop, costs, breakdowns, best_idx


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pop-size", type=int, default=15,
                         help="Np, population size (default 15, min 6)")
    parser.add_argument("--n-generations", type=int, default=20,
                         help="number of generations (default 20)")
    parser.add_argument("--F", type=float, default=0.5, help="DE differential weight (default 0.5)")
    parser.add_argument("--CR", type=float, default=0.9, help="DE crossover probability (default 0.9)")
    parser.add_argument("--n-seeds", type=int, default=100,
                         help="test cases per evaluation, seeds 0..n_seeds-1 (default 100)")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() - 1),
                         help="parallel SUMO instances per evaluation (default: cpu_count-1)")
    parser.add_argument("--rng-seed", type=int, default=0,
                         help="seed for the OPTIMIZER's own randomness (mutation/crossover/LHS), "
                              "distinct from the simulation traffic seeds (default 0)")
    parser.add_argument("--resume", action="store_true",
                         help="resume from de_checkpoint.npz if present, instead of restarting")
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    log_path = BASE_DIR / "optimization_log.jsonl"
    checkpoint_path = BASE_DIR / "de_checkpoint.npz"
    scratch_base = BASE_DIR / "de_scratch"
    scratch_base.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.rng_seed)

    total_episodes = args.n_seeds * args.pop_size * (1 + args.n_generations)

    print("=" * 90)
    print("DIPF v4 differential evolution optimization")
    print(f"Population size (Np): {args.pop_size}")
    print(f"Generations         : {args.n_generations}")
    print(f"F / CR              : {args.F} / {args.CR}")
    print(f"Seeds/evaluation    : {args.n_seeds} (0..{args.n_seeds - 1}, fixed across the whole run)")
    print(f"Parallelism         : {args.n_jobs} concurrent SUMO instances")
    print(f"Estimated total episode runs: ~{total_episodes:,}")
    print("Search space (log-uniform):")
    for pname in PARAM_NAMES:
        lo, hi = BOUNDS[pname]
        print(f"  {pname:<12} [{lo}, {hi}]")
    print(f"Log        : {log_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print("=" * 90)

    try:
        log_pop, costs, breakdowns, best_idx = differential_evolution(
            args.pop_size, args.n_generations, args.F, args.CR, seeds,
            args.n_jobs, scratch_base, log_path, checkpoint_path, rng,
            args.resume,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- last checkpoint on disk still reflects the most recent "
              "completed generation. Re-run with --resume to continue.")
        raise
    finally:
        shutil.rmtree(scratch_base, ignore_errors=True)

    best_params = log_vec_to_params(log_pop[best_idx])
    best_breakdown = {k: float(breakdowns[k][best_idx]) for k in ("C_collision", "C_dist", "C_time")}

    print("\n" + "=" * 90)
    print("OPTIMIZATION COMPLETE")
    print(f"Best C      : {costs[best_idx]:.4f}  "
          f"[collision={best_breakdown['C_collision']:.4f} "
          f"dist={best_breakdown['C_dist']:.4f} "
          f"time={best_breakdown['C_time']:.4f}]")
    print("Best params :")
    for k, v in best_params.items():
        print(f"  {k:<12} = {v:.6g}")
    print("=" * 90)

    best_file = BASE_DIR / "best_params.json"
    with best_file.open("w", encoding="utf-8") as f:
        json.dump({
            "best_C": float(costs[best_idx]),
            "best_C_breakdown": best_breakdown,
            "best_params": best_params,
        }, f, indent=2)
    print(f"\nBest parameters written to: {best_file}")


if __name__ == "__main__":
    main()
