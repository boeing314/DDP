"""
Per-episode differential evolution (v4 algorithm) over the episode dataset.

For every episode in episodes/ (built by build_episodes.py) this runs its own
DE search for the dipf.* parameters listed in config.yaml -> episode_opt.bounds
and stores that episode's best parameters.

One evaluation = one simulation of the episode with a candidate parameter
vector: the captured SUMO state is loaded, the controller continues from the
captured ego state, and the run ends when the ego has covered `distance_m`
(default 150 m) from its start (success), collides, or reaches `max_time_s`
(timeout). The episode is deterministic, so a single run scores a candidate.
The score is the controller's episode cost C (computed over the segment),
lower is better.

DE per episode (identical to v4/v6 optimize.py):
  - log10 parameter space, Latin hypercube initial population
    (member 0 replaced by the v6 best parameters when include_baseline is
    true, so the result is never worse than the v6 parameters)
  - for each member i: DE/best/2 mutation v = best + F*(r1 - r2 + r3 - r4),
    clipped to the bounds; exponential crossover with CR; greedy selection
    (trial replaces member i iff its C is lower)
  - best member updated at the end of each generation
  - optional early stop: no improvement of the best C for `patience`
    generations

Episodes are optimised in parallel (one episode per worker process). Results
are written per episode, so the run can be stopped and restarted: episodes
that already have a result are skipped (use --rerun to redo them).

Output (episode_opt.results_dir, default episode_opt_results/):
    <episode_id>.json   baseline + best parameters/cost/outcome + per-generation history
    summary.csv         one row per finished episode (rebuilt at the end, or --summarize)

Usage:
    python optimize_episodes.py                         # all usable episodes
    python optimize_episodes.py --limit 20              # first 20 episodes (quick test)
    python optimize_episodes.py --match s004            # episodes whose id contains 's004'
    python optimize_episodes.py --pop-size 10 --n-generations 8 --n-jobs 8
    python optimize_episodes.py --verbose               # also print every candidate
    python optimize_episodes.py --summarize             # only rebuild summary.csv
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from capture_states import load_v6_params

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
EPI = CONFIG["episode_opt"]

BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in EPI["bounds"].items()}
PARAM_NAMES = list(BOUNDS)
D = len(PARAM_NAMES)
LOG_LO = np.array([math.log10(BOUNDS[p][0]) for p in PARAM_NAMES])
LOG_HI = np.array([math.log10(BOUNDS[p][1]) for p in PARAM_NAMES])
RESULT_KEYS = ("C", "C_collision", "C_dist", "C_time", "success", "final_progress", "min_ttc")


# =============================================================================
# Parameter <-> log space (as in v4)
# =============================================================================

def log_vec_to_params(v):
    return {p: float(10.0 ** v[i]) for i, p in enumerate(PARAM_NAMES)}


def params_to_log_vec(params):
    return np.clip(np.array([math.log10(params[p]) for p in PARAM_NAMES]), LOG_LO, LOG_HI)


def sample_lhs_log(n, rng):
    u = np.empty((n, D))
    for d in range(D):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return LOG_LO + u * (LOG_HI - LOG_LO)


def mutate_best_2(log_pop, i, best_idx, F, rng):
    candidates = [k for k in range(len(log_pop)) if k not in (i, best_idx)]
    r1, r2, r3, r4 = rng.choice(candidates, size=4, replace=False)
    v = log_pop[best_idx] + F * (log_pop[r1] - log_pop[r2] + log_pop[r3] - log_pop[r4])
    return np.clip(v, LOG_LO, LOG_HI)


def exponential_crossover(theta, v, CR, rng):
    u = theta.copy()
    j = int(rng.integers(0, D))
    k = 0
    while True:
        u[(j + k) % D] = v[(j + k) % D]
        k += 1
        if rng.random() >= CR or k == D:
            break
    return u


# =============================================================================
# Worker process setup
# =============================================================================

_PARENT_PID = None


def _init_worker(parent_pid):
    """
    Ctrl+C goes to the whole process group; only the main process should
    react to it (an interrupted evaluation left workers in a broken state).
    Workers also stop by themselves once the main process is gone.
    """
    global _PARENT_PID
    _PARENT_PID = parent_pid
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _check_parent_alive():
    if _PARENT_PID is not None and os.getppid() != _PARENT_PID:
        raise SystemExit("main optimize_episodes.py process is gone -- worker stopping")


# =============================================================================
# One evaluation = one simulation of the episode
# =============================================================================

def evaluate(ep_dir, params, s, scratch):
    _check_parent_alive()
    if not s["subprocess"]:
        return evaluate_in_process(ep_dir, params, s, scratch)
    env = os.environ.copy()
    env["DIPF_PARAMS"] = json.dumps(params)
    env["DIPF_OUTPUT_DIR"] = str(scratch)
    cmd = [sys.executable, str(CONTROLLER), "--episode", str(ep_dir),
           "--distance", str(s["distance"]), "--max-time", str(s["max_time"])]
    fail = {"C": math.inf, "C_collision": math.nan, "C_dist": math.nan, "C_time": math.nan,
            "success": 0.0, "final_progress": math.nan, "min_ttc": math.nan, "status": "error"}
    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, capture_output=True, text=True,
                              timeout=s["eval_timeout"])
    except subprocess.TimeoutExpired:
        return {**fail, "status": "eval_timeout"}
    mfile = scratch / f"results_{ep_dir.name}" / "metrics.json"
    if proc.returncode != 0 or not mfile.exists():
        return fail
    m = json.loads(mfile.read_text(encoding="utf-8"))
    mfile.unlink()
    out = {k: float(m[k]) if m.get(k) is not None else math.nan for k in RESULT_KEYS}
    if out["C"] != out["C"]:          # NaN cost (e.g. empty trajectory) -> worst
        out["C"] = math.inf
    out["status"] = m["status"]
    return out


def evaluate_in_process(ep_dir, params, s, scratch):
    """
    Same evaluation as the subprocess path, but inside this worker process:
    dipf_controller is imported once per worker and each call starts a fresh
    SUMO with a fast connect (verified to give byte-identical trajectories
    to the subprocess path; ~0.4 s instead of ~1.7 s per evaluation).
    """
    import dipf_controller as dc   # imported once per worker process
    fail = {"C": math.inf, "C_collision": math.nan, "C_dist": math.nan, "C_time": math.nan,
            "success": 0.0, "final_progress": math.nan, "min_ttc": math.nan, "status": "error"}
    try:
        m = dc.run_episode(ep_dir, params, s["distance"], s["max_time"], output_dir=scratch)
    except Exception as exc:
        print(f"[{Path(ep_dir).name}] evaluation error: {type(exc).__name__}: {exc}", flush=True)
        return fail
    out = {k: float(m[k]) if m.get(k) is not None else math.nan for k in RESULT_KEYS}
    if out["C"] != out["C"]:
        out["C"] = math.inf
    out["status"] = m["status"]
    return out


def fmt(r):
    return (f"C={r['C']:.4f} {r['status']:<9} prog={r['final_progress']:.3f} "
            f"[coll={r['C_collision']:.3f} dist={r['C_dist']:.3f} time={r['C_time']:.3f}]")


# =============================================================================
# DE for one episode
# =============================================================================

def optimize_episode(ep_dir, s, baseline_params):
    ep_dir = Path(ep_dir)
    ep_id = ep_dir.name
    tag = f"[{ep_id}]"
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng([s["rng_seed"], zlib.crc32(ep_id.encode())])
    pop_size, F, CR = s["pop_size"], s["F"], s["CR"]
    t0 = time.time()
    n_evals = 0

    print(f"{tag} start: seed={meta['seed']} {meta['capture_label']} t={meta['time']:.1f}s "
          f"x0={meta['start_x']:.1f} -> target {meta['target_x']:.1f} ({meta['segment_m']:.0f} m), "
          f"original outcome={meta['original_outcome']}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"epopt_{ep_id}_") as tmp:
        scratch = Path(tmp)

        # ---- initial population (LHS, member 0 = v6 baseline)
        log_pop = sample_lhs_log(pop_size, rng)
        if s["include_baseline"]:
            log_pop[0] = params_to_log_vec(baseline_params)
        # exact parameter values of each member (log10 -> 10** round trips
        # change the last digits, and outcomes can be that sensitive)
        pop_params = [dict(baseline_params) if (i == 0 and s["include_baseline"]) else log_vec_to_params(log_pop[i])
                      for i in range(pop_size)]
        results = []
        for i in range(pop_size):
            r = evaluate(ep_dir, pop_params[i], s, scratch)
            n_evals += 1
            results.append(r)
            if s["verbose"]:
                print(f"{tag}   init[{i:2d}] {fmt(r)}", flush=True)
        costs = np.array([r["C"] for r in results])
        baseline_result = results[0] if s["include_baseline"] else evaluate(ep_dir, baseline_params, s, scratch)
        if not s["include_baseline"]:
            n_evals += 1
        best_idx = int(np.argmin(costs))
        print(f"{tag} baseline (v6 params): {fmt(baseline_result)}", flush=True)
        print(f"{tag} gen  0/{s['n_generations']}: best {fmt(results[best_idx])}  "
              f"({pop_size} evals, {time.time() - t0:.0f}s)", flush=True)

        history = [{"generation": 0, "best_C": float(costs[best_idx]),
                    "best_success": results[best_idx]["success"], "accepted": pop_size}]
        no_improve = 0

        # ---- generations
        for gen in range(1, s["n_generations"] + 1):
            prev_best = costs[best_idx]
            accepted = 0
            for i in range(pop_size):
                v = mutate_best_2(log_pop, i, best_idx, F, rng)
                u = exponential_crossover(log_pop[i], v, CR, rng)
                trial = log_vec_to_params(u)
                r = evaluate(ep_dir, trial, s, scratch)
                n_evals += 1
                ok = r["C"] < costs[i]
                if ok:
                    log_pop[i], costs[i], results[i], pop_params[i] = u, r["C"], r, trial
                    accepted += 1
                if s["verbose"]:
                    print(f"{tag}   gen {gen} cand[{i:2d}] {fmt(r)} "
                          f"{'< parent -> ACCEPTED' if ok else '>= parent -> rejected'}", flush=True)
            best_idx = int(np.argmin(costs))
            history.append({"generation": gen, "best_C": float(costs[best_idx]),
                            "best_success": results[best_idx]["success"], "accepted": accepted})
            print(f"{tag} gen {gen:2d}/{s['n_generations']}: best {fmt(results[best_idx])}  "
                  f"accepted {accepted}/{pop_size}  ({time.time() - t0:.0f}s)", flush=True)

            no_improve = no_improve + 1 if costs[best_idx] >= prev_best - 1e-12 else 0
            if s["patience"] and no_improve >= s["patience"]:
                print(f"{tag} early stop: no improvement for {no_improve} generations", flush=True)
                break

    n_errors = sum(r["status"] == "error" for r in results + [baseline_result])
    if n_errors:
        # Results from failed simulations are not results: no file is
        # written, so the episode is redone on the next run.
        raise RuntimeError(f"{n_errors} evaluations failed with an error -- result not saved")

    best_params = pop_params[best_idx]
    best = results[best_idx]
    out = {
        "episode_id": ep_id,
        "meta": meta,
        "baseline": {"params": baseline_params, "result": baseline_result},
        "best": {"params": best_params, "result": best},
        "improvement_C": baseline_result["C"] - best["C"],
        "history": history,
        "n_evals": n_evals,
        "seconds": round(time.time() - t0, 1),
        "settings": {k: s[k] for k in ("pop_size", "n_generations", "F", "CR", "distance", "max_time",
                                       "include_baseline", "patience", "rng_seed")},
        "bounds": BOUNDS,
    }
    # write-then-rename: an interruption can never leave a half-written result
    final = Path(s["results_dir"]) / f"{ep_id}.json"
    tmp_file = final.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp_file, final)
    print(f"{tag} DONE in {out['seconds']:.0f}s, {n_evals} evals: baseline C={baseline_result['C']:.4f} "
          f"({baseline_result['status']}) -> best C={best['C']:.4f} ({best['status']})", flush=True)
    return out


# =============================================================================
# Summary
# =============================================================================

def write_summary(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("s*_c*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        b, bl, m = r["best"], r["baseline"]["result"], r["meta"]
        rows.append({
            "episode_id": r["episode_id"], "seed": m["seed"], "capture_label": m["capture_label"],
            "time": m["time"], "segment_m": m["segment_m"], "original_outcome": m["original_outcome"],
            "baseline_C": bl["C"], "baseline_status": bl["status"],
            "best_C": b["result"]["C"], "best_status": b["result"]["status"],
            "best_progress": b["result"]["final_progress"], "improvement_C": r["improvement_C"],
            "n_evals": r["n_evals"], "seconds": r["seconds"],
            **{f"p_{k}": v for k, v in b["params"].items()},
        })
    if not rows:
        return None
    path = Path(results_dir) / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"Summary: {n} episodes | baseline success {sum(r['baseline_status'] == 'success' for r in rows)}/{n} "
          f"-> optimised success {sum(r['best_status'] == 'success' for r in rows)}/{n} | "
          f"mean C {np.mean([r['baseline_C'] for r in rows if math.isfinite(r['baseline_C'])]):.3f} -> "
          f"{np.mean([r['best_C'] for r in rows if math.isfinite(r['best_C'])]):.3f}")
    print(f"         {path}")
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", default=str(BASE_DIR / EPI.get("episodes_dir", "episodes")))
    ap.add_argument("--results-dir", default=str(BASE_DIR / EPI.get("results_dir", "episode_opt_results")))
    ap.add_argument("--match", default=None, help="only episodes whose id contains this text")
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (after --match)")
    ap.add_argument("--pop-size", type=int, default=int(EPI.get("pop_size", 10)))
    ap.add_argument("--n-generations", type=int, default=int(EPI.get("n_generations", 10)))
    ap.add_argument("--F", type=float, default=float(EPI.get("F", 0.5)))
    ap.add_argument("--CR", type=float, default=float(EPI.get("CR", 0.9)))
    ap.add_argument("--patience", type=int, default=EPI.get("patience"))
    ap.add_argument("--n-jobs", type=int, default=int(EPI.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--rerun", action="store_true", help="redo episodes that already have a result")
    ap.add_argument("--verbose", action="store_true", help="print every candidate evaluation")
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary.csv and exit")
    ap.add_argument("--subprocess", action="store_true",
                    help="evaluate each candidate in a new Python process (slower; default: in-process)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        write_summary(results_dir)
        return
    if args.pop_size < 6:
        raise SystemExit("pop_size must be >= 6 for DE/best/2")

    episodes_dir = Path(args.episodes_dir)
    index = list(csv.DictReader((episodes_dir / "index.csv").open(encoding="utf-8")))
    eps = [r["episode_id"] for r in index if r["usable"] == "True"]
    n_unusable = len(index) - len(eps)
    if args.match:
        eps = [e for e in eps if args.match in e]
    if args.limit:
        eps = eps[:args.limit]
    # Resume check: a result is only "done" if it is readable and none of
    # its stored evaluations failed; anything else is moved to invalid/ and
    # that episode is redone. Leftover *.tmp files (interrupted writes) are
    # removed.
    for tmp_file in results_dir.glob("*.json.tmp"):
        tmp_file.unlink()
    n_invalid = 0
    for f in results_dir.glob("s*_c*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            ok = "error" not in (r["baseline"]["result"]["status"], r["best"]["result"]["status"])
        except Exception:
            ok = False
        if not ok:
            (results_dir / "invalid").mkdir(exist_ok=True)
            os.replace(f, results_dir / "invalid" / f.name)
            n_invalid += 1
    if n_invalid:
        print(f"Moved {n_invalid} invalid result file(s) to {results_dir / 'invalid'} -- those episodes will be redone")
    done = {f.stem for f in results_dir.glob("s*_c*.json")}
    todo = eps if args.rerun else [e for e in eps if e not in done]

    v6_dir = (BASE_DIR / CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve()
    baseline_params, source = load_v6_params(v6_dir)
    missing = set(PARAM_NAMES) - set(baseline_params)
    if missing:
        raise SystemExit(f"v6 parameters lack {sorted(missing)} -- episode_opt.bounds must match v6's tuned keys")
    baseline_params = {k: float(baseline_params[k]) for k in PARAM_NAMES}

    s = {
        "pop_size": args.pop_size, "n_generations": args.n_generations, "F": args.F, "CR": args.CR,
        "patience": args.patience, "distance": float(EPI.get("distance_m", 150.0)),
        "max_time": float(EPI.get("max_time_s", 60.0)), "eval_timeout": float(EPI.get("eval_timeout_s", 300.0)),
        "include_baseline": bool(EPI.get("include_baseline", True)), "rng_seed": int(EPI.get("rng_seed", 0)),
        "verbose": args.verbose, "results_dir": str(results_dir), "subprocess": args.subprocess,
    }
    evals_per_ep = args.pop_size * (1 + args.n_generations)

    print("=" * 100)
    print("v7 per-episode differential evolution")
    print(f"Episodes     : {len(index)} in {episodes_dir}  ({n_unusable} unusable: < min segment)")
    print(f"Selected     : {len(eps)}  | already done: {len(eps) - len(todo)}  | to run now: {len(todo)}")
    print(f"Segment      : {s['distance']:g} m from the ego's start in each state, timeout {s['max_time']:g} s")
    print(f"DE           : pop {args.pop_size}, {args.n_generations} generations, F={args.F}, CR={args.CR}, "
          f"patience={args.patience}, baseline in init={s['include_baseline']}")
    print(f"Evaluations  : {evals_per_ep} per episode (max) -> {evals_per_ep * len(todo):,} total (max)")
    print(f"Parallelism  : {args.n_jobs} episodes at a time  "
          f"({'one new Python process per evaluation' if args.subprocess else 'in-process evaluation, fresh SUMO each'})")
    print(f"Baseline     : {source}")
    print("Search space (log-uniform):")
    for p in PARAM_NAMES:
        print(f"  {p:<16} [{BOUNDS[p][0]}, {BOUNDS[p][1]}]   v6 value {baseline_params[p]:.6g}")
    print(f"Results      : {results_dir}")
    print("=" * 100, flush=True)

    t_start = time.time()
    n_done, n_fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker, initargs=(os.getpid(),)) as ex:
        futs = {ex.submit(optimize_episode, episodes_dir / e, s, baseline_params): e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                fut.result()
                n_done += 1
            except Exception as exc:
                n_fail += 1
                print(f"[{e}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            el = time.time() - t_start
            fin = n_done + n_fail
            eta = el / fin * (len(todo) - fin)
            print(f"===== progress {fin}/{len(todo)} episodes ({n_fail} failed) | elapsed {el / 60:.1f} min | "
                  f"ETA {eta / 60:.1f} min =====", flush=True)

    print("=" * 100)
    print(f"Finished {n_done} episodes ({n_fail} failed) in {(time.time() - t_start) / 60:.1f} min")
    write_summary(results_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
