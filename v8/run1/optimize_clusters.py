"""
Step 3 -- one differential-evolution search per situation cluster.

For every cluster (cluster_assignment.csv) this finds ONE parameter set that
maximises the average success over all of that cluster's TRAINING episodes:

    J(theta) = (1 - success_rate) + 1e-4 * mean_C        (lower is better)

  success_rate / mean_C over the cluster's training episodes (x `futures`
  random futures each), every episode simulated from its saved state for
  cluster_opt.distance_m metres (in-process, fresh SUMO per run).

DE (same algorithm as v4/v6/v7): log10 space, Latin hypercube start (member 0
= v6 parameters), DE/best/2 mutation, exponential crossover, greedy selection,
best updated per generation, optional early stop (patience).

tau_x_base is NOT tuned: it stays at dipf.tau_x_base in config.yaml (1.0).
The 7 tuned keys and their bounds are in cluster_opt.bounds.

After each cluster, its best set and v6 are also scored on the cluster's
VALIDATION episodes (not used by the search).

Outputs (cluster_opt.results_dir, default cluster_opt_results/):
    cluster_<j>.json     best params, train/val success vs v6, history
    cluster_params.json  everything cluster_controller.py needs (written at the end;
                         clusters with too few training episodes fall back to v6, and so do
                         clusters whose optimised set does worse than v6 on their VALIDATION
                         episodes -- cluster_opt.validation_gate)

Resumable: clusters that already have cluster_<j>.json are skipped (--rerun to redo).

Usage:
    python optimize_clusters.py
    python optimize_clusters.py --clusters 0 3 7          # only these clusters
    python optimize_clusters.py --pop-size 8 --n-generations 6 --n-jobs 7
    python optimize_clusters.py --collect                 # only (re)write cluster_params.json
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
OPT = CONFIG["cluster_opt"]
SIT = CONFIG["situations"]
BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in OPT["bounds"].items()}
PARAMS = list(BOUNDS)
D = len(PARAMS)
LOG_LO = np.array([math.log10(BOUNDS[p][0]) for p in PARAMS])
LOG_HI = np.array([math.log10(BOUNDS[p][1]) for p in PARAMS])
if "tau_x_base" in BOUNDS:
    raise SystemExit("tau_x_base must not be in cluster_opt.bounds (it is fixed in v8)")


# =============================================================================
# DE helpers (v4 algorithm, log10 space)
# =============================================================================

def to_params(v):
    return {p: float(10.0 ** v[i]) for i, p in enumerate(PARAMS)}


def to_log(params):
    return np.clip(np.array([math.log10(params[p]) for p in PARAMS]), LOG_LO, LOG_HI)


def lhs(n, rng):
    u = np.empty((n, D))
    for d in range(D):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return LOG_LO + u * (LOG_HI - LOG_LO)


def mutate(pop, i, best, F, rng):
    r1, r2, r3, r4 = rng.choice([k for k in range(len(pop)) if k not in (i, best)], size=4, replace=False)
    return np.clip(pop[best] + F * (pop[r1] - pop[r2] + pop[r3] - pop[r4]), LOG_LO, LOG_HI)


def crossover(theta, v, CR, rng):
    u = theta.copy()
    j, k = int(rng.integers(0, D)), 0
    while True:
        u[(j + k) % D] = v[(j + k) % D]
        k += 1
        if rng.random() >= CR or k == D:
            return u


# =============================================================================
# Evaluation of one parameter set on a group of episodes
# =============================================================================

def _worker_init():
    sys.argv = [sys.argv[0]]


def _run(job):
    ep_id, params, sumo_seed = job
    import dipf_controller as dc
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_episode(BASE_DIR / SIT.get("episodes_dir", "episodes") / ep_id, params,
                               float(OPT["distance_m"]), float(OPT["max_time_s"]), output_dir=tmp,
                               sumo_seed=sumo_seed)
            C = float(m["C"]) if m["C"] == m["C"] else 30.0
            return m["status"], C
        except Exception:
            return "error", 30.0


def evaluate(pool, params, episodes, futures):
    seeds = [None] if futures <= 1 else list(range(1, futures + 1))
    jobs = [(e, params, sd) for e in episodes for sd in seeds]
    res = list(pool.map(_run, jobs, chunksize=1))
    st = [r[0] for r in res]
    succ = st.count("success") / len(st)
    meanC = float(np.mean([r[1] for r in res]))
    return {"J": (1.0 - succ) + 1e-4 * meanC, "success": succ, "mean_C": meanC,
            "n_runs": len(res), "errors": st.count("error")}


# =============================================================================
# One cluster
# =============================================================================

def optimize_cluster(pool, j, train_eps, val_eps, v6, args, results_dir):
    tag = f"[cluster {j}]"
    rng = np.random.default_rng([int(OPT.get("rng_seed", 0)), j])
    futures = int(OPT.get("futures", 1))
    t0 = time.time()
    print(f"\n{tag} {len(train_eps)} training / {len(val_eps)} validation episodes, "
          f"{len(train_eps) * futures} simulations per candidate", flush=True)

    v6_tuned = {p: v6[p] for p in PARAMS}                       # tau_x from config (1.0)
    base = evaluate(pool, v6_tuned, train_eps, futures)
    print(f"{tag} v6 params (tau_x=1): train success {base['success']:.1%}, mean C {base['mean_C']:.3f}", flush=True)

    pop = lhs(args.pop_size, rng)
    pop_params = [to_params(v) for v in pop]
    if OPT.get("include_baseline", True):
        pop[0], pop_params[0] = to_log(v6_tuned), dict(v6_tuned)
    scores = [base if (k == 0 and OPT.get("include_baseline", True)) else evaluate(pool, pop_params[k], train_eps, futures)
              for k in range(args.pop_size)]
    J = np.array([s["J"] for s in scores])
    best = int(np.argmin(J))
    history = [{"generation": 0, "best_J": float(J[best]), "best_success": scores[best]["success"]}]
    print(f"{tag} gen  0: best train success {scores[best]['success']:.1%} (J {J[best]:.4f})  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    stale = 0
    for gen in range(1, args.n_generations + 1):
        prev = J[best]
        acc = 0
        for i in range(args.pop_size):
            u = crossover(pop[i], mutate(pop, i, best, args.F, rng), args.CR, rng)
            trial = to_params(u)
            s = evaluate(pool, trial, train_eps, futures)
            if s["J"] < J[i]:
                pop[i], pop_params[i], scores[i], J[i] = u, trial, s, s["J"]
                acc += 1
        best = int(np.argmin(J))
        history.append({"generation": gen, "best_J": float(J[best]), "best_success": scores[best]["success"],
                        "accepted": acc})
        print(f"{tag} gen {gen:2d}: best train success {scores[best]['success']:.1%} (J {J[best]:.4f}), "
              f"accepted {acc}/{args.pop_size}  [{time.time() - t0:.0f}s]", flush=True)
        stale = stale + 1 if J[best] >= prev - 1e-12 else 0
        if args.patience and stale >= args.patience:
            print(f"{tag} early stop: no improvement for {stale} generations", flush=True)
            break

    best_params = pop_params[best]
    val_best = evaluate(pool, best_params, val_eps, futures) if val_eps else None
    val_v6 = evaluate(pool, v6_tuned, val_eps, futures) if val_eps else None
    if val_eps:
        print(f"{tag} validation: best {val_best['success']:.1%} vs v6 {val_v6['success']:.1%} "
              f"({len(val_eps)} episodes)", flush=True)
    out = {"cluster": j, "n_train": len(train_eps), "n_val": len(val_eps),
           "best_params": {**best_params, "tau_x_base": float(CONFIG["dipf"]["tau_x_base"])},
           "train": {"best": scores[best], "v6": base}, "val": {"best": val_best, "v6": val_v6},
           "history": history, "seconds": round(time.time() - t0, 1),
           "settings": {"distance_m": OPT["distance_m"], "max_time_s": OPT["max_time_s"], "futures": futures,
                        "pop_size": args.pop_size, "n_generations": args.n_generations, "F": args.F, "CR": args.CR}}
    tmp = results_dir / f"cluster_{j}.json.tmp"
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, results_dir / f"cluster_{j}.json")
    print(f"{tag} DONE in {out['seconds']:.0f}s", flush=True)
    return out


# =============================================================================
# Collect
# =============================================================================

def collect(results_dir, k, v6):
    fallback = {**{p: v6[p] for p in PARAMS}, "tau_x_base": float(CONFIG["dipf"]["tau_x_base"])}
    clusters, rows = {}, []
    for j in range(k):
        f = results_dir / f"cluster_{j}.json"
        r = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        gated = (r is not None and OPT.get("validation_gate", True) and r["val"]["best"] is not None
                 and r["val"]["best"]["success"] < r["val"]["v6"]["success"])
        if r is not None:
            clusters[str(j)] = {"params": fallback if gated else r["best_params"],
                                "source": "v6 (validation gate)" if gated else "optimised",
                                "optimised_params": r["best_params"], "n_train": r["n_train"],
                                "train_success": r["train"]["best"]["success"],
                                "train_success_v6": r["train"]["v6"]["success"],
                                "val_success": (r["val"]["best"] or {}).get("success"),
                                "val_success_v6": (r["val"]["v6"] or {}).get("success")}
        else:
            clusters[str(j)] = {"params": fallback, "source": "v6 fallback (not optimised)"}
        rows.append((j, clusters[str(j)]))
    (results_dir / "cluster_params.json").write_text(json.dumps(
        {"fallback": fallback, "clusters": clusters,
         "note": "tau_x_base fixed; parameters per situation cluster (clusters.npz)"}, indent=2), encoding="utf-8")

    print(f"\n{'cl':>3s} {'source':>11s} {'n_train':>7s} {'train best':>10s} {'train v6':>9s} {'val best':>9s} {'val v6':>7s}")
    for j, c in rows:
        if "train_success" in c:
            vb = "-" if c["val_success"] is None else f"{c['val_success']:.1%}"
            vv = "-" if c["val_success_v6"] is None else f"{c['val_success_v6']:.1%}"
            print(f"{j:3d} {('gated->v6' if c['source'].startswith('v6') else 'optimised'):>11s} {c['n_train']:7d} {c['train_success']:10.1%} "
                  f"{c['train_success_v6']:9.1%} {vb:>9s} {vv:>7s}")
        else:
            print(f"{j:3d} {'v6 fallback':>11s}")
    print(f"\nSaved {results_dir / 'cluster_params.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=int, nargs="+", default=None)
    ap.add_argument("--pop-size", type=int, default=int(OPT.get("pop_size", 10)))
    ap.add_argument("--n-generations", type=int, default=int(OPT.get("n_generations", 10)))
    ap.add_argument("--F", type=float, default=float(OPT.get("F", 0.5)))
    ap.add_argument("--CR", type=float, default=float(OPT.get("CR", 0.9)))
    ap.add_argument("--patience", type=int, default=OPT.get("patience"))
    ap.add_argument("--n-jobs", type=int, default=int(OPT.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--results-dir", default=str(BASE_DIR / OPT.get("results_dir", "cluster_opt_results")))
    args = ap.parse_args()
    if args.pop_size < 6:
        raise SystemExit("pop_size must be >= 6 for DE/best/2")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    asg = pd.read_csv(BASE_DIR / "cluster_assignment.csv")
    k = int(np.load(BASE_DIR / "clusters.npz")["centres"].shape[0])

    from capture_states import load_v6_params
    v6, v6_src = load_v6_params((BASE_DIR / CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve())
    v6 = {key: float(val) for key, val in v6.items()}

    if args.collect:
        collect(results_dir, k, v6)
        return

    todo = []
    for j in (args.clusters if args.clusters is not None else range(k)):
        tr = asg[(asg.cluster == j) & (asg.split == "train")].episode_id.tolist()
        va = asg[(asg.cluster == j) & (asg.split == "val")].episode_id.tolist()
        if len(tr) < int(OPT.get("min_train_episodes", 15)):
            print(f"[cluster {j}] only {len(tr)} training episodes -> skipped (v6 fallback)")
            continue
        if (results_dir / f"cluster_{j}.json").exists() and not args.rerun:
            print(f"[cluster {j}] already done -> skipped")
            continue
        todo.append((j, tr, va))

    futures = int(OPT.get("futures", 1))
    evals_max = sum(len(tr) * futures * args.pop_size * (1 + args.n_generations) for _, tr, _ in todo)
    rate = 0.06 * float(OPT["distance_m"]) / 150.0 * 11 / max(1, args.n_jobs)   # s per run (rough)
    print("=" * 100)
    print("Step 3: one DE per situation cluster")
    print(f"Clusters    : {len(todo)} to run (K = {k})")
    print(f"Objective   : (1 - success) + 1e-4*C over each cluster's training episodes, "
          f"{OPT['distance_m']:g} m, {futures} future(s) per episode")
    print(f"Tuned       : {', '.join(PARAMS)}   (tau_x_base fixed at {CONFIG['dipf']['tau_x_base']})")
    print(f"DE          : pop {args.pop_size}, {args.n_generations} generations, F {args.F}, CR {args.CR}, "
          f"patience {args.patience}")
    print(f"Baseline    : {v6_src}")
    print(f"Budget      : up to {evals_max:,} simulations, very roughly {evals_max * rate / 3600:.1f} h "
          f"with {args.n_jobs} parallel (less with early stopping)")
    print("=" * 100, flush=True)

    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_worker_init) as pool:
        for j, tr, va in todo:
            optimize_cluster(pool, j, tr, va, v6, args, results_dir)

    collect(results_dir, k, v6)


if __name__ == "__main__":
    main()
