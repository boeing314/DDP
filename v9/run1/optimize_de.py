"""
v9: differential evolution (v4 algorithm) for ONE fixed parameter set -- the baseline.

Searches the dipf.* keys in config.yaml -> optimization.bounds (the DIPF
potential coefficients; tau_x_base is fixed and not searched) for the lowest

    J = (1 - success_rate) + w_progress * (1 - mean_progress)

over full-road runs on the training seeds (optimization.train_seeds). The same
seeds are used for every candidate (common random numbers).

DE: log10 space, Latin hypercube initial population (member 0 = the values
currently in config.yaml), DE/best/2 mutation, exponential crossover, greedy
selection, best updated once per generation. A checkpoint is written after the
initial population and after every generation; --resume continues from it
(random state included). At the end the best set is scored once on the
VALIDATION seeds, next to the starting values.

Outputs:
    de_log.jsonl        every evaluation
    de_checkpoint.npz   population / costs (+ de_checkpoint.rng.json)
    de_best.json        best parameters, train + validation results

Usage:
    python optimize_de.py                     # settings from config.yaml -> optimization
    python optimize_de.py --resume
    python optimize_de.py --n-generations 10 --pop-size 12 --n-jobs 7
    python optimize_de.py --resume --n-generations 12 --apply   # stop early, validate, write config
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from runner import BASE_DIR, OPT, apply_to_config, config_params, evaluate, fmt, make_pool, seed_range

BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in OPT["bounds"].items()}
NAMES = list(BOUNDS)
D = len(NAMES)
LO = np.array([math.log10(BOUNDS[k][0]) for k in NAMES])
HI = np.array([math.log10(BOUNDS[k][1]) for k in NAMES])
if "tau_x_base" in BOUNDS:
    raise SystemExit("tau_x_base is fixed in v9 -- remove it from optimization.bounds")

LOG, CKPT, BEST = BASE_DIR / "de_log.jsonl", BASE_DIR / "de_checkpoint.npz", BASE_DIR / "de_best.json"


def to_params(v):
    return {k: float(10.0 ** v[i]) for i, k in enumerate(NAMES)}


def to_log(p):
    return np.clip([math.log10(p[k]) for k in NAMES], LO, HI)


def lhs(n, rng):
    u = np.empty((n, D))
    for d in range(D):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return LO + u * (HI - LO)


def mutate(pop, i, best, F, rng):
    r = rng.choice([k for k in range(len(pop)) if k not in (i, best)], size=4, replace=False)
    return np.clip(pop[best] + F * (pop[r[0]] - pop[r[1]] + pop[r[2]] - pop[r[3]]), LO, HI)


def crossover(theta, v, CR, rng):
    u = theta.copy()
    j, k = int(rng.integers(0, D)), 0
    while True:
        u[(j + k) % D] = v[(j + k) % D]
        k += 1
        if rng.random() >= CR or k == D:
            return u


def save_ckpt(gen, pop, pop_params, J, summaries, rng):
    np.savez(CKPT, generation=gen, pop=pop, J=J)
    (CKPT.with_suffix(".rng.json")).write_text(json.dumps(
        {"rng": rng.bit_generator.state, "pop_params": pop_params, "summaries": summaries},
        default=lambda o: o.tolist() if hasattr(o, "tolist") else o))


def load_ckpt(rng):
    z = np.load(CKPT)
    extra = json.loads(CKPT.with_suffix(".rng.json").read_text())
    st = extra["rng"]
    st["state"]["state"], st["state"]["inc"] = int(st["state"]["state"]), int(st["state"]["inc"])
    rng.bit_generator.state = st
    return int(z["generation"]), z["pop"], extra["pop_params"], z["J"], extra["summaries"]


def log(**kw):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop-size", type=int, default=int(OPT.get("pop_size", 15)))
    ap.add_argument("--n-generations", type=int, default=int(OPT.get("n_generations", 20)))
    ap.add_argument("--F", type=float, default=float(OPT.get("F", 0.5)))
    ap.add_argument("--CR", type=float, default=float(OPT.get("CR", 0.9)))
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write the best set into config.yaml at the end")
    ap.add_argument("--train-seeds", type=int, nargs=2, default=None, help="override optimization.train_seeds")
    ap.add_argument("--val-seeds", type=int, nargs=2, default=None, help="override optimization.validation_seeds")
    args = ap.parse_args()
    if args.pop_size < 6:
        raise SystemExit("pop_size must be >= 6 for DE/best/2")

    train = seed_range(args.train_seeds or OPT["train_seeds"])
    val = seed_range(args.val_seeds or OPT["validation_seeds"])
    rng = np.random.default_rng(int(OPT.get("rng_seed", 0)))
    start = config_params()
    pool, n_jobs = make_pool(args.n_jobs)

    print("=" * 100)
    print("v9 differential evolution -- one fixed parameter set (baseline)")
    print(f"Objective  : J = (1 - success) + {OPT.get('w_progress', 0.2)} * (1 - progress), full road, "
          f"limit {OPT.get('max_time_s', 300)} s")
    print(f"Seeds      : train {train[0]}..{train[-1]} ({len(train)}), validation {val[0]}..{val[-1]} ({len(val)})")
    print(f"DE         : pop {args.pop_size}, {args.n_generations} generations, F {args.F}, CR {args.CR} "
          f"-> up to {len(train) * args.pop_size * (1 + args.n_generations):,} runs, {n_jobs} parallel")
    print("Search (log-uniform)          start value")
    for k in NAMES:
        print(f"  {k:<16s} [{BOUNDS[k][0]:g}, {BOUNDS[k][1]:g}]   {start[k]:.6g}")
    print("=" * 100, flush=True)

    t0 = time.time()
    try:
        if args.resume and CKPT.exists():
            gen0, pop, pop_params, J, summ = load_ckpt(rng)
            print(f"Resumed after generation {gen0} (best J {J.min():.4f})", flush=True)
        else:
            gen0 = 0
            pop = lhs(args.pop_size, rng)
            pop[0] = to_log(start)
            pop_params = [start] + [to_params(v) for v in pop[1:]]
            J, summ = np.zeros(args.pop_size), [None] * args.pop_size
            for i in range(args.pop_size):
                s, _ = evaluate(pool, pop_params[i], train)
                J[i], summ[i] = s["J"], s
                log(generation=0, member=i, params=pop_params[i], **s)
                print(f"  init[{i:2d}] {fmt(s)}{'   (config.yaml start)' if i == 0 else ''}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            save_ckpt(0, pop, pop_params, J, summ, rng)
        best = int(np.argmin(J))

        for gen in range(gen0 + 1, args.n_generations + 1):
            print(f"\n=== generation {gen}/{args.n_generations}  best so far: {fmt(summ[best])}", flush=True)
            acc = 0
            for i in range(args.pop_size):
                u = crossover(pop[i], mutate(pop, i, best, args.F, rng), args.CR, rng)
                trial = to_params(u)
                s, _ = evaluate(pool, trial, train)
                parent = float(J[i])
                ok = s["J"] < parent
                log(generation=gen, member=i, params=trial, accepted=bool(ok), parent_J=parent, **s)
                if ok:
                    pop[i], pop_params[i], J[i], summ[i] = u, trial, s["J"], s
                    acc += 1
                print(f"  cand[{i:2d}] {fmt(s)}  {'ACCEPTED' if ok else 'rejected'} (parent J {parent:.4f})"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
            best = int(np.argmin(J))
            save_ckpt(gen, pop, pop_params, J, summ, rng)
            print(f"  -> generation {gen}: accepted {acc}/{args.pop_size}, best J {J[best]:.4f}", flush=True)

        best_params = pop_params[best]
        print(f"\nValidation on {len(val)} unseen seeds ...", flush=True)
        v_best, _ = evaluate(pool, best_params, val)
        v_start, _ = evaluate(pool, start, val)
    except KeyboardInterrupt:
        print("\nInterrupted -- the last completed generation is checkpointed; rerun with --resume.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    out = {"best_params": best_params, "train": summ[best], "validation": v_best,
           "validation_start_values": v_start, "start_params": start,
           "train_seeds": OPT["train_seeds"], "validation_seeds": OPT["validation_seeds"]}
    BEST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(f"Best (train)        : {fmt(summ[best])}")
    print(f"Best (validation)   : {fmt(v_best)}")
    print(f"Start (validation)  : {fmt(v_start)}")
    for k in NAMES:
        print(f"  {k:<16s} = {best_params[k]!r}")
    print(f"Saved {BEST}")
    if args.apply:
        apply_to_config(best_params)
        print("Best parameters written to config.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
