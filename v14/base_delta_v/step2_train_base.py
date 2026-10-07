"""
Step 2 (v14/base) -- train the best FIXED parameter set: the strong baseline for v14/adaptive.

All tuned DIPF keys (config.yaml -> pipeline.base_search.bounds) are searched with
CMA-ES, with the SAME scenarios, futures, objective, budget (popsize x generations)
and validation-based selection as the adaptive policy:

    J(theta) = success rate + w_progress * mean progress
               over the train scenarios x futures 1..n (takeover -> x = 400 m), fixed parameters theta

Search space: u in [0,1]^7, theta_k = 10^(lo_k + u_k (hi_k - lo_k)) (log-uniform ranges).
CMA-ES starts at the start set (v12 library set 2) with sigma0 in units of the ranges.
Samples outside [0,1] are clipped to the box (and lightly penalised so the search stays inside).

Every val_every generations the search mean and the best train candidate are scored on
the validation scenarios; base_best.json = the validated set with the highest J
(possibly the start set itself). v14/adaptive then uses it as its p0.

Outputs:
    base_best.json        the selected fixed set (+ start / selection results)
    cmaes_state.json      CMA-ES state, history, validation checkpoints (resume)
    cmaes_log.jsonl       every candidate

Usage:
    python step2_train_base.py --n-jobs 7       # start or resume
    python step2_train_base.py --finish         # no more generations: validate the mean, write base_best.json
    python step2_train_base.py --restart
"""

import argparse
import json
import math
import shutil
import time

import numpy as np

from common import BASE, BASE_DIR, CMA, CMAES, END_X, FUT, ROAD_LENGTH, SEARCH, evaluate, make_pool, split_ids, summarize

NAMES = list(SEARCH["bounds"])
LO = np.array([math.log10(float(SEARCH["bounds"][k][0])) for k in NAMES])
HI = np.array([math.log10(float(SEARCH["bounds"][k][1])) for k in NAMES])
STATE, LOG, OUT = BASE_DIR / "cmaes_state.json", BASE_DIR / "cmaes_log.jsonl", BASE_DIR / "base_best.json"


def to_params(u):
    v = LO + np.clip(u, 0.0, 1.0) * (HI - LO)
    return {k: float(10.0 ** v[i]) for i, k in enumerate(NAMES)}


def to_u(params):
    return np.clip((np.array([math.log10(params[k]) for k in NAMES]) - LO) / (HI - LO), 0.0, 1.0)


def outside(u):
    return float(np.sum(np.maximum(0.0, -u) + np.maximum(0.0, u - 1.0)))


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    tmp.replace(path)


def fmt(s):
    return f"J {s['J']:.4f}  success {s['success']:.3f}  progress {s['progress']:.3f}"


def describe(u):
    p = to_params(u)
    return ", ".join(f"{k} {p[k]:.3g} (x{p[k] / BASE[k]:.2f})" for k in NAMES)


def validate(st, pool, val, u, kind, gen):
    s = summarize(evaluate(pool, [("fixed", to_params(u))], val, FUT["val"])[0])
    st["checkpoints"].append({"gen": gen, "kind": kind, "u": list(map(float, np.clip(u, 0, 1))), "val": s})
    print(f"  validation ({kind}, gen {gen}): {fmt(s)}   [start set: {fmt(st['start_val'])}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--generations", type=int, default=int(CMA["generations"]))
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    if args.restart:
        old = BASE_DIR / f"cmaes_old_{time.strftime('%Y%m%d_%H%M%S')}"
        old.mkdir()
        for f in (STATE, LOG, OUT):
            if f.exists():
                shutil.move(str(f), old / f.name)
        print(f"Old results moved to {old}")

    train, val = split_ids("train"), split_ids("val")
    fp = {"start": BASE, "bounds": SEARCH["bounds"], "train": [train[0], train[-1], len(train)],
          "val": [val[0], val[-1], len(val)], "futures": FUT, "end_x": END_X, "road_length_m": ROAD_LENGTH,
          "w_progress": CMA.get("w_progress", 0.1)}
    u0 = to_u(BASE)
    pool, n_jobs = make_pool(args.n_jobs)
    es = CMAES(u0, float(CMA["sigma0"]), CMA.get("popsize"), int(CMA.get("rng_seed", 0)))

    print("=" * 100)
    print("Step 2 (v14/base): CMA-ES for the best FIXED parameter set")
    print(f"Search  : {len(NAMES)} parameters {NAMES}, log-uniform ranges, popsize {es.lam}, sigma0 {CMA['sigma0']} "
          f"(of each range), <= {args.generations} generations")
    print(f"Scenario: train {len(train)} x {FUT['train']} futures = {len(train) * FUT['train']} runs per candidate, "
          f"validation {len(val)} x {FUT['val']} every {CMA['val_every']} generations;  {n_jobs} parallel")
    print("=" * 100, flush=True)

    t0 = time.time()
    try:
        if STATE.exists():
            st = json.loads(STATE.read_text())
            if st["fingerprint"] != fp:
                raise SystemExit(f"{STATE.name} was made with other settings -- use --restart")
            es.load(st["cmaes"])
            print(f"Resumed after generation {es.gen}: best train {fmt(st['best_train']['train'])}", flush=True)
        else:
            print("Start set (v12 set 2) on train and validation ...", flush=True)
            st = {"fingerprint": fp, "start_train": summarize(evaluate(pool, [None], train, FUT["train"])[0]),
                  "history": [], "checkpoints": []}
            st["start_val"] = summarize(evaluate(pool, [None], val, FUT["val"])[0])
            st["best_train"] = {"u": list(map(float, u0)), "train": st["start_train"], "gen": 0}
            st["checkpoints"].append({"gen": 0, "kind": "start", "u": list(map(float, u0)), "val": st["start_val"]})
            print(f"  start train: {fmt(st['start_train'])}\n  start val  : {fmt(st['start_val'])}", flush=True)
            st["cmaes"] = es.state()
            save(st, STATE)

        while not args.finish and es.gen < args.generations:
            g = es.gen + 1
            X = es.ask()
            S = [summarize(r) for r in evaluate(pool, [("fixed", to_params(u)) for u in X], train, FUT["train"])]
            J = np.array([s["J"] for s in S])
            es.tell(X, -J + 0.1 * np.array([outside(u) for u in X]))
            with LOG.open("a", encoding="utf-8") as f:
                for i, (u, s) in enumerate(zip(X, S)):
                    f.write(json.dumps({"gen": g, "member": i, "params": to_params(u), **s}) + "\n")
            k = int(J.argmax())
            if J[k] > st["best_train"]["train"]["J"]:
                st["best_train"] = {"u": list(map(float, np.clip(X[k], 0, 1))), "train": S[k], "gen": g}
            st["history"].append({"gen": g, "J_best": float(J.max()), "J_median": float(np.median(J)),
                                  "J_worst": float(J.min()), "success_best": S[k]["success"], "sigma": es.sigma,
                                  "time_s": time.time() - t0})
            print(f"gen {g:3d}/{args.generations}: J best {J.max():.4f} / median {np.median(J):.4f} / worst {J.min():.4f}"
                  f"  (success of best {S[k]['success']:.3f}; start {st['start_train']['success']:.3f})  "
                  f"sigma {es.sigma:.3f}  [{time.time() - t0:.0f}s]", flush=True)
            if g % int(CMA["val_every"]) == 0:
                validate(st, pool, val, es.m, "mean", g)
                bt = st["best_train"]
                if bt["gen"] > 0 and not any(c["u"] == bt["u"] for c in st["checkpoints"]):
                    validate(st, pool, val, np.array(bt["u"]), "best train", bt["gen"])
                print(f"  search mean: {describe(es.m)}", flush=True)
            st["cmaes"] = es.state()
            save(st, STATE)

        if es.gen > 0 and not any(c["gen"] == es.gen and c["kind"] == "mean" for c in st["checkpoints"]):
            validate(st, pool, val, es.m, "mean", es.gen)
            save(st, STATE)
    except KeyboardInterrupt:
        print("\nInterrupted -- the last finished generation is saved; rerun to continue (or --finish).")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    sel = max(st["checkpoints"], key=lambda c: (round(c["val"]["J"], 9), -c["gen"]))
    params = to_params(np.array(sel["u"]))
    OUT.write_text(json.dumps({"params": params, "selected": {"gen": sel["gen"], "kind": sel["kind"], "val": sel["val"]},
                               "start": {"params": BASE, "train": st["start_train"], "val": st["start_val"]},
                               "generations_run": es.gen}, indent=2))
    print("\n" + "=" * 100)
    print(f"Selected : {sel['kind']} of generation {sel['gen']}  validation {fmt(sel['val'])}")
    print(f"Start set: validation {fmt(st['start_val'])}")
    for k in NAMES:
        print(f"  {k:<17s} {params[k]:.6g}   (start {BASE[k]:.6g})")
    print(f"Saved {OUT}  -> used as p0 by v14/adaptive")
    print("=" * 100)


if __name__ == "__main__":
    main()
