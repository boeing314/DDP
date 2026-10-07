"""
Step 2 -- train the adaptive-parameter policy with CMA-ES on closed-loop segment success.

The policy (policy.py) turns 9 traffic features into multipliers for the adapted
DIPF parameters every control step. Its weights x = (W, b) are searched with CMA-ES:

    J(x) = success rate + w_progress * mean progress
           over the train scenarios x futures 1..n, driving the segment (takeover -> x=400 m)
           with the policy x on top of the base parameters (config.yaml -> pipeline.adaptive)

CMA-ES starts at x = 0 (= the fixed base parameters) with step size sigma0. Each
generation it samples popsize weight vectors around its mean, runs all of them
(one parallel batch, same scenarios and futures for every candidate), and moves
its mean / covariance towards the better half.

Every val_every generations the search MEAN (and the best train candidate) is
scored on the validation scenarios. The final policy is the validated one with
the highest validation J -- possibly x = 0 (the base) if nothing generalised.

--bias-only trains only b (W = 0): a plain re-tuning of the base parameters with
the same machinery. Comparing it with the full policy (step 3) shows how much of a
gain comes from ADAPTING the parameters rather than from better fixed values.

Outputs (suffix _biasonly with --bias-only):
    policy.json        the selected policy (+ base / selection results)
    cmaes_state.json   CMA-ES state, history, validation checkpoints (resume)
    cmaes_log.jsonl    every candidate

Usage:
    python step2_train_policy.py --n-jobs 7          # start or resume
    python step2_train_policy.py --finish            # no more generations: validate the mean, write policy.json
    python step2_train_policy.py --bias-only         # control experiment
    python step2_train_policy.py --restart           # move old results away and start over
"""

import argparse
import json
import shutil
import time

import numpy as np

import policy as pol
from common import (ADA, BASE, BASE_DIR, CMA, CMAES, FUT, PARAMS, PIPE, ROAD_LENGTH, END_X, evaluate,
                    make_policy, make_pool, split_ids, summarize)


def files(bias_only):
    sfx = "_biasonly" if bias_only else ""
    return (BASE_DIR / f"cmaes_state{sfx}.json", BASE_DIR / f"cmaes_log{sfx}.jsonl", BASE_DIR / f"policy{sfx}.json")


def fingerprint(bias_only, train, val):
    return {"params": PARAMS, "features": pol.FEATURES, "base": BASE, "limits": ADA["limits"], "clip": ADA["clip"],
            "smooth_s": ADA["smooth_s"], "signs": ADA.get("signs"), "train": [train[0], train[-1], len(train)],
            "val": [val[0], val[-1], len(val)], "futures": FUT, "bias_only": bias_only, "end_x": END_X,
            "road_length_m": ROAD_LENGTH, "w_progress": CMA.get("w_progress", 0.1)}


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    tmp.replace(path)


def fmt(s):
    return f"J {s['J']:.4f}  success {s['success']:.3f}  progress {s['progress']:.3f}"


def describe(x, bias_only):
    """Largest effects of a weight vector, in words."""
    W, b = pol.unpack(x, PARAMS, ADA.get("signs"), bias_only)
    out = [f"{p}: base x{np.exp(np.clip(b[i], -ADA['clip'], ADA['clip'])):.2f}" for i, p in enumerate(PARAMS)]
    flat = sorted(((abs(W[i, j]), i, j) for i in range(len(PARAMS)) for j in range(len(pol.FEATURES))), reverse=True)
    for _, i, j in flat[:4]:
        if abs(W[i, j]) > 0.05:
            out.append(f"{PARAMS[i]} x{np.exp(W[i, j]):.2f} per unit {pol.FEATURES[j]}")
    return "; ".join(out)


def validate(st, pool, val, x, kind, gen, bias_only):
    s = summarize(evaluate(pool, [make_policy(x, bias_only)], val, FUT["val"])[0])
    st["checkpoints"].append({"gen": gen, "kind": kind, "x": list(map(float, x)), "val": s})
    print(f"  validation ({kind}, gen {gen}): {fmt(s)}   [base: {fmt(st['base_val'])}]", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--generations", type=int, default=int(CMA["generations"]))
    ap.add_argument("--bias-only", action="store_true", help="train only b (W = 0): control experiment")
    ap.add_argument("--finish", action="store_true", help="no more generations; validate the mean and write policy.json")
    ap.add_argument("--restart", action="store_true", help="move old results away and start over")
    args = ap.parse_args()
    STATE, LOG, OUT = files(args.bias_only)

    if args.restart:
        old = BASE_DIR / f"cmaes_old_{time.strftime('%Y%m%d_%H%M%S')}"
        old.mkdir()
        for f in (STATE, LOG, OUT):
            if f.exists():
                shutil.move(str(f), old / f.name)
        print(f"Old results moved to {old}")

    train, val = split_ids("train"), split_ids("val")
    fp = fingerprint(args.bias_only, train, val)
    N = pol.n_dims(len(PARAMS), args.bias_only)
    pool, n_jobs = make_pool(args.n_jobs)
    es = CMAES(np.zeros(N), float(CMA["sigma0"]), CMA.get("popsize"), int(CMA.get("rng_seed", 0)))

    print("=" * 100)
    print(f"Step 2: CMA-ES for the adaptive-parameter policy{' (BIAS ONLY: W = 0)' if args.bias_only else ''}")
    print(f"Adapted : {PARAMS}  (others fixed at the base values), features: {len(pol.FEATURES)}")
    print(f"Search  : {N} dims, popsize {es.lam}, sigma0 {CMA['sigma0']}, <= {args.generations} generations; "
          f"clip |z| <= {ADA['clip']:.2f} (x{np.exp(ADA['clip']):.0f}), smoothing T = {ADA['smooth_s']} s")
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
            print("Base parameters (x = 0) on train and validation ...", flush=True)
            tb = time.time()
            base_runs = evaluate(pool, [None], train, FUT["train"])[0]
            per_run = (time.time() - tb) * n_jobs / (len(train) * FUT["train"])
            st = {"fingerprint": fp, "base_train": summarize(base_runs), "history": [], "checkpoints": [],
                  "best_train": {"x": [0.0] * N, "train": summarize(base_runs), "gen": 0}}
            st["base_val"] = summarize(evaluate(pool, [None], val, FUT["val"])[0])
            st["checkpoints"].append({"gen": 0, "kind": "base", "x": [0.0] * N, "val": st["base_val"]})
            gen_min = es.lam * len(train) * FUT["train"] * per_run * 1.2 / n_jobs / 60   # +20% for the features
            print(f"  base train: {fmt(st['base_train'])}\n  base val  : {fmt(st['base_val'])}")
            print(f"  ~{per_run:.1f} s per run per worker -> ~{gen_min:.0f} min per generation, "
                  f"~{gen_min * args.generations / 60:.1f} h for {args.generations} generations", flush=True)
            st["cmaes"] = es.state()
            save(st, STATE)

        while not args.finish and es.gen < args.generations:
            g = es.gen + 1
            X = es.ask()
            runs = evaluate(pool, [make_policy(x, args.bias_only) for x in X], train, FUT["train"])
            S = [summarize(r) for r in runs]
            J = np.array([s["J"] for s in S])
            es.tell(X, -J)
            with LOG.open("a", encoding="utf-8") as f:
                for i, (x, s) in enumerate(zip(X, S)):
                    f.write(json.dumps({"gen": g, "member": i, "x": list(map(float, x)), **s}) + "\n")
            k = int(J.argmax())
            if J[k] > st["best_train"]["train"]["J"]:
                st["best_train"] = {"x": list(map(float, X[k])), "train": S[k], "gen": g}
            st["history"].append({"gen": g, "J_best": float(J.max()), "J_median": float(np.median(J)),
                                  "J_worst": float(J.min()), "success_best": S[k]["success"], "sigma": es.sigma,
                                  "time_s": time.time() - t0})
            print(f"gen {g:3d}/{args.generations}: J best {J.max():.4f} / median {np.median(J):.4f} / worst {J.min():.4f}"
                  f"  (success of best {S[k]['success']:.3f}; base {st['base_train']['success']:.3f})  "
                  f"sigma {es.sigma:.3f}  [{time.time() - t0:.0f}s]", flush=True)
            if g % int(CMA["val_every"]) == 0:
                validate(st, pool, val, es.m, "mean", g, args.bias_only)
                bt = st["best_train"]
                if bt["gen"] > 0 and not any(c["x"] == bt["x"] for c in st["checkpoints"]):
                    validate(st, pool, val, np.array(bt["x"]), "best train", bt["gen"], args.bias_only)
                print(f"  search mean: {describe(es.m, args.bias_only)}", flush=True)
            st["cmaes"] = es.state()
            save(st, STATE)

        if not any(c["gen"] == es.gen and c["kind"] == "mean" for c in st["checkpoints"]) and es.gen > 0:
            validate(st, pool, val, es.m, "mean", es.gen, args.bias_only)
            save(st, STATE)
    except KeyboardInterrupt:
        print("\nInterrupted -- the last finished generation is saved; rerun to continue (or --finish).")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    sel = max(st["checkpoints"], key=lambda c: (round(c["val"]["J"], 9), -c["gen"]))
    x = np.array(sel["x"])
    out = {"policy": make_policy(x, args.bias_only), "x": sel["x"], "bias_only": args.bias_only,
           "selected": {"gen": sel["gen"], "kind": sel["kind"], "val": sel["val"]},
           "base": {"train": st["base_train"], "val": st["base_val"]}, "base_params": BASE,
           "generations_run": es.gen, "summary": describe(x, args.bias_only)}
    OUT.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 100)
    print(f"Selected: {sel['kind']} of generation {sel['gen']}  validation {fmt(sel['val'])}")
    print(f"Base    : validation {fmt(st['base_val'])}")
    if sel["kind"] == "base":
        print("No validated policy beat the base parameters -- policy.json = the base (W = 0, b = 0).")
    print(f"Policy  : {out['summary']}")
    print(f"Saved {OUT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
