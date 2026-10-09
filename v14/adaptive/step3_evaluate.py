"""
Step 3 (v14/adaptive) -- closed-loop test on FRESH scenarios (pipeline.seeds.test; never used in training).

Compared on the same scenarios x futures (runs are exactly repeatable, so every policy
meets identical traffic randomness):
    base            the TRAINED fixed set from v14/base (= the adaptive policy with x = 0);
                    reproduces v14/base step 3 exactly
    base start      the fixed set v14/base's search started from (base_best.json -> start:
                    the previous v14/base best from 'v14 copy', old tau)
    adaptive        the trained policy (policy.json)
    bias part       the trained policy without its feature weights (W = 0, same b):
                    a fixed re-tuning; adaptive vs bias part = what the ADAPTATION adds
    bias-only       policy_biasonly.json if you trained the control experiment
                    (step 2 --bias-only): the best fixed re-tuning found with the same budget

Paired comparison per run (scenario, future): McNemar exact test on the runs where
exactly one of the two policies succeeded.

Outputs:
    test_results.csv   per scenario: success probability of every policy
    test_summary.json

Usage:
    python step3_evaluate.py --n-jobs 7
    python step3_evaluate.py --policy policy.json --futures 5
"""

import argparse
import json
import math
import time

import numpy as np

from common import ADA, BASE_DIR, FUT, bias_part, evaluate, make_pool, per_scenario, split_ids, summarize


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def paired(ra, rb):
    """Runs where only A / only B succeeded, McNemar p, per-scenario mean difference +- SE."""
    b = c = 0
    for sid in ra:
        for (oa, _), (ob, _) in zip(ra[sid], rb[sid]):
            if oa == oa and ob == ob:
                b += int(oa == 1.0 and ob == 0.0)
                c += int(oa == 0.0 and ob == 1.0)
    pa, pb = per_scenario(ra), per_scenario(rb)
    d = np.array([pa[s] - pb[s] for s in ra])
    return {"only_a": b, "only_b": c, "mcnemar_p": mcnemar(b, c), "mean_diff": float(d.mean()),
            "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="policy.json")
    ap.add_argument("--futures", type=int, default=FUT["test"])
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    pj = json.loads((BASE_DIR / args.policy).read_text())
    pols = {"base": None, "adaptive": pj["policy"], "bias part": bias_part(pj["policy"])}
    start = json.loads((BASE_DIR / ADA["base_from"]).read_text())["start"]["params"]
    pols["base start"] = ("fixed", start)
    bo = BASE_DIR / "policy_biasonly.json"
    if bo.exists() and not pj.get("bias_only"):
        pols["bias-only (trained)"] = json.loads(bo.read_text())["policy"]
    test = split_ids("test")

    pool, n_jobs = make_pool(args.n_jobs)
    print("=" * 90)
    print(f"Step 3: test on {len(test)} fresh scenarios x {args.futures} futures, policies: {list(pols)}")
    print(f"        {len(pols) * len(test) * args.futures:,} runs, {n_jobs} parallel")
    print("=" * 90, flush=True)
    t0 = time.time()
    try:
        runs = dict(zip(pols, evaluate(pool, list(pols.values()), test, args.futures)))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    print(f"done [{time.time() - t0:.0f}s]\n")

    summ = {k: summarize(r) for k, r in runs.items()}
    print(f"{'policy':<22s} {'success':>8s} {'progress':>9s}")
    for k, s in summ.items():
        print(f"{k:<22s} {s['success']:8.3f} {s['progress']:9.3f}")
    comps = {}
    for a, b in [("adaptive", "base"), ("adaptive", "bias part"), ("adaptive", "bias-only (trained)"),
                 ("bias part", "base"), ("base", "base start"), ("adaptive", "base start")]:
        if a in runs and b in runs:
            c = comps[f"{a} vs {b}"] = paired(runs[a], runs[b])
            print(f"{a} vs {b}: {c['mean_diff']:+.3f} +- {c['se']:.3f} (SE) per scenario; runs only {a} succeeded "
                  f"{c['only_a']}, only {b} {c['only_b']} -> McNemar p = {c['mcnemar_p']:.3g}")
    print("\nReading: 'adaptive vs base' = what adapting adds on top of the best fixed set found with the same"
          "\nsearch budget (the main result); 'base vs base start' = what training the fixed set gained;"
          "\n'adaptive vs bias part' = the adapting part of the policy itself.")

    ps = {k: per_scenario(r) for k, r in runs.items()}
    with (BASE_DIR / "test_results.csv").open("w", encoding="utf-8") as f:
        f.write("sid," + ",".join(k.replace(" ", "_") for k in runs) + "\n")
        for sid in test:
            f.write(sid + "," + ",".join(f"{ps[k][sid]:.4f}" for k in runs) + "\n")
    (BASE_DIR / "test_summary.json").write_text(json.dumps({"summary": summ, "paired": comps, "futures": args.futures,
                                                             "n_scenarios": len(test), "policy": args.policy}, indent=2))
    print(f"\nSaved {BASE_DIR / 'test_results.csv'} and {BASE_DIR / 'test_summary.json'}")


if __name__ == "__main__":
    main()
