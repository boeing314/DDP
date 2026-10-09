"""
Step 3 (v14/base) -- test the trained fixed set on FRESH scenarios (pipeline.seeds.test).

    start set     pipeline.base_search.start_params (the search start: v14/base's previous best, old tau)
    trained base  base_best.json (the strong fixed baseline v14/adaptive builds on)

    v13 adaptive  (optional, --v13-policy) the v13 policy; it was trained on top of v12 set 2 (no longer
                  the start set here); it runs as in v13 -- on the same runs as the others

Runs are exactly repeatable, so v14/adaptive's step 3 reproduces the "trained base" numbers.

Usage:
    python step3_evaluate.py --n-jobs 7
    python step3_evaluate.py --n-jobs 7 --v13-policy ../../v13/run1/policy.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from common import BASE_DIR, FUT, evaluate, make_pool, per_scenario, split_ids, summarize


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--futures", type=int, default=FUT["test"])
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--v13-policy", default=None, help="also test this v13 policy.json (built on v12 set 2)")
    args = ap.parse_args()
    best = json.loads((BASE_DIR / "base_best.json").read_text())
    test = split_ids("test")
    pool, n_jobs = make_pool(args.n_jobs)
    print(f"Step 3 (v14/base): {len(test)} test scenarios x {args.futures} futures, start set vs trained base, "
          f"{n_jobs} parallel", flush=True)
    t0 = time.time()
    cands = [None, ("fixed", best["params"])]
    if args.v13_policy:
        cands.append(json.loads(Path(args.v13_policy).read_text())["policy"])
    try:
        runs = evaluate(pool, cands, test, args.futures)
        start, trained = runs[0], runs[1]
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    ss, st = summarize(start), summarize(trained)
    b = sum(int(a[0] == 1.0 and s[0] == 0.0) for sid in test for a, s in zip(trained[sid], start[sid]))
    c = sum(int(a[0] == 0.0 and s[0] == 1.0) for sid in test for a, s in zip(trained[sid], start[sid]))
    pa, pb = per_scenario(trained), per_scenario(start)
    d = np.array([pa[s] - pb[s] for s in test])
    print(f"done [{time.time() - t0:.0f}s]\n")
    print(f"start set            : success {ss['success']:.3f}  progress {ss['progress']:.3f}")
    print(f"trained base         : success {st['success']:.3f}  progress {st['progress']:.3f}")
    print(f"trained - start: {d.mean():+.3f} +- {d.std(ddof=1) / math.sqrt(len(d)):.3f} (SE); runs only trained succeeded "
          f"{b}, only start {c} -> McNemar p = {mcnemar(b, c):.3g}")
    out = {"start": ss, "trained_base": st, "only_trained": b, "only_start": c, "mcnemar_p": mcnemar(b, c),
           "futures": args.futures, "n_scenarios": len(test)}
    if args.v13_policy:
        v13 = runs[2]
        sv = summarize(v13)
        b2 = sum(int(a[0] == 1.0 and s[0] == 0.0) for sid in test for a, s in zip(v13[sid], trained[sid]))
        c2 = sum(int(a[0] == 0.0 and s[0] == 1.0) for sid in test for a, s in zip(v13[sid], trained[sid]))
        pv = per_scenario(v13)
        d2 = np.array([pv[s] - pa[s] for s in test])
        print(f"v13 adaptive         : success {sv['success']:.3f}  progress {sv['progress']:.3f}")
        print(f"v13 adaptive - trained base: {d2.mean():+.3f} +- {d2.std(ddof=1) / math.sqrt(len(d2)):.3f} (SE); "
              f"runs only v13 succeeded {b2}, only trained base {c2} -> McNemar p = {mcnemar(b2, c2):.3g}")
        out.update({"v13_adaptive": sv, "v13_only": b2, "trained_only_vs_v13": c2, "v13_mcnemar_p": mcnemar(b2, c2)})
    (BASE_DIR / "test_summary.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {BASE_DIR / 'test_summary.json'}")


if __name__ == "__main__":
    main()
