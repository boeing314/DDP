"""
Compare parameter sets on full-road runs over (fresh) seeds -- paired, same traffic.

Sets (--sets, first one is the reference for the paired tests):
    config        the dipf.* values in config.yaml
    best          de_best.json (v9) or optuna_best.json (v10), whichever exists
    v6_old        v6's tuned parameters, incl. its own tau_x_base (reference.v6_params)

Usage:
    python evaluate.py                                   # config vs v6_old on seeds 1000..1499
    python evaluate.py --sets best config v6_old --seeds 1000 1999
"""

import argparse
import json
import math
import time
from pathlib import Path

from runner import BASE_DIR, CONFIG, config_params, evaluate, fmt, make_pool


def load_set(name):
    if name == "config":
        return config_params()
    if name == "best":
        for f in ("de_best.json", "optuna_best.json"):
            if (BASE_DIR / f).exists():
                return json.loads((BASE_DIR / f).read_text())["best_params"]
        raise SystemExit("no de_best.json / optuna_best.json yet")
    if name == "v6_old":
        v6 = json.loads((BASE_DIR / CONFIG["reference"]["v6_params"]).read_text())["best_params"]
        return {k: float(v) for k, v in v6.items()}
    raise SystemExit(f"unknown set {name}")


def mcnemar(a_ok, b_ok):
    only_a = sum(x and not y for x, y in zip(a_ok, b_ok))
    only_b = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["config", "v6_old"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[1000, 1499])
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    pool, n = make_pool(args.n_jobs)
    print(f"Full road, seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} runs per set, {n} parallel)", flush=True)
    res = {}
    for name in args.sets:
        t0 = time.time()
        s, rows = evaluate(pool, load_set(name), seeds)
        res[name] = (s, [r["status"] == "success" for r in rows])
        print(f"  {name:12s} {fmt(s)}  [{time.time() - t0:.0f}s]", flush=True)
    pool.shutdown()
    ref = args.sets[0]
    for name in args.sets[1:]:
        a, b, p = mcnemar(res[ref][1], res[name][1])
        print(f"paired {ref} vs {name}: only {ref} succeeds {a}, only {name} succeeds {b}, exact McNemar p = {p:.4f}")
    out = BASE_DIR / f"eval_{'_'.join(args.sets)}_{seeds[0]}_{seeds[-1]}.json"
    out.write_text(json.dumps({k: v[0] for k, v in res.items()}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
