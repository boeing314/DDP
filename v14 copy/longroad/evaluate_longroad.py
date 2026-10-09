"""
v14/longroad* -- base vs adaptive on a long drive (evaluation only, nothing is retrained).

Road, finish x and time limit come from config.yaml (longroad: 1250 m road, finish x = 1100 m,
~1000 m, 200 s; longroad4000: 4250 m road, finish x = 4100 m, ~4000 m, 600 s). The car nearest
x = 100 m at t = 30 s becomes the ego. Policies (config.yaml -> pipeline.policies selects which),
all on the same scenarios x futures (runs are exactly repeatable, so they meet identical traffic):

    base        the trained fixed set   (../base/base_best.json)
    adaptive    the trained policy       (../adaptive/policy.json, built on that base)
    v12 set 2   the fixed set ../base started from

Reported:
    success over the whole ~1000 m, progress, paired McNemar tests;
    distance profile: fraction of runs that covered >= d metres without a collision,
    every profile_step_m metres, then "finished" (= success) -- shows whether the gap
    between the policies grows with distance.

Usage:
    python step1_make_scenarios.py --n-jobs 7          # once: the 1250 m test scenarios (~1 min)
    python evaluate_longroad.py --n-jobs 7             # ~1 h
    python evaluate_longroad.py --n-jobs 7 --n-scenarios 100 --futures 2    # quicker (~15 min)
"""

import argparse
import json
import math
import time

import numpy as np

from common import BASE, BASE_DIR, END_X, FUT, MAX_TIME, PIPE, evaluate, load_ego, make_pool, split_ids, summarize

STEP = int(PIPE.get("profile_step_m", 100))
DISTANCES = list(range(STEP, int(END_X - 100) - STEP + 1, STEP))   # + a final 'finished' row (segments vary a little)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def paired(ra, rb):
    b = sum(int(x[0] == 1.0 and y[0] == 0.0) for sid in ra for x, y in zip(ra[sid], rb[sid]))
    c = sum(int(x[0] == 0.0 and y[0] == 1.0) for sid in ra for x, y in zip(ra[sid], rb[sid]))
    return b, c, mcnemar(b, c)


def covered(runs, seg):
    """Metres covered without a collision, per run (a success covers the whole segment)."""
    return [p * seg[sid] for sid, v in runs.items() for _, p in v if p == p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--futures", type=int, default=FUT["test"])
    ap.add_argument("--n-scenarios", type=int, default=None, help="only the first n test scenarios")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    best = json.loads((BASE_DIR / "../base/base_best.json").read_text())
    adaptive = json.loads((BASE_DIR / "../adaptive/policy.json").read_text())["policy"]
    pols = {"base": None, "adaptive": adaptive, "v12 set 2": ("fixed", best["start"]["params"])}
    pols = {k: v for k, v in pols.items() if k in PIPE.get("policies", list(pols))}
    test = split_ids("test")[:args.n_scenarios]
    seg = {sid: END_X - float(load_ego(sid)["centre_x"]) for sid in test}

    pool, n_jobs = make_pool(args.n_jobs)
    print("=" * 90)
    print(f"v14/longroad: {len(test)} scenarios x {args.futures} futures, ~{np.mean(list(seg.values())):.0f} m per run "
          f"(x -> {END_X:g} m, limit {MAX_TIME:g} s), policies {list(pols)}")
    print(f"{len(pols) * len(test) * args.futures:,} runs, {n_jobs} parallel")
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
    print(f"{'policy':<12s} {'success (~%d m)' % round(END_X - 100, -2):>18s} {'progress':>9s}")
    for k, s in summ.items():
        print(f"{k:<12s} {s['success']:18.3f} {s['progress']:9.3f}")
    comps = {}
    for a, b in [(a, b) for a, b in [("adaptive", "base"), ("base", "v12 set 2"), ("adaptive", "v12 set 2")]
                 if a in runs and b in runs]:
        x, y, p = paired(runs[a], runs[b])
        comps[f"{a} vs {b}"] = {"only_a": x, "only_b": y, "mcnemar_p": p,
                                "diff": summ[a]["success"] - summ[b]["success"]}
        print(f"{a} vs {b}: {summ[a]['success'] - summ[b]['success']:+.3f}; runs only {a} succeeded {x}, "
              f"only {b} {y} -> McNemar p = {p:.3g}")

    cov = {k: np.array(covered(r, seg)) for k, r in runs.items()}
    prof = {k: [float(np.mean(c >= d - 1e-6)) for d in DISTANCES] for k, c in cov.items()}
    print("\nFraction of runs that covered at least d metres without a collision:")
    print(f"{'d (m)':>7s}" + "".join(f"{k:>12s}" for k in pols) + f"{'adapt/base':>12s}")
    for k in pols:
        prof[k].append(summ[k]["success"])
    for i, d in enumerate(DISTANCES + ["finished"]):
        ratio = prof["adaptive"][i] / prof["base"][i] if prof["base"][i] > 0 else float("nan")
        print(f"{d:>7}" + "".join(f"{prof[k][i]:12.3f}" for k in pols) + f"{ratio:12.2f}")
    print("\n(The ratio column shows whether the adaptive advantage grows or shrinks with distance.\n"
          " Timeouts count as covering the distance they reached.)")

    out = {"summary": summ, "paired": comps, "distance_profile": {"d_m": DISTANCES + ["finished"], **prof},
           "n_scenarios": len(test), "futures": args.futures, "end_x_m": END_X, "max_time_s": MAX_TIME,
           "base_params": BASE}
    (BASE_DIR / "longroad_summary.json").write_text(json.dumps(out, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 3.8))
        xs = DISTANCES + [END_X - float(np.mean([load_ego(s)["centre_x"] for s in test]))]
        for k, c in zip(pols, ("C0", "C1", "C7")):
            plt.plot(xs, prof[k], "-o", ms=3, color=c, label=k)
        plt.xlabel("distance driven (m)")
        plt.ylabel("fraction of runs without a collision")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(BASE_DIR / "longroad_profile.png", dpi=150)
        print(f"\nSaved {BASE_DIR / 'longroad_summary.json'} and {BASE_DIR / 'longroad_profile.png'}")
    except Exception as exc:
        print(f"\nSaved {BASE_DIR / 'longroad_summary.json'} (no figure: {exc})")


if __name__ == "__main__":
    main()
