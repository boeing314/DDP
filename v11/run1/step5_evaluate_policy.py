"""
Step 5 -- closed-loop test on FRESH scenarios (pipeline.seeds.test, never used in
steps 2-4).

At the takeover the model reads the encoding, predicts the success probability of
every library set and the ego drives the whole segment with the most likely set --
unless that set's predicted advantage over the best single set is below the margin
chosen in step 4, then with the best single set.
Because that decision is made once per segment, running EVERY set on every test
scenario gives the model's runs (its chosen set) and every comparison policy's runs
from the same simulations:
    set 0, best single set (from step 4), model, random set, oracle (hindsight).

Per-run comparison model vs best single: McNemar exact test on (scenario, future)
pairs where exactly one of the two succeeded.

Outputs:
    test_results.csv   per scenario: predicted probs, chosen set, success prob of every set
    test_summary.json

Usage:
    python step5_evaluate_policy.py
    python step5_evaluate_policy.py --futures 5 --n-jobs 7
    python step5_evaluate_policy.py --show s05003      # predicted probs + a command to watch it in sumo-gui
"""

import argparse
import json
import math
import time

import numpy as np

from common import (BASE_DIR, END_X, FUTURES, MAX_TIME, PIPE, TAU, existing, load_ego, load_model, make_pool,
                    predict_proba, prob, run_outcomes, seed_range)

OUT_CSV = BASE_DIR / "test_results.csv"
OUT_JSON = BASE_DIR / "test_summary.json"


def mcnemar(b, c):
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--futures", type=int, default=FUTURES)
    ap.add_argument("--seeds", type=int, nargs=2, default=None, help="override pipeline.seeds.test")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--show", type=str, default=None, help="scenario id: print the model's view and a GUI command")
    args = ap.parse_args()

    model = load_model()
    meta = model["meta"]
    sets, K, kbest, margin = meta["sets"], meta["K"], meta["best_single"], meta["margin"]

    if args.show:
        ego = load_ego(args.show)
        p = predict_proba(model, np.array([ego["encoding"]]))[0]
        k = int(p.argmax()) if p.max() - p[kbest] > margin else kbest
        print(f"{args.show}: start x={ego['centre_x']:.1f} m, y={ego['centre_y']:.2f} m, vx={ego['vx']:.1f} m/s")
        print("predicted success: " + "  ".join(f"set {i}: {v:.2f}" for i, v in enumerate(p)) + f"  -> set {k} (best single {kbest}, margin {margin:g})")
        print("watch it:\n  DIPF_PARAMS='" + json.dumps(sets[k]) + f"' python dipf_controller.py --episode "
              f"{BASE_DIR / 'scenarios' / args.show} --distance {END_X - ego['centre_x']:.1f} --max-time {MAX_TIME:g} --gui")
        return

    sids = existing(seed_range(args.seeds or PIPE["seeds"]["test"]))
    if not sids:
        raise SystemExit("No test scenarios -- run step 1 (it creates pipeline.seeds.test too)")
    X = np.array([load_ego(s)["encoding"] for s in sids])
    P = predict_proba(model, X)
    choice = P.argmax(1)
    choice = np.where(P[np.arange(len(P)), choice] - P[:, kbest] > margin, choice, kbest)
    futs = list(range(1, args.futures + 1))

    pool, n_jobs = make_pool(args.n_jobs)
    print("=" * 90)
    print(f"Step 5: closed-loop test on {len(sids)} fresh scenarios ({sids[0]}..{sids[-1]}), "
          f"{K} sets x {args.futures} futures = {len(sids) * K * args.futures:,} runs, {n_jobs} parallel")
    print("=" * 90, flush=True)
    t0 = time.time()
    try:
        runs = run_outcomes(pool, sets, sids, futs)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    print(f"done [{time.time() - t0:.0f}s]")

    Y = np.array([[prob(runs[k][s]) for k in range(K)] for s in sids])
    idx = np.arange(len(sids))
    pol = {"set 0": np.zeros(len(sids), int), f"best single (set {kbest})": np.full(len(sids), kbest),
           "model (argmax)": P.argmax(1), f"model + fallback (m={margin:g})": choice}
    summary = {name: {"success": float(Y[idx, c].mean()), "solved": float((Y[idx, c] >= TAU).mean())}
               for name, c in pol.items()}
    summary["random set"] = {"success": float(Y.mean()), "solved": float((Y >= TAU).mean())}
    summary["oracle (hindsight)"] = {"success": float(Y.max(1).mean()), "solved": float((Y.max(1) >= TAU).mean())}

    # McNemar on individual runs: model's set vs best single set
    b = c = 0
    for i, s in enumerate(sids):
        for om, ob in zip(runs[choice[i]][s], runs[kbest][s]):
            if om == om and ob == ob:
                b += int(om == 1.0 and ob == 0.0)
                c += int(om == 0.0 and ob == 1.0)
    d = Y[idx, choice] - Y[:, kbest]

    print(f"\n{'policy':<26s} {'mean success':>13s} {'solved':>8s}")
    for name, v in summary.items():
        print(f"{name:<26s} {v['success']:13.3f} {v['solved']:8.1%}")
    print(f"\nmodel + fallback - best single: {d.mean():+.3f} +- {d.std(ddof=1) / math.sqrt(len(d)):.3f} (SE) per scenario; "
          f"runs where only the model succeeded {b}, only best single {c} -> McNemar p = {mcnemar(b, c):.3g}")
    print("model + fallback picks: " + "  ".join(f"set {k}: {int((choice == k).sum())}" for k in range(K)))
    hit = (Y[idx, choice] >= TAU)
    conf = P[idx, choice]
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  predicted p of the chosen set in [{lo:.1f},{min(hi, 1):.1f}): {m.sum():4d} scenarios, "
                  f"actual success {Y[idx, choice][m].mean():.3f}, solved {hit[m].mean():.1%}")

    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write(",".join(["sid", "chosen"] + [f"pred_{k}" for k in range(K)] + [f"p_{k}" for k in range(K)]) + "\n")
        for i, s in enumerate(sids):
            f.write(",".join([s, str(choice[i])] + [f"{v:.4f}" for v in P[i]] + [f"{v:.4f}" for v in Y[i]]) + "\n")
    summary["paired_vs_best_single"] = {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
                                        "runs_only_model": b, "runs_only_best_single": c, "mcnemar_p": mcnemar(b, c)}
    summary["n_scenarios"], summary["futures"] = len(sids), args.futures
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
