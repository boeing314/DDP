"""
Step 4 -- what did the policy learn? Figures in figures/ and a text summary.

    fig_training.png      CMA-ES progress: best / median candidate J per generation, validation checkpoints
    fig_weights.png       W (parameter x feature, log-multiplier per unit feature) and the bias b
    fig_traces.png        3 test runs: traffic features and the parameter multipliers p/p0 over time
    fig_multipliers.png   distribution of p/p0 over all steps of the traced test runs

Usage:
    python step4_analyze.py --n-jobs 7
    python step4_analyze.py --policy policy.json --n-traces 40
"""

import argparse
import csv
import glob
import json
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import policy as pol  # noqa: E402
from common import BASE, BASE_DIR, END_X, MAX_TIME, SCEN_DIR, load_ego, make_pool, split_ids  # noqa: E402

FIG = BASE_DIR / "figures"


def _trace(job):
    sid, policy, fut = job
    import dipf_controller as dc
    dc.ADAPT_LOG = True
    with tempfile.TemporaryDirectory() as tmp:
        m = dc.run_episode(SCEN_DIR / sid, BASE, END_X - load_ego(sid)["centre_x"], MAX_TIME, output_dir=tmp,
                           sumo_seed=fut, policy=policy)
        traj = [(float(r["time"]), float(r["x"]), float(r["vx"])) for r in
                csv.DictReader(open(glob.glob(tmp + "/*/trajectory.csv")[0]))]
    return sid, m["status"], [list(map(float, r)) for r in dc.ADAPT_TRACE], traj


def fig_training(state, path):
    h = state["history"]
    if not h:
        return
    g = [r["gen"] for r in h]
    plt.figure(figsize=(8, 4.2))
    plt.plot(g, [r["J_best"] for r in h], "-o", ms=3, label="best candidate (train)")
    plt.plot(g, [r["J_median"] for r in h], "-", alpha=0.7, label="median candidate (train)")
    plt.axhline(state["base_train"]["J"], color="k", ls="--", lw=1, label="base parameters (train)")
    cp = [c for c in state["checkpoints"] if c["kind"] != "base"]
    if cp:
        plt.plot([c["gen"] for c in cp if c["kind"] == "mean"], [c["val"]["J"] for c in cp if c["kind"] == "mean"],
                 "s", color="C3", label="search mean (validation)")
    plt.axhline(state["base_val"]["J"], color="C3", ls=":", lw=1, label="base parameters (validation)")
    plt.xlabel("generation")
    plt.ylabel("J = success + w * progress")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


FEATURE_LABELS = {"ttc_front": "TTC ahead", "ttc_side_front": "TTC front side",
                  "ttc_rear_side": "TTC rear, sides", "gap_front": "Gap front",
                  "gap_side_front": "Gap front side", "tight_lateral": "Road clearance",
                  "speed": "Speed", "density": "Density"}


def fig_weights(policy, path):
    W, b = np.array(policy["W"]), np.array(policy["b"])
    feats = policy.get("features", pol.FEATURES)
    M = np.hstack([W, b[:, None]])
    lim = max(0.5, float(np.abs(M).max()))
    plt.figure(figsize=(10, 2.6 + 0.4 * len(policy["params"])))
    plt.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    plt.colorbar(label="log-multiplier (e^w per unit feature)")
    plt.xticks(range(M.shape[1]), [FEATURE_LABELS.get(f, f) for f in feats] + ["bias b"], rotation=35, ha="right",
               fontsize=8)
    plt.yticks(range(M.shape[0]), policy["params"], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    plt.title("learned weights: p = p0 * exp(clip(W f + b))")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_traces(traces, params, path):
    F = len(pol.FEATURES)
    show = [pol.FEATURES.index(k) for k in ("ttc_front", "gap_front", "tight_lateral", "ttc_rear_side")]
    fig, axes = plt.subplots(2, len(traces), figsize=(4.6 * len(traces), 6), squeeze=False, sharex="col")
    for c, (sid, status, tr, traj) in enumerate(traces):
        T = np.array(tr)
        if not len(T):
            continue
        t = T[:, 0] - T[0, 0]
        for j in show:
            axes[0, c].plot(t, T[:, 1 + j], label=pol.FEATURES[j])
        axes[0, c].set_title(f"{sid}: {status}", fontsize=10)
        axes[0, c].set_ylim(-0.05, 1.05)
        for k, p in enumerate(params):
            axes[1, c].plot(t, T[:, 1 + F + k] / BASE[p], label=p)
        axes[1, c].set_yscale("log")
        axes[1, c].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[1, c].set_xlabel("time since takeover (s)")
    axes[0, 0].set_ylabel("feature value")
    axes[1, 0].set_ylabel("p / p0")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_multipliers(traces, params, path):
    F = len(pol.FEATURES)
    rows = np.vstack([np.array(tr) for _, _, tr, _ in traces if len(tr)])
    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3.2), squeeze=False)
    for k, p in enumerate(params):
        m = rows[:, 1 + F + k] / BASE[p]
        axes[0, k].hist(np.log10(m), bins=40, color=f"C{k}")
        axes[0, k].set_title(p, fontsize=10)
        axes[0, k].set_xlabel("log10(p / p0)")
    axes[0, 0].set_ylabel("control steps")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="policy.json")
    ap.add_argument("--n-traces", type=int, default=30, help="test scenarios traced (future 1)")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    pj = json.loads((BASE_DIR / args.policy).read_text())
    policy = pj["policy"]
    state_f = BASE_DIR / ("cmaes_state_biasonly.json" if pj.get("bias_only") else "cmaes_state.json")

    if state_f.exists():
        fig_training(json.loads(state_f.read_text()), FIG / "fig_training.png")
    fig_weights(policy, FIG / "fig_weights.png")

    test = split_ids("test")[:args.n_traces]
    pool, _ = make_pool(args.n_jobs)
    traces = list(pool.map(_trace, [(sid, policy, 1) for sid in test]))
    pool.shutdown()
    # three examples: prefer runs where the parameters moved the most
    moved = sorted(traces, key=lambda r: -np.ptp(np.log(np.array(r[2])[:, 1 + len(pol.FEATURES):]), axis=0).max()
                   if len(r[2]) else 0)
    fig_traces(moved[:3], policy["params"], FIG / "fig_traces.png")
    fig_multipliers(traces, policy["params"], FIG / "fig_multipliers.png")

    print("Policy:", pj.get("summary", ""))
    print(f"Selected: {pj['selected']['kind']} of generation {pj['selected']['gen']}, validation J "
          f"{pj['selected']['val']['J']:.4f} (base {pj['base']['val']['J']:.4f})")
    F = len(pol.FEATURES)
    rows = np.vstack([np.array(tr) for _, _, tr, _ in traces if len(tr)])
    print(f"Parameter multipliers over {len(rows)} control steps of {len(traces)} test runs (p/p0):")
    for k, p in enumerate(policy["params"]):
        m = rows[:, 1 + F + k] / BASE[p]
        print(f"  {p:<17s} median x{np.median(m):.2f}   5-95%: x{np.percentile(m, 5):.2f} .. x{np.percentile(m, 95):.2f}")
    print(f"Figures in {FIG}")


if __name__ == "__main__":
    main()
