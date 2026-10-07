"""
Step 4 -- DIPF controller that picks its potential coefficients by traffic situation.

Same controller as dipf_controller.py, plus one hook before each control step:
  1. slot-based situation features of the current traffic (situation.py -- the
     same function the clusters were built from)
  2. nearest situation-cluster centre (clusters.npz), with hysteresis: the
     cluster only changes if the new centre is clearly nearer
  3. that cluster's optimised parameters (cluster_opt_results/cluster_params.json;
     clusters that were not optimised use the v6 fallback, tau_x_base fixed)

Per-step cluster and parameters are written to <results>/params.csv.

Usage:
    python cluster_controller.py --seed 1000                       # full road, prints result
    python cluster_controller.py --seed 1000 --gui
    python cluster_controller.py --episode episodes/s0007_c0_start --distance 300
    python cluster_controller.py --eval-test                        # held-out test episodes vs fixed v6
    python cluster_controller.py --eval-seeds 1000 1499             # full road on fresh seeds vs fixed v6

Options: --update-every N (re-identify the cluster every N steps, default 10 = 1 s)
         --hysteresis H   (switch only if the new centre is >= H x nearer, default 0.1)
"""

import argparse
import csv
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

import dipf_controller as dc
from situation import FEATURES, situation_features

BASE_DIR = Path(__file__).resolve().parent
traci, tc = dc.traci, dc.tc
SIT = dc.CONFIG["situations"]
OPT = dc.CONFIG["cluster_opt"]
FEATURE_RADIUS = 5000.0      # context-subscription radius > road length: every vehicle is visible


class ClusterPolicy:
    def __init__(self, clusters_file, params_file, update_every=10, hysteresis=0.1):
        z = np.load(clusters_file)
        if list(z["features"]) != FEATURES:
            raise ValueError(f"{clusters_file} was built with different features")
        self.centres, self.mean, self.std = z["centres"], z["mean"], z["std"]
        cp = json.loads(Path(params_file).read_text(encoding="utf-8"))
        self.params = [cp["clusters"][str(j)]["params"] for j in range(len(self.centres))]
        self.sources = [cp["clusters"][str(j)]["source"] for j in range(len(self.centres))]
        self.update_every = max(1, int(update_every))
        self.hysteresis = float(hysteresis)
        self.reset()

    def reset(self):
        self.step, self.current, self.log = 0, None, []

    def features(self, x, y, vx, vy):
        ctx = traci.vehicle.getContextSubscriptionResults(dc.EGO_ID) or {}
        vehicles = []
        for vid, r in ctx.items():
            if vid == dc.EGO_ID:
                continue
            length = dc._vehicle_static(vid)[3]
            xf, yf = r[tc.VAR_POSITION]
            xo, yo = dc.front_to_center(xf, yf, length, r[tc.VAR_ANGLE])
            vehicles.append((xo, yo, r[tc.VAR_SPEED], r[tc.VAR_SPEED_LAT], length))
        return np.array(situation_features(
            (x, y, vx, vy), vehicles, dc.lane_centers_y(), dc.road_bounds_y(), dc.EGO_LENGTH, dc.EGO_WIDTH,
            float(SIT["slot_range_m"]), float(SIT["ttc_cap_s"]), float(SIT["density_range_m"])))

    def update(self, x, y, vx, vy):
        changed = False
        if self.current is None or self.step % self.update_every == 0:
            z = (self.features(x, y, vx, vy) - self.mean) / self.std
            d = np.sqrt(((self.centres - z) ** 2).sum(1))
            best = int(np.argmin(d))
            if self.current is None or d[best] < (1.0 - self.hysteresis) * d[self.current]:
                changed = self.current is not None and best != self.current
                self.current = best
        self.step += 1
        p = self.params[self.current]
        self.log.append({"time": traci.simulation.getTime(), "cluster": self.current, "changed": changed, **p})
        return p


POLICY = None
_original_step = dc.ego_control_step


def cluster_ego_control_step(dt, hist):
    x, y = dc.vehicle_center(dc.EGO_ID, dc.EGO_LENGTH)
    dc.set_dipf_params(POLICY.update(x, y, cluster_ego_control_step.vx, cluster_ego_control_step.vy))
    return _original_step(dt, hist)


def install(update_every=10, hysteresis=0.1, clusters_file=None, params_file=None):
    global POLICY
    POLICY = ClusterPolicy(clusters_file or BASE_DIR / "clusters.npz",
                           params_file or BASE_DIR / OPT.get("results_dir", "cluster_opt_results") / "cluster_params.json",
                           update_every, hysteresis)
    # controller state lives on the function object named ego_control_step (see v7 mlp_dipf_controller)
    for k in ("prev_ax", "prev_ay", "vx", "vy", "angle"):
        setattr(cluster_ego_control_step, k, getattr(dc.ego_control_step, k))
    dc.ego_control_step = cluster_ego_control_step
    dc.context_radius = lambda: FEATURE_RADIUS


def write_log(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    keys = ["time", "cluster", "changed"] + sorted({k for r in POLICY.log for k in r} - {"time", "cluster", "changed"})
    with (results_dir / "params.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(POLICY.log)


def v6_params():
    from capture_states import load_v6_params
    v6, src = load_v6_params((BASE_DIR / dc.CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve())
    return {k: float(v) for k, v in v6.items()}, src


# =============================================================================
# Batch evaluation (in-process, fresh SUMO per run)
# =============================================================================

PARAMS_FILE = None   # set by --params


def _init(use_policy, update_every, hysteresis, params_file=None):
    sys.argv = [sys.argv[0]]
    if use_policy:
        install(update_every, hysteresis, params_file=params_file)


def _job(job):
    kind, key, params, use_policy, a, b, out_root = job          # kind: "episode" | "seed"
    if use_policy:
        POLICY.reset()
    out = Path(out_root) / str(key)
    try:
        if kind == "episode":
            m = dc.run_episode(BASE_DIR / SIT.get("episodes_dir", "episodes") / key, params, a, b, output_dir=out)
        else:
            m = dc.run_seed(int(key), params, output_dir=out, max_time=a)
    except Exception as exc:
        return {"key": key, "status": "error", "C": math.nan, "progress": math.nan}
    row = {"key": key, "status": m["status"], "C": float(m["C"]), "progress": float(m["final_progress"])}
    if use_policy:
        run_dir = out / dc.results_run_name()
        write_log(run_dir)
        cl = [r["cluster"] for r in POLICY.log]
        row["switches"] = sum(x != y for x, y in zip(cl, cl[1:]))
        row["clusters_used"] = len(set(cl))
    return row


def compare(kind, keys, a, b, update_every, hysteresis, n_jobs, out_dir, label):
    v6, src = v6_params()
    print("=" * 96)
    print(f"{label}: {len(keys)} runs each | cluster policy (update every {update_every} steps, "
          f"hysteresis {hysteresis}) vs fixed v6 ({src})")
    print("=" * 96, flush=True)
    res = {}
    for name, use_policy, params in (("cluster", True, {}), ("v6", False, v6)):
        t0 = time.time()
        with ProcessPoolExecutor(n_jobs, initializer=_init,
                                 initargs=(use_policy, update_every, hysteresis, PARAMS_FILE)) as ex:
            res[name] = list(ex.map(_job, [(kind, k, params, use_policy, a, b, out_dir / name) for k in keys], chunksize=1))
        print(f"  {name:7s} done in {time.time() - t0:.0f}s", flush=True)

    def summ(rows):
        st = [r["status"] for r in rows]
        return {"success": st.count("success") / len(st), "collision": st.count("collision") / len(st),
                "other": 1 - (st.count("success") + st.count("collision")) / len(st), "errors": st.count("error"),
                "progress": float(np.nanmean([r["progress"] for r in rows])),
                "C": float(np.nanmean([r["C"] for r in rows]))}

    s_c, s_v = summ(res["cluster"]), summ(res["v6"])
    print(f"\n{'':16s} {'success':>8s} {'collision':>10s} {'timeout':>8s} {'progress':>9s} {'mean C':>8s}")
    for lab, s in (("cluster policy", s_c), ("fixed v6", s_v)):
        print(f"{lab:16s} {s['success']:8.1%} {s['collision']:10.1%} {s['other']:8.1%} {s['progress']:9.3f} {s['C']:8.3f}"
              + (f"  ({s['errors']} errors)" if s["errors"] else ""))
    only_c = sum(x["status"] == "success" and y["status"] != "success" for x, y in zip(res["cluster"], res["v6"]))
    only_v = sum(x["status"] != "success" and y["status"] == "success" for x, y in zip(res["cluster"], res["v6"]))
    n_disc = only_c + only_v
    p = min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(0, min(only_c, only_v) + 1)) / 2 ** n_disc) if n_disc else 1.0
    print(f"\npaired: only cluster policy succeeds {only_c}, only v6 succeeds {only_v}  "
          f"(exact McNemar p = {p:.3f})")
    sw = [r.get("switches", 0) for r in res["cluster"]]
    print(f"cluster switches per run: mean {np.mean(sw):.1f}, max {max(sw)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "cluster_status", "cluster_progress", "cluster_C", "switches", "v6_status", "v6_progress", "v6_C"])
        for x, y in zip(res["cluster"], res["v6"]):
            w.writerow([x["key"], x["status"], x["progress"], x["C"], x.get("switches", ""), y["status"], y["progress"], y["C"]])
    (out_dir / "summary.json").write_text(json.dumps({"cluster": s_c, "v6": s_v, "only_cluster": only_c,
                                                      "only_v6": only_v, "mcnemar_p": p}, indent=2))
    print(f"Saved {out_dir / 'summary.csv'}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--distance", type=float, default=float(OPT["distance_m"]))
    ap.add_argument("--max-time", type=float, default=float(OPT["max_time_s"]))
    ap.add_argument("--update-every", type=int, default=10, help="re-identify the cluster every N steps (default 10)")
    ap.add_argument("--hysteresis", type=float, default=0.1,
                    help="switch only if the new centre is closer by this fraction (default 0.1)")
    ap.add_argument("--eval-test", action="store_true", help="test-split episodes: cluster policy vs fixed v6")
    ap.add_argument("--eval-seeds", type=int, nargs=2, metavar=("START", "END"), default=None,
                    help="full-road runs on seeds START..END: cluster policy vs fixed v6")
    ap.add_argument("--eval-max-time", type=float, default=300.0, help="full-road limit after ego departure (s)")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--eval-out", default=None)
    ap.add_argument("--params", default=None, help="cluster_params.json (default: cluster_opt.results_dir)")
    args = ap.parse_args()
    global PARAMS_FILE
    PARAMS_FILE = args.params

    if args.eval_test:
        asg = pd.read_csv(BASE_DIR / "cluster_assignment.csv")
        keys = asg[asg.split == "test"].episode_id.tolist()
        compare("episode", keys, args.distance, args.max_time, args.update_every, args.hysteresis, args.n_jobs,
                Path(args.eval_out or BASE_DIR / "eval_test"), f"Test episodes ({args.distance:g} m)")
        return
    if args.eval_seeds:
        keys = list(range(args.eval_seeds[0], args.eval_seeds[1] + 1))
        compare("seed", keys, args.eval_max_time, None, args.update_every, args.hysteresis, args.n_jobs,
                Path(args.eval_out or BASE_DIR / f"eval_seeds_{keys[0]}_{keys[-1]}"),
                f"Full road (goal x={dc.EGO_GOAL_X:g} m), seeds {keys[0]}..{keys[-1]}")
        return

    install(args.update_every, args.hysteresis, params_file=args.params)
    if args.episode:
        dc.EPISODE_DIR = Path(args.episode).resolve()
        dc.EPISODE_EGO = json.loads((dc.EPISODE_DIR / "ego.json").read_text(encoding="utf-8"))
        dc.EPISODE_MAX_TIME = args.max_time
        dc.SEED = dc.EPISODE_EGO.get("seed")
        dc.EGO_START_X = float(dc.EPISODE_EGO["centre_x"])
        dc.EGO_GOAL_X = min(dc.EGO_START_X + args.distance, dc.ROAD_LENGTH_M - dc.GOAL_X_OFFSET)
        dc.SEGMENT_LENGTH_M = dc.EGO_GOAL_X - dc.EGO_START_X
        dc.RUN_NAME = f"cluster_results_{dc.EPISODE_DIR.name}"
    else:
        if args.seed is not None:
            dc.SEED = args.seed
        dc.RUN_NAME = f"cluster_results_{dc.SEED}"
    dc.SUMO_GUI = args.gui
    print(f"Cluster policy: {len(POLICY.centres)} clusters, update every {args.update_every} steps, "
          f"hysteresis {args.hysteresis}")
    dc.run_simulation()
    out = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results"))) / dc.RUN_NAME
    write_log(out)
    cl = [r["cluster"] for r in POLICY.log]
    print(f"Clusters used: {sorted(set(cl))}, switches: {sum(a != b for a, b in zip(cl, cl[1:]))} -> {out / 'params.csv'}")


if __name__ == "__main__":
    main()
