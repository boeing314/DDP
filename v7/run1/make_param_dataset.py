"""
Build a CSV dataset from the per-episode optimisation results: the ego's starting
state, the 6 nearest vehicles (skewed distance), and that episode's best DIPF
parameters.

For every result in episode_opt_results/<episode_id>.json:

  ego      x_ego, y_ego     ego body centre at the episode start (ego.json)
           vx_ego, vy_ego   controller-held ego velocity (ego.json)
  vehicles every other vehicle in the episode's SUMO state, read through TraCI
           exactly as the controller sees obstacles: body centre (front bumper
           converted with length + heading), vx = SUMO speed, vy = SUMO
           lateral speed
  nearest  the K (default 6) vehicles with the smallest skewed distance
               d = factor * hypot(x_veh - x_ego, y_veh - y_ego)
               factor = 0.33 if x_veh > x_ego (centre ahead of the ego), else 1
           ordered nearest first; relative values are vehicle - ego. Fewer than
           K vehicles in the scene -> the remaining columns are left empty.
  params   the episode's best parameters (best.params)

Columns:
    x_ego, y_ego, vx_ego, vy_ego,
    x_rel_veh1, y_rel_veh1, vx_rel_veh1, vy_rel_veh1, ... (up to veh6),
    lambda_obs, alpha, tau_x_base, tau_y_base, sx, delta_lane, obs_radius_ahead, y_margin

Usage:
    python make_param_dataset.py                       # -> episode_param_dataset.csv
    python make_param_dataset.py --with-id             # also episode_id / outcome columns (first)
    python make_param_dataset.py --k 6 --ahead-factor 0.33 --n-jobs 4
"""

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARAM_COLS = ["lambda_obs", "alpha", "tau_x_base", "tau_y_base", "sx", "delta_lane", "obs_radius_ahead", "y_margin"]


def vehicles_in_state(ep_dir):
    """Load the episode's SUMO state and return the ego centre + every other vehicle."""
    sys.argv = [sys.argv[0]]
    import dipf_controller as dc          # once per worker process
    traci, tc = dc.traci, dc.tc

    dc.QUIET = True
    cmd = [dc.sumolib.checkBinary("sumo"), "-n", str(dc.NET_FILE), "--load-state", str(ep_dir / "state.xml"),
           "--step-length", str(dc.STEP_LENGTH), "--lateral-resolution", str(dc.LATERAL_RESOLUTION),
           "--no-step-log", "true", "--no-warnings", "true"]
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        dc.start_sumo_fast(cmd)           # silences traci's per-retry messages
    try:
        ego_xy = dc.vehicle_center(dc.EGO_ID, dc.EGO_LENGTH)
        out = []
        for vid in traci.vehicle.getIDList():
            if vid == dc.EGO_ID:
                continue
            length = traci.vehicle.getLength(vid)
            x, y = dc.vehicle_center(vid, length)
            out.append((vid, x, y, traci.vehicle.getSpeed(vid), traci.vehicle.getLateralSpeed(vid)))
    finally:
        traci.close()
    return ego_xy, out


def row_for_episode(result_file, episodes_dir, k, factor, with_id):
    r = json.loads(Path(result_file).read_text(encoding="utf-8"))
    ep_id = r["episode_id"]
    ep_dir = episodes_dir / ep_id
    ego = json.loads((ep_dir / "ego.json").read_text(encoding="utf-8"))
    xe, ye, vxe, vye = float(ego["centre_x"]), float(ego["centre_y"]), float(ego["vx"]), float(ego["vy"])

    (sx_e, sy_e), vehicles = vehicles_in_state(ep_dir)
    if abs(sx_e - xe) > 1e-3 or abs(sy_e - ye) > 1e-3:
        raise RuntimeError(f"{ep_id}: ego centre in state ({sx_e:.3f},{sy_e:.3f}) != ego.json ({xe:.3f},{ye:.3f})")

    def skewed(v):
        _, x, y, _, _ = v
        d = math.hypot(x - xe, y - ye)
        return (factor * d if x > xe else d), v[0]

    nearest = sorted(vehicles, key=skewed)[:k]

    row = {}
    if with_id:
        row.update({"episode_id": ep_id, "best_status": r["best"]["result"]["status"],
                    "best_C": r["best"]["result"]["C"]})
    row.update({"x_ego": xe, "y_ego": ye, "vx_ego": vxe, "vy_ego": vye})
    for n in range(k):
        if n < len(nearest):
            _, x, y, vx, vy = nearest[n]
            vals = (x - xe, y - ye, vx - vxe, vy - vye)
        else:
            vals = ("", "", "", "")
        for name, val in zip(("x_rel", "y_rel", "vx_rel", "vy_rel"), vals):
            row[f"{name}_veh{n + 1}"] = val
    for p in PARAM_COLS:
        row[p] = r["best"]["params"][p]
    return ep_id, row, len(vehicles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(BASE_DIR / "episode_opt_results"))
    ap.add_argument("--episodes-dir", default=str(BASE_DIR / "episodes"))
    ap.add_argument("--out", default=str(BASE_DIR / "episode_param_dataset.csv"))
    ap.add_argument("--k", type=int, default=6, help="number of nearest vehicles (default 6)")
    ap.add_argument("--ahead-factor", type=float, default=0.33,
                    help="distance multiplier for vehicles whose centre is ahead of the ego (default 0.33)")
    ap.add_argument("--with-id", action="store_true", help="prepend episode_id, best_status, best_C columns")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 3))
    args = ap.parse_args()

    results = sorted(Path(args.results_dir).glob("s*_c*.json"))
    episodes_dir = Path(args.episodes_dir)
    print("=" * 90)
    print("Episode -> parameter dataset")
    print(f"Results     : {len(results)} in {args.results_dir}")
    print(f"Nearest     : {args.k} vehicles by skewed distance (x {args.ahead_factor} if centre ahead of ego, x 1 otherwise)")
    print(f"Output      : {args.out}")
    print(f"Parallelism : {args.n_jobs}")
    print("=" * 90, flush=True)
    if not results:
        raise SystemExit("no result files found")

    rows, failures, n_short = {}, [], 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(row_for_episode, f, episodes_dir, args.k, args.ahead_factor, args.with_id): f for f in results}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                ep_id, row, n_veh = fut.result()
                rows[ep_id] = row
                n_short += n_veh < args.k
            except Exception as exc:
                failures.append((futs[fut].stem, str(exc)))
                print(f"  {futs[fut].stem}: FAILED -- {exc}", flush=True)
            if i % 100 == 0 or i == len(results):
                print(f"  {i}/{len(results)} episodes ({time.time() - t0:.0f}s)", flush=True)

    ordered = [rows[k] for k in sorted(rows)]            # episode_id order
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered[0]))
        w.writeheader()
        w.writerows(ordered)

    print("=" * 90)
    print(f"Rows written : {len(ordered)}  ({len(failures)} failed, {n_short} with fewer than {args.k} other vehicles)")
    print(f"Columns      : {len(ordered[0])}")
    print(f"File         : {args.out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
