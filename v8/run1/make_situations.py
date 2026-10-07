"""
Step 1 -- situation features for every episode's starting state.

For each episode in episodes/ (state.xml + ego.json) the saved SUMO state is
loaded, every other vehicle is read through TraCI (body centre from the front
bumper + heading, SUMO speed, SUMO lateral speed, length) and the slot-based
features of situation.py are computed with the ego's centre and controller
velocity from ego.json -- exactly what cluster_controller.py computes while
driving.

Output: situations.csv  (episode_id, seed, capture_label, <features>)

Usage:
    python make_situations.py
    python make_situations.py --n-jobs 6
"""

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from situation import FEATURES, situation_features

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
SIT = CONFIG["situations"]


def features_for_episode(ep_dir):
    sys.argv = [sys.argv[0]]
    import dipf_controller as dc                      # once per worker
    traci = dc.traci
    ego = json.loads((ep_dir / "ego.json").read_text(encoding="utf-8"))
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))

    dc.QUIET = True
    cmd = [dc.sumolib.checkBinary("sumo"), "-n", str(dc.NET_FILE), "--load-state", str(ep_dir / "state.xml"),
           "--step-length", str(dc.STEP_LENGTH), "--lateral-resolution", str(dc.LATERAL_RESOLUTION),
           "--no-step-log", "true", "--no-warnings", "true"]
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        dc.start_sumo_fast(cmd)
    try:
        vehicles = []
        for vid in traci.vehicle.getIDList():
            if vid == dc.EGO_ID:
                continue
            length = traci.vehicle.getLength(vid)
            x, y = dc.vehicle_center(vid, length)
            vehicles.append((x, y, traci.vehicle.getSpeed(vid), traci.vehicle.getLateralSpeed(vid), length))
    finally:
        dc.close_sumo_fast()

    feats = situation_features(
        (float(ego["centre_x"]), float(ego["centre_y"]), float(ego["vx"]), float(ego["vy"])),
        vehicles, dc.lane_centers_y(), dc.road_bounds_y(), dc.EGO_LENGTH, dc.EGO_WIDTH,
        float(SIT["slot_range_m"]), float(SIT["ttc_cap_s"]), float(SIT["density_range_m"]))
    return {"episode_id": ep_dir.name, "seed": meta["seed"], "capture_label": meta["capture_label"],
            **dict(zip(FEATURES, feats))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", default=str(BASE_DIR / SIT.get("episodes_dir", "episodes")))
    ap.add_argument("--out", default=str(BASE_DIR / "situations.csv"))
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    ep_root = Path(args.episodes_dir)
    index = list(csv.DictReader((ep_root / "index.csv").open(encoding="utf-8")))
    eps = [ep_root / r["episode_id"] for r in index if r["usable"] == "True"]
    print("=" * 90)
    print("Step 1: situation features")
    print(f"Episodes : {len(eps)} from {ep_root}")
    print(f"Features : {len(FEATURES)} -> {', '.join(FEATURES)}")
    print("=" * 90, flush=True)

    rows, fails = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(features_for_episode, e): e for e in eps}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                fails.append(futs[fut].name)
                print(f"  {futs[fut].name}: FAILED -- {exc}", flush=True)
            if i % 200 == 0 or i == len(eps):
                print(f"  {i}/{len(eps)} ({time.time() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["episode_id"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(fails)} failed) -> {args.out}")


if __name__ == "__main__":
    main()
