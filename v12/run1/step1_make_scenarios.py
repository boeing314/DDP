"""
Step 1 -- scenarios: one saved starting state per seed.

For each seed: traffic runs for ego.convert_time_s, the car nearest ego.convert_x_m
becomes the ego, and at that moment the full SUMO state (all vehicles + RNG) and
the controller-held ego state are saved, together with the 80-value traffic
encoding (= the model input). The run stops right there.

    scenarios/s<seed>/state.xml   SUMO state at the takeover
    scenarios/s<seed>/ego.json    ego state + "encoding" (80 values, encoding.FEATURES order)
    scenarios/s<seed>/meta.json   seed, start x, segment end, ego vehicle id

Every later step replays a scenario from state.xml with several random futures.
Existing scenarios are skipped (delete scenarios/ to rebuild).

Usage:
    python step1_make_scenarios.py                    # all seed ranges in config.yaml -> pipeline.seeds
    python step1_make_scenarios.py --seeds 0 99
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from concurrent.futures import as_completed

from common import CONFIG, END_X, PIPE, SCEN_DIR, existing, make_pool, scen_id, seed_range


def make(seed):
    import dipf_controller as dc
    out = SCEN_DIR / scen_id(seed)
    with tempfile.TemporaryDirectory() as tmp:
        cap = os.path.join(tmp, "cap")
        dc.CAPTURE_TIMES = [float(dc.CONVERT_TIME)]
        dc.CAPTURE_DIR = Path(cap)
        dc.CAPTURE_STOP = True
        dc.LOG_ENCODING = True
        try:
            dc.run_seed(seed, {}, output_dir=os.path.join(tmp, "res"), max_time=dc.CONVERT_TIME + 5.0)
        finally:
            dc.CAPTURE_TIMES, dc.CAPTURE_STOP, dc.LOG_ENCODING = [], False, False
        egos = sorted(os.listdir(cap)) if os.path.isdir(cap) else []
        ego_f = [f for f in egos if f.startswith("ego_")]
        if not ego_f:
            return seed, None
        ego = json.loads(open(os.path.join(cap, ego_f[0])).read())
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(cap, ego["state_file"]), out / "state.xml")
        (out / "ego.json").write_text(json.dumps(ego, indent=1))
        meta = {"seed": seed, "road_length_m": dc.ROAD_LENGTH_M, "ego_id": ego["ego_id"], "time": ego["time"],
                "start_x": ego["centre_x"],
                "start_y": ego["centre_y"], "start_vx": ego["vx"], "end_x": END_X,
                "segment_m": END_X - ego["centre_x"]}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return seed, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs=2, default=None, help="only this inclusive range")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    if args.seeds:
        seeds = seed_range(args.seeds)
    else:
        seeds = sorted({s for r in PIPE["seeds"].values() for s in seed_range(r)})
    existing(seeds)                                  # stops if scenarios/ is from another road length
    todo = [s for s in seeds if not (SCEN_DIR / scen_id(s) / "ego.json").exists()]
    SCEN_DIR.mkdir(parents=True, exist_ok=True)
    pool, n = make_pool(args.n_jobs)
    print("=" * 90)
    print("Step 1: scenarios (takeover at "
          f"t={CONFIG['ego']['convert_time_s']} s, car nearest x={CONFIG['ego']['convert_x_m']} m, segment to x={END_X:g} m)")
    print(f"Seeds: {len(seeds)} ({len(seeds) - len(todo)} already exist, {len(todo)} to create), {n} parallel")
    print("=" * 90, flush=True)
    t0, made, failed = time.time(), 0, []
    futs = {pool.submit(make, s): s for s in todo}
    for i, f in enumerate(as_completed(futs), 1):
        s = futs[f]
        try:
            _, meta = f.result()
            if meta is None:
                failed.append(s)
            else:
                made += 1
        except Exception as exc:
            failed.append(s)
            print(f"  seed {s}: FAILED -- {exc}", flush=True)
        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    pool.shutdown()
    print(f"Created {made} scenarios; {len(failed)} seeds without a usable car to take over"
          + (f" (e.g. {failed[:5]})" if failed else ""))


if __name__ == "__main__":
    main()
