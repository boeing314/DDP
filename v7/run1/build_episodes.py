"""
Collect every captured state (states/seed_*/) into one flat episode folder.

Each captured state is one episode -- a seed with 4 captures gives 4
episodes. For every episode this writes

    episodes/<episode_id>/state.xml   the SUMO state (copied)
    episodes/<episode_id>/ego.json    the controller-held ego state (copied)
    episodes/<episode_id>/meta.json   where it came from + segment info

and one episodes/index.csv. <episode_id> = s<seed:04d>_c<k>_<label>, e.g.
s0042_c2_between_2.

The segment to drive in an episode is `distance_m` metres (default 150) from
the ego's centre in the state, capped at the road goal; episodes whose
capped segment is shorter than `min_distance_m` are still copied but marked
usable = false (optimize_episodes.py skips them).

Usage:
    python build_episodes.py                 # states/ -> episodes/
    python build_episodes.py --distance 150 --min-distance 50
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
SIM, EGO = CONFIG["simulation"], CONFIG["ego"]
ROAD_GOAL_X = float(SIM["road_length_m"]) - float(EGO["goal_x_offset_from_end_m"])
EPI = CONFIG.get("episode_opt", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states-dir", default=str(BASE_DIR / CONFIG["capture"].get("output_dir", "states")))
    parser.add_argument("--out-dir", default=str(BASE_DIR / EPI.get("episodes_dir", "episodes")))
    parser.add_argument("--distance", type=float, default=float(EPI.get("distance_m", 150.0)))
    parser.add_argument("--min-distance", type=float, default=float(EPI.get("min_distance_m", 50.0)))
    args = parser.parse_args()

    states_dir, out_dir = Path(args.states_dir), Path(args.out_dir)
    infos = sorted((json.loads(f.read_text(encoding="utf-8")) for f in states_dir.glob("seed_*/info.json")),
                   key=lambda r: r["seed"])
    if not infos:
        raise SystemExit(f"No captured seeds found in {states_dir}")

    print("=" * 90)
    print("Building episode dataset")
    print(f"States from : {states_dir}  ({len(infos)} seeds)")
    print(f"Episodes to : {out_dir}")
    print(f"Segment     : {args.distance:g} m from the ego's start (capped at road goal x={ROAD_GOAL_X:g}); "
          f"usable if >= {args.min_distance:g} m")
    print("=" * 90)

    if out_dir.exists():
        print(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows = []
    for info in infos:
        seed = info["seed"]
        for k, cap in enumerate(info["captures"]):
            if not cap.get("state_file"):
                continue
            src = states_dir / f"seed_{seed}"
            ego = json.loads((src / cap["ego_file"]).read_text(encoding="utf-8"))
            ep_id = f"s{seed:04d}_c{k}_{cap['label']}"
            ep_dir = out_dir / ep_id
            ep_dir.mkdir()
            shutil.copy(src / cap["state_file"], ep_dir / "state.xml")
            shutil.copy(src / cap["ego_file"], ep_dir / "ego.json")

            x0 = float(ego["centre_x"])
            target_x = min(x0 + args.distance, ROAD_GOAL_X)
            seg = target_x - x0
            meta = {
                "episode_id": ep_id,
                "seed": seed,
                "capture_index": k,
                "capture_label": cap["label"],
                "time": ego["time"],
                "original_outcome": info["outcome"],
                "original_t_end": info["t_end"],
                "original_time_to_end": cap["time_to_end"],
                "start_x": x0,
                "start_y": float(ego["centre_y"]),
                "start_vx": float(ego["vx"]),
                "start_vy": float(ego["vy"]),
                "target_x": target_x,
                "segment_m": seg,
                "segment_capped": seg < args.distance,
                "usable": seg >= args.min_distance,
                "n_vehicles": ego.get("n_vehicles"),
            }
            (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rows.append(meta)

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_label = {}
    for r in rows:
        by_label[r["capture_label"]] = by_label.get(r["capture_label"], 0) + 1
    n_capped = sum(r["segment_capped"] for r in rows)
    n_unusable = sum(not r["usable"] for r in rows)
    print(f"Episodes written : {len(rows)}")
    for lab, n in by_label.items():
        print(f"  {lab:<18} {n}")
    print(f"Segment capped by road end : {n_capped}   (shorter than {args.distance:g} m)")
    print(f"Unusable (< {args.min_distance:g} m left)   : {n_unusable}")
    print(f"Index : {out_dir / 'index.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
