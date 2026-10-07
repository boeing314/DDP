"""
Starting-condition capture (v7).

Reads the optimised dipf.* parameters from v6, then for every seed in a range:

  1. Pass 1: runs the episode once and records when the ego starts (its first
     control step) and when the episode ends (goal reached, collision, stuck
     or timeout).
  2. Picks the capture times:
         t_start                      ego just started
         t_last = t_end - lead_time   lead_time_s before the finish/crash
         n_between times              equally spaced between the two
     (all snapped to the simulation step grid).
  3. Pass 2: re-runs the same seed with the same parameters (SUMO and the
     controller are deterministic) and saves at each capture time:
         state_t<time>.xml   full SUMO state, incl. RNG (traci saveState)
         ego_t<time>.json    controller-held ego state (vx, vy, prev ax/ay,
                             heading) that the SUMO state does not contain
     Pass 2 must end the same way as pass 1 (same status and end time);
     otherwise the seed is flagged as non-deterministic.

If the episode is shorter than lead_time_s, there is no "lead_time before the
end" moment after the start, so only the start state is captured and the seed
is flagged `short_episode`.

Output (capture.output_dir, default states/):
    params_used.json        the dipf.* values used and where they came from
    index.csv               one row per seed
    seed_<s>/info.json      outcome, times, capture list for that seed
    seed_<s>/state_t*.xml, ego_t*.json

Usage:
    python capture_states.py                    # seeds from config.yaml -> capture
    python capture_states.py --seeds 0 9        # seeds 0..9 inclusive
    python capture_states.py --n-jobs 4
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONTROLLER = BASE_DIR / "dipf_controller.py"
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
CAP = CONFIG["capture"]
STEP = float(CONFIG["simulation"]["step_length"])

# Sections whose values must match between v6 and v7 for v6's optimised
# parameters to mean the same thing here (dipf.* tuned keys are overridden
# from v6 anyway).
COMPARED_SECTIONS = ("simulation", "ego", "dipf", "cost", "movement",
                     "vehicle_types", "flows", "obstacle_effective_dimensions")


# =============================================================================
# Optimised parameters from v6
# =============================================================================

def load_v6_params(v6_dir):
    """
    Returns (params, source). Preference order:
      1. best_params.json         -- a finished optimize.py run
      2. optimization_log.jsonl   -- lowest-cost accepted evaluation, i.e. the
                                     current population best of an unfinished
                                     run (same as de_checkpoint.npz's best,
                                     but stored with parameter names)
      3. v6 config.yaml           -- dipf.* values (e.g. after --apply)
    """
    best_file = v6_dir / "best_params.json"
    if best_file.exists():
        data = json.loads(best_file.read_text(encoding="utf-8"))
        return data["best_params"], f"{best_file} (C={data.get('best_C')})"

    log_file = v6_dir / "optimization_log.jsonl"
    if log_file.exists():
        rows = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        accepted = [r for r in rows if r.get("accepted") and r.get("cost") == r.get("cost")]
        if accepted:
            best = min(accepted, key=lambda r: r["cost"])
            return best["params"], (f"{log_file} (unfinished run: best accepted evaluation, "
                                    f"generation {best['generation']}, member {best['member']}, "
                                    f"C={best['cost']:.4f})")

    v6_cfg = yaml.safe_load((v6_dir / "config.yaml").read_text(encoding="utf-8"))
    keys = list(v6_cfg.get("optimization", {}).get("bounds", {})) or list(v6_cfg["dipf"])
    return {k: v6_cfg["dipf"][k] for k in keys}, f"{v6_dir / 'config.yaml'} (dipf section)"


def config_differences(v6_dir, tuned):
    """Settings (other than the tuned dipf.* keys) where v6 and v7 differ."""
    v6_cfg = yaml.safe_load((v6_dir / "config.yaml").read_text(encoding="utf-8"))
    diffs = []
    for sec in COMPARED_SECTIONS:
        a, b = v6_cfg.get(sec, {}), CONFIG.get(sec, {})
        for k in sorted(set(a) | set(b)):
            if sec == "dipf" and k in tuned:
                continue
            if a.get(k) != b.get(k):
                diffs.append(f"{sec}.{k}: v6={a.get(k)!r}  v7={b.get(k)!r}")
    return diffs


# =============================================================================
# One seed
# =============================================================================

def snap(t):
    return round(round(t / STEP) * STEP, 6)


def run_controller(seed, params, out_dir, extra_args=()):
    env = os.environ.copy()
    env["DIPF_PARAMS"] = json.dumps(params)
    env["DIPF_OUTPUT_DIR"] = str(out_dir)
    proc = subprocess.run(
        [sys.executable, str(CONTROLLER), "--seed", str(seed), *extra_args],
        cwd=str(BASE_DIR), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed}: controller failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    res = out_dir / f"results_{seed}"
    metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8"))
    with (res / "trajectory.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"seed {seed}: ego never started (empty trajectory)")
    return metrics, float(rows[0]["time"]), float(rows[-1]["time"])


def capture_seed(seed, params, out_root, lead, n_between):
    seed_dir = out_root / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix=f"v7_seed{seed}_") as tmp:
        tmp = Path(tmp)

        # Pass 1: find start and end
        m1, t_start, t_end = run_controller(seed, params, tmp / "pass1")

        t_last = t_end - lead
        short = t_last <= t_start
        if short:
            times = [snap(t_start)]
            labels = ["start"]
        else:
            n = n_between + 1
            times = [snap(t_start + k * (t_last - t_start) / n) for k in range(n + 1)]
            labels = ["start"] + [f"between_{k}" for k in range(1, n_between + 1)] + [f"end_minus_{lead:g}s"]

        # Pass 2: same run, saving the states
        m2, t_start2, t_end2 = run_controller(
            seed, params, tmp / "pass2",
            ["--capture-times", ",".join(f"{t:.6f}" for t in times), "--capture-dir", str(seed_dir)],
        )

    deterministic = (m1["status"] == m2["status"]) and math.isclose(t_end, t_end2, abs_tol=1e-6)

    captures = []
    for label, t_req in zip(labels, times):
        ego_file = seed_dir / f"ego_t{t_req:.1f}.json"
        ego = json.loads(ego_file.read_text(encoding="utf-8")) if ego_file.exists() else None
        captures.append({
            "label": label,
            "time": t_req,
            "time_to_end": round(t_end - t_req, 6),
            "state_file": ego["state_file"] if ego else None,
            "ego_file": ego_file.name if ego else None,
            "ego_centre": [ego["centre_x"], ego["centre_y"]] if ego else None,
            "ego_v": [ego["vx"], ego["vy"]] if ego else None,
        })

    info = {
        "seed": seed,
        "outcome": m1["status"],
        "final_progress": m1["final_progress"],
        "t_start": t_start,
        "t_end": t_end,
        "duration": round(t_end - t_start, 6),
        "short_episode": short,
        "deterministic": deterministic,
        "pass2_outcome": m2["status"],
        "pass2_t_end": t_end2,
        "captures": captures,
    }
    (seed_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("START", "END"),
                        default=[int(CAP["seed_start"]), int(CAP["seed_end"])],
                        help="inclusive seed range (default capture.seed_start / seed_end)")
    parser.add_argument("--n-jobs", type=int,
                        default=int(CAP.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--lead-time", type=float, default=float(CAP.get("lead_time_s", 10.0)))
    parser.add_argument("--n-between", type=int, default=int(CAP.get("n_between", 2)))
    parser.add_argument("--output-dir", type=str, default=str(CAP.get("output_dir", "states")))
    args = parser.parse_args()

    v6_dir = (BASE_DIR / CAP.get("v6_dir", "../../v6/run1")).resolve()
    params, source = load_v6_params(v6_dir)
    out_root = (BASE_DIR / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))

    print("=" * 90)
    print("v7 state capture")
    print(f"Params from : {source}")
    for k, v in params.items():
        print(f"  {k:<16} = {v!r}")
    diffs = config_differences(v6_dir, params)
    if diffs:
        print("WARNING: v6 and v7 config.yaml differ in settings that affect the episode:")
        for d in diffs:
            print(f"  {d}")
    print(f"Seeds       : {seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print(f"Captures    : start, {args.n_between} between, end - {args.lead_time:g} s")
    print(f"Output      : {out_root}")
    print("=" * 90)

    (out_root / "params_used.json").write_text(
        json.dumps({"source": source, "params": params, "config_differences_v6_v7": diffs}, indent=2),
        encoding="utf-8")

    infos, failures = [], []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(capture_seed, s, params, out_root, args.lead_time, args.n_between): s for s in seeds}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                info = fut.result()
            except Exception as exc:
                failures.append((s, str(exc)))
                print(f"seed {s:4d}: FAILED -- {str(exc).splitlines()[0]}")
                continue
            infos.append(info)
            flags = (" short" if info["short_episode"] else "") + ("" if info["deterministic"] else " NON-DETERMINISTIC")
            print(f"seed {s:4d}: {info['outcome']:9s} start={info['t_start']:7.1f}s end={info['t_end']:7.1f}s "
                  f"captures={len([c for c in info['captures'] if c['state_file']])}{flags}")

    infos.sort(key=lambda r: r["seed"])
    max_caps = args.n_between + 2
    with (out_root / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "outcome", "final_progress", "t_start", "t_end", "duration",
                    "short_episode", "deterministic"]
                   + [c for k in range(max_caps) for c in (f"cap{k}_label", f"cap{k}_time", f"cap{k}_state_file")])
        for r in infos:
            caps = [x for c in r["captures"] for x in (c["label"], c["time"], c["state_file"])]
            w.writerow([r["seed"], r["outcome"], r["final_progress"], r["t_start"], r["t_end"], r["duration"],
                        r["short_episode"], r["deterministic"]] + caps + [""] * (3 * max_caps - len(caps)))

    n_short = sum(r["short_episode"] for r in infos)
    n_nondet = sum(not r["deterministic"] for r in infos)
    print("=" * 90)
    print(f"Done: {len(infos)} seeds captured, {len(failures)} failed, {n_short} short episodes, "
          f"{n_nondet} non-deterministic")
    print(f"Index: {out_root / 'index.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
