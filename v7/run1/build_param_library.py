"""
Step 1 of the classification approach: a library of candidate DIPF parameter
sets, and the outcome of every candidate on every episode.

1. Candidates (written once to <out-dir>/candidates.json):
     - candidate 0  = the v6 best parameters (the one-set-for-all baseline)
     - candidates 1.. = medoids of k-means clusters of the per-episode best
       parameter sets (episode_opt_results/), clustered in log10 space scaled
       to the search bounds. A medoid is an actual optimised set (the member
       nearest its cluster centre) -- averages of good sets are not good sets.
2. Evaluation: every candidate is simulated on every episode (150 m segment,
   same as optimize_episodes.py; in-process, fresh SUMO per run). Episodes
   are deterministic, so this gives an exact success / cost matrix.
   One line per finished episode is appended to <out-dir>/eval.jsonl, so the
   run can be stopped and restarted (finished episodes are skipped).
3. <out-dir>/eval_matrix.csv: episode_id, ok_<k> (1 = success), C_<k> per candidate.

Usage:
    python build_param_library.py                       # 24 medoids + v6, all episodes
    python build_param_library.py --n-clusters 16
    python build_param_library.py --limit 20 --out-dir param_library_test   # quick test
    python build_param_library.py --matrix-only         # rebuild eval_matrix.csv from eval.jsonl
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
EPI = CONFIG["episode_opt"]
BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in EPI["bounds"].items()}
PARAMS = list(BOUNDS)


# =============================================================================
# Candidates
# =============================================================================

def scaled_log(P):
    lo = np.log10([BOUNDS[p][0] for p in PARAMS])
    hi = np.log10([BOUNDS[p][1] for p in PARAMS])
    return (np.log10(P) - lo) / (hi - lo)


def kmeans(Z, k, rng, iters=200):
    # k-means++ initialisation
    centres = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d2 = np.min(((Z[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
    C = np.array(centres)
    for _ in range(iters):
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, lab


def make_candidates(results_dir, n_clusters, seed, v6_params):
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(results_dir.glob("s*_c*.json"))]
    rows = [r for r in rows if r["best"]["result"]["status"] == "success"]
    P = np.array([[r["best"]["params"][p] for p in PARAMS] for r in rows])
    Z = scaled_log(P)
    C, lab = kmeans(Z, n_clusters, np.random.default_rng(seed))
    cands = [{"id": 0, "source": "v6 best parameters", "params": {p: float(v6_params[p]) for p in PARAMS}}]
    for j in range(n_clusters):
        members = np.where(lab == j)[0]
        if len(members) == 0:
            continue
        m = members[np.argmin(((Z[members] - C[j]) ** 2).sum(-1))]
        cands.append({"id": len(cands), "source": f"medoid of cluster {j} ({len(members)} episodes), "
                                                  f"from {rows[m]['episode_id']}",
                      "params": {p: float(P[m, i]) for i, p in enumerate(PARAMS)}})
    return cands, len(rows)


# =============================================================================
# Evaluation
# =============================================================================

def _worker_init():
    sys.argv = [sys.argv[0]]


def eval_episode(ep_id, cands, distance, max_time):
    import dipf_controller as dc                      # once per worker
    status, cost = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for c in cands:
            try:
                m = dc.run_episode(BASE_DIR / "episodes" / ep_id, c["params"], distance, max_time, output_dir=tmp)
                status.append(m["status"])
                cost.append(float(m["C"]) if m["C"] == m["C"] else math.inf)
            except Exception as exc:
                status.append("error")
                cost.append(math.inf)
    return {"episode_id": ep_id, "status": status, "C": cost}


def write_matrix(out_dir, n_cands):
    rows = {}
    for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if len(r["status"]) == n_cands and "error" not in r["status"]:
                rows[r["episode_id"]] = r
    with (out_dir / "eval_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_id"] + [f"ok_{k}" for k in range(n_cands)] + [f"C_{k}" for k in range(n_cands)])
        for ep in sorted(rows):
            r = rows[ep]
            w.writerow([ep] + [int(s == "success") for s in r["status"]] + r["C"])
    ok = np.array([[s == "success" for s in rows[e]["status"]] for e in sorted(rows)])
    print(f"eval_matrix.csv: {len(rows)} episodes x {n_cands} candidates")
    if len(ok):
        print(f"  success rate per candidate: " + " ".join(f"{k}:{ok[:, k].mean():.0%}" for k in range(n_cands)))
        print(f"  episodes solved by at least one candidate (oracle): {ok.any(1).mean():.1%}   "
              f"by v6 (candidate 0): {ok[:, 0].mean():.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(BASE_DIR / EPI.get("results_dir", "episode_opt_results")))
    ap.add_argument("--episodes-dir", default=str(BASE_DIR / EPI.get("episodes_dir", "episodes")))
    ap.add_argument("--out-dir", default=str(BASE_DIR / "param_library"))
    ap.add_argument("--n-clusters", type=int, default=24, help="medoid candidates (plus v6 = candidate 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (testing)")
    ap.add_argument("--n-jobs", type=int, default=int(EPI.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--matrix-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_file = out_dir / "candidates.json"

    from capture_states import load_v6_params
    v6, v6_source = load_v6_params((BASE_DIR / CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve())

    if cand_file.exists():
        cands = json.loads(cand_file.read_text(encoding="utf-8"))
        print(f"Using existing {cand_file} ({len(cands)} candidates; delete it to re-cluster)")
    else:
        cands, n_src = make_candidates(Path(args.results_dir), args.n_clusters, args.seed, v6)
        cand_file.write_text(json.dumps(cands, indent=2), encoding="utf-8")
        print(f"Clustered {n_src} successful per-episode best sets into {args.n_clusters} clusters -> "
              f"{len(cands)} candidates (incl. v6) -> {cand_file}")

    if args.matrix_only:
        write_matrix(out_dir, len(cands))
        return

    index = list(csv.DictReader((Path(args.episodes_dir) / "index.csv").open(encoding="utf-8")))
    eps = [r["episode_id"] for r in index if r["usable"] == "True"]
    if args.limit:
        eps = eps[:args.limit]
    done = set()
    if (out_dir / "eval.jsonl").exists():
        for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if len(r["status"]) == len(cands) and "error" not in r["status"]:
                    done.add(r["episode_id"])
    todo = [e for e in eps if e not in done]
    distance, max_time = float(EPI.get("distance_m", 150.0)), float(EPI.get("max_time_s", 60.0))

    print("=" * 90)
    print("Parameter library evaluation")
    print(f"Candidates : {len(cands)}  (0 = v6 from {v6_source})")
    print(f"Episodes   : {len(eps)} selected, {len(eps) - len(todo)} already done, {len(todo)} to run")
    print(f"Simulations: {len(todo) * len(cands):,}  ({distance:g} m, timeout {max_time:g} s, {args.n_jobs} parallel)")
    print("=" * 90, flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_worker_init) as ex, \
            (out_dir / "eval.jsonl").open("a", encoding="utf-8") as log:
        futs = {ex.submit(eval_episode, e, cands, distance, max_time): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} episodes | {el / 60:.1f} min | ETA {el / i * (len(todo) - i) / 60:.1f} min",
                      flush=True)

    write_matrix(out_dir, len(cands))


if __name__ == "__main__":
    main()
