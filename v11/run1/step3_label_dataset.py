"""
Step 3 -- dataset: run EVERY library set on EVERY dataset scenario (futures 1..M)
and store the success probabilities as the model's labels.

    dataset.csv   one row per scenario: sid, seed, start_x/y/vx, the 80 encoding
                  values (encoding.FEATURES), p_0 .. p_{K-1} (success prob of set k)

The labels are complete (no "best set only" labels): the model learns, for each
set separately, how likely it is to work here, and a scenario where several sets
work teaches it that all of them are fine.

Results that step 2 already has (library sets on pool / validation scenarios,
same futures) are reused. Every finished run batch is appended to labels.jsonl,
so an interrupted run continues where it stopped. Labels are keyed by the set's
parameter values: if the library changes, old labels are not reused.

Usage:
    python step3_label_dataset.py
    python step3_label_dataset.py --n-jobs 7
    python step3_label_dataset.py --seeds 0 999          # only part of pipeline.seeds.dataset
"""

import argparse
import hashlib
import json
import time

import numpy as np

from common import (BASE_DIR, FUTURES, PIPE, TAU, existing, load_ego, make_pool, prob, run_outcomes,
                    seed_range)
from encoding import FEATURES as ENC

LIBRARY = BASE_DIR / "library.json"
STATE = BASE_DIR / "library_state.json"
LABELS = BASE_DIR / "labels.jsonl"
DATA = BASE_DIR / "dataset.csv"
CHUNK = 150          # scenarios per batch (x futures runs)


def set_key(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def load_labels():
    done = {}
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[(d["sid"], d["key"])] = d["p"]
    return done


def append(rows):
    with LABELS.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def prefill(keys, done):
    """Copy step-2 results (same futures 1..M) for the library sets."""
    if not STATE.exists():
        return 0
    st = json.loads(STATE.read_text())
    rows = []
    for s in st["sets"]:
        key = set_key(s["params"])
        if key not in keys:
            continue
        if st["futures"] == FUTURES:
            rows += [{"sid": sid, "key": key, "p": p, "src": "step2"} for sid, p in s["train"].items()
                     if (sid, key) not in done]
        if st["val_futures"] >= FUTURES:
            rows += [{"sid": sid, "key": key, "p": prob(o[:FUTURES]), "src": "step2"} for sid, o in s["val"].items()
                     if (sid, key) not in done]
    for r in rows:
        done[(r["sid"], r["key"])] = r["p"]
    append(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs=2, default=None, help="override pipeline.seeds.dataset")
    args = ap.parse_args()

    lib = json.loads(LIBRARY.read_text())
    sets = [s["params"] for s in lib["sets"]]
    keys = [set_key(p) for p in sets]
    K = len(sets)
    sids = existing(seed_range(args.seeds or PIPE["seeds"]["dataset"]))
    done = load_labels()
    reused = prefill(set(keys), done)
    todo = {k: [s for s in sids if (s, keys[k]) not in done] for k in range(K)}
    n_runs = sum(len(v) for v in todo.values()) * FUTURES

    pool, n_jobs = make_pool(args.n_jobs)
    print("=" * 90)
    print(f"Step 3: label {len(sids)} dataset scenarios with all {K} library sets ({FUTURES} futures each)")
    print(f"Already labelled: {sum(len(sids) - len(v) for v in todo.values())} of {len(sids) * K} "
          f"(reused {reused} from step 2) -> {n_runs:,} runs to do, {n_jobs} parallel")
    print("=" * 90, flush=True)

    t0, finished = time.time(), 0
    try:
        for k in range(K):
            for i in range(0, len(todo[k]), CHUNK):
                chunk = todo[k][i:i + CHUNK]
                out = run_outcomes(pool, [sets[k]], chunk, list(range(1, FUTURES + 1)))[0]
                rows = [{"sid": sid, "key": keys[k], "p": prob(o), "outcomes": o} for sid, o in out.items()]
                append(rows)
                for r in rows:
                    done[(r["sid"], r["key"])] = r["p"]
                finished += len(chunk) * FUTURES
                el = time.time() - t0
                print(f"  set {k}: {min(i + CHUNK, len(todo[k]))}/{len(todo[k])} scenarios, mean p "
                      f"{np.mean([r['p'] for r in rows]):.3f}   total {finished:,}/{n_runs:,} runs  "
                      f"[{el:.0f}s, ETA {el / finished * (n_runs - finished) / 60:.0f} min]", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted -- finished batches are saved; rerun to continue.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    rows, missing = [], 0
    for sid in sids:
        if not all((sid, key) in done for key in keys):
            missing += 1
            continue
        ego = load_ego(sid)
        rows.append([sid, ego["seed"], ego["centre_x"], ego["centre_y"], ego["vx"]] + list(ego["encoding"])
                    + [done[(sid, key)] for key in keys])
    header = ["sid", "seed", "start_x", "start_y", "start_vx"] + ENC + [f"p_{k}" for k in range(K)]
    with DATA.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    Y = np.array([r[-K:] for r in rows])
    print("\n" + "=" * 90)
    print(f"dataset.csv: {len(rows)} scenarios x {K} sets" + (f" ({missing} incomplete, left out)" if missing else ""))
    print(f"{'set':>4s} {'mean success p':>15s} {'solves (p>=tau)':>16s} {'only this set':>14s}")
    solved = Y >= TAU
    for k in range(K):
        print(f"{k:4d} {Y[:, k].mean():15.3f} {solved[:, k].mean():16.1%} {np.sum(solved[:, k] & (solved.sum(1) == 1)):14d}")
    print(f"Best single set: {Y.mean(0).max():.3f} (set {int(Y.mean(0).argmax())})   "
          f"oracle choice: {Y.max(1).mean():.3f}   some set solves: {solved.any(1).mean():.1%}   "
          f"several sets solve: {(solved.sum(1) > 1).mean():.1%}")
    print(f"Saved {DATA}")


if __name__ == "__main__":
    main()
