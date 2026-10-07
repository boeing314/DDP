"""
Step 2 of the classification approach: an MLP that picks which parameter set
(from a small library) to use in the current traffic situation.

Inputs  : the same 28 features as the regression model (ego state + 6 nearest
          vehicles, skewed distance) from episode_param_dataset_ids.csv
Library : chosen from build_param_library.py's candidates using TRAINING
          episodes only -- v6 first, then greedily the candidate that solves
          the most training episodes not yet solved, up to --k sets
Labels  : for each episode and library set, 1 if that set reaches the 150 m
          target (success), else 0 -- read from param_library/eval_matrix.csv.
          Many sets can be correct for one episode; all of them are labelled 1.
Model   : MLP 28 -> hidden -> K logits, sigmoid outputs = predicted probability
          that each set succeeds here (multi-label, binary cross-entropy);
          the pick is the set with the highest probability.
Split   : by seed, same procedure/seed as train_param_mlp.py (same test
          episodes when the matrix covers the whole dataset).

Because every set's actual outcome on every episode is in the matrix, the
classifier's test success is looked up exactly -- no extra simulation:
    picked set succeeds?   compared with
    v6 for every episode / best single set (on train) / oracle (any library set)

Output files (prefix --out, default param_classifier):
    param_classifier.npz           weights, normalisation, library parameter sets
    param_classifier_metrics.json  library choice, coverage, test results
    param_classifier_test_predictions.csv  per test episode: pick, probabilities, outcome

Usage:
    python train_param_classifier.py
    python train_param_classifier.py --k 6 --hidden 32 32
    python train_param_classifier.py --library-dir param_library_test     # quick test
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_param_mlp import FEATURES, MLP, train

BASE_DIR = Path(__file__).resolve().parent


def seed_split(groups, fractions, seed):
    """Identical procedure to train_param_mlp.py (first draw of a fresh rng)."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    n_tr = int(round(fractions[0] * len(uniq)))
    n_va = int(round(fractions[1] * len(uniq)))
    g_tr, g_va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
    return np.array(["train" if g in g_tr else "val" if g in g_va else "test" for g in groups])


def greedy_library(OK_train, k, start=(0,)):
    """v6 (candidate 0) first, then repeatedly the candidate covering most unsolved training episodes."""
    chosen = list(start)
    covered = OK_train[:, chosen].any(1)
    while len(chosen) < k:
        gains = [(OK_train[~covered, j].sum(), -j) if j not in chosen else (-1, 0) for j in range(OK_train.shape[1])]
        best = int(np.argmax([g[0] for g in gains]))
        if gains[best][0] <= 0:
            break
        chosen.append(best)
        covered |= OK_train[:, best]
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-dir", default=str(BASE_DIR / "param_library"))
    ap.add_argument("--data", default=str(BASE_DIR / "episode_param_dataset_ids.csv"))
    ap.add_argument("--out", default=str(BASE_DIR / "param_classifier"))
    ap.add_argument("--k", type=int, default=8, help="number of parameter sets in the final library (incl. v6)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lib_dir = Path(args.library_dir)
    cands = json.loads((lib_dir / "candidates.json").read_text(encoding="utf-8"))
    mat = pd.read_csv(lib_dir / "eval_matrix.csv")
    feats = pd.read_csv(args.data)
    df = feats.merge(mat, on="episode_id", how="inner").reset_index(drop=True)
    n_c = len(cands)
    OK_all = df[[f"ok_{j}" for j in range(n_c)]].to_numpy(bool)
    C_all = df[[f"C_{j}" for j in range(n_c)]].to_numpy(float)

    groups = df["episode_id"].str.slice(1, 5).astype(int).to_numpy()
    split = seed_split(groups, args.split, args.seed)
    tr, va, te = split == "train", split == "val", split == "test"

    # ---- library from training episodes only
    lib = greedy_library(OK_all[tr], args.k)
    K = len(lib)
    OK, Cm = OK_all[:, lib], C_all[:, lib]

    print("=" * 90)
    print("Parameter-set classifier")
    print(f"Data      : {len(df)} episodes with features + library outcomes ({len(np.unique(groups))} seeds)")
    print(f"Split     : train {tr.sum()} / val {va.sum()} / test {te.sum()} (by seed)")
    print(f"Candidates: {n_c} -> library of {K} (greedy coverage on training episodes)")
    print(f"  {'#':>2s} {'cand':>4s}  {'success alone (train)':>21s}  {'library coverage train/val/test':>32s}  source")
    for i, j in enumerate(lib):
        cov = [OK[m][:, :i + 1].any(1).mean() for m in (tr, va, te)]
        print(f"  {i:2d} {j:4d}  {OK_all[tr, j].mean():21.1%}  {cov[0]:10.1%} {cov[1]:10.1%} {cov[2]:10.1%}  "
              f"{cands[j]['source']}")
    print("=" * 90, flush=True)

    # ---- model
    rng = np.random.default_rng(args.seed + 1)
    X = df[FEATURES].to_numpy(float)
    Y = OK.astype(float)
    norm = {"x_mean": X[tr].mean(0), "x_std": X[tr].std(0) + 1e-9}
    Xn = (X - norm["x_mean"]) / norm["x_std"]
    sizes = [len(FEATURES)] + args.hidden + [K]
    model = MLP(sizes, rng, loss="bce")
    print(f"Network   : {' -> '.join(map(str, sizes))}, sigmoid outputs, BCE, Adam lr={args.lr}, L2 {args.weight_decay}")
    t0 = time.time()
    _, best_epoch = train(model, Xn[tr], Y[tr], Xn[va], Y[va], args.epochs, args.lr, args.batch,
                          args.weight_decay, args.patience, rng, log_every=200)
    print(f"Training done in {time.time() - t0:.0f}s, best epoch {best_epoch}")

    # ---- evaluation by lookup
    prob = 1.0 / (1.0 + np.exp(-model.predict(Xn)))
    pick = prob.argmax(1)
    best_single = int(np.argmax(OK[tr].mean(0)))

    def evaluate(mask):
        rows = np.where(mask)[0]
        succ_pick = OK[rows, pick[rows]]
        C_pick = Cm[rows, pick[rows]]
        return {
            "classifier": {"success": float(succ_pick.mean()), "mean_C": float(np.mean(C_pick[np.isfinite(C_pick)]))},
            "v6_only": {"success": float(OK[rows, 0].mean()), "mean_C": float(np.mean(Cm[rows, 0]))},
            "best_single_set": {"set": best_single, "success": float(OK[rows, best_single].mean()),
                                "mean_C": float(np.mean(Cm[rows, best_single]))},
            "random_set": {"success": float(OK[rows].mean())},
            "oracle_library": {"success": float(OK[rows].any(1).mean())},
        }

    res = {name: evaluate(m) for name, m in (("train", tr), ("val", va), ("test", te))}
    print(f"\n{'success rate':28s} {'train':>8s} {'val':>8s} {'test':>8s}")
    for key, label in (("classifier", "classifier pick"), ("v6_only", "v6 for every episode"),
                       ("best_single_set", f"best single set (#{lib[best_single]})"),
                       ("random_set", "random library set"), ("oracle_library", "oracle (any library set)")):
        print(f"{label:28s} " + " ".join(f"{res[s][key]['success']:8.1%}" for s in ("train", "val", "test")))
    hit = OK[te].any(1)
    print(f"\nTest: classifier mean C {res['test']['classifier']['mean_C']:.3f} vs v6 {res['test']['v6_only']['mean_C']:.3f}; "
          f"on test episodes where some library set works ({hit.sum()}), the pick works "
          f"{OK[np.where(te)[0][hit], pick[np.where(te)[0][hit]]].mean():.1%}")

    # ---- save
    library = [{"candidate": int(j), "source": cands[j]["source"], "params": cands[j]["params"]} for j in lib]
    meta = {"kind": "classifier", "sizes": sizes, "features": FEATURES, "loss": "bce",
            "targets": [f"set_{i}" for i in range(K)], "library": library}
    arrays = {f"W{i}": W for i, W in enumerate(model.W)} | {f"b{i}": b for i, b in enumerate(model.b)}
    np.savez(f"{args.out}.npz", **arrays, x_mean=norm["x_mean"], x_std=norm["x_std"],
             y_mean=np.zeros(K), y_std=np.ones(K), meta=json.dumps(meta))
    out = df.loc[te, ["episode_id"]].copy()
    out["pick"] = pick[te]
    out["pick_success"] = OK[te, pick[te]].astype(int)
    out["v6_success"] = OK[te, 0].astype(int)
    out["any_success"] = OK[te].any(1).astype(int)
    for i in range(K):
        out[f"p_set{i}"] = prob[te, i]
    out.to_csv(f"{args.out}_test_predictions.csv", index=False)
    Path(f"{args.out}_metrics.json").write_text(json.dumps(
        {"library": library, "results": res, "best_epoch": best_epoch, "split_rows":
         {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "args": vars(args)}, indent=2),
        encoding="utf-8")
    print(f"\nSaved: {args.out}.npz, {args.out}_metrics.json, {args.out}_test_predictions.csv")


if __name__ == "__main__":
    main()
