"""
Step 2 -- split the episodes by seed and cluster the TRAINING situations.

  - train / val / test split BY SEED (all captures of one seed stay together)
  - features standardised with training statistics
  - k-means (k-means++ init, n_init restarts, best inertia) on training episodes only;
    val / test episodes are assigned to the nearest centre
  - each cluster gets a readable description (lane, leader distance / closing,
    neighbouring lanes, density) to make "slow leader close ahead", "boxed in",
    "open road" ... visible

Outputs:
    clusters.npz            centres (standardised), feature mean / std, feature names
    cluster_assignment.csv  episode_id, seed, split, cluster, distance to centre
    clusters_summary.csv    per cluster: sizes, mean raw features, description

Usage:
    python cluster_situations.py                  # K from config.yaml (situations.n_clusters)
    python cluster_situations.py --k 12
    python cluster_situations.py --scan 4 30      # inertia / silhouette for K = 4..30, then exit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from situation import FEATURES

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
SIT = CONFIG["situations"]


def seed_split(seeds, fractions, split_seed):
    rng = np.random.default_rng(split_seed)
    uniq = rng.permutation(np.unique(seeds))
    n_tr = int(round(fractions[0] * len(uniq)))
    n_va = int(round(fractions[1] * len(uniq)))
    tr, va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
    return np.array(["train" if s in tr else "val" if s in va else "test" for s in seeds])


def kmeans(Z, k, rng, n_init=10, iters=300):
    best = None
    for _ in range(n_init):
        C = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        inertia = float(((Z - C[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    return best


def silhouette(Z, lab):
    D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
    s = np.zeros(len(Z))
    for i in range(len(Z)):
        same = lab == lab[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def describe(m, slot_range):
    """Readable label from a cluster's mean raw features."""
    lane = ("left lane" if m["left_exists"] < 0.5 else "right lane" if m["right_exists"] < 0.5 else "middle lane")
    g, dv, ttc = m["gap_own_ahead"], m["dv_own_ahead"], m["ttc_ahead"]
    if g > 0.8 * slot_range:
        lead = "open road ahead"
    else:
        dist = "close" if g < 15 else "medium" if g < 40 else "far"
        speed = "slower" if dv < -1.5 else "faster" if dv > 1.5 else "same-speed"
        lead = f"{speed} leader {dist} ahead ({g:.0f} m" + (f", TTC {ttc:.0f} s)" if ttc < 0.9 * SIT["ttc_cap_s"] else ")")

    def side(name):
        if m[f"{name}_exists"] < 0.5:
            return None
        ga, gb = m[f"gap_{name}_ahead"], m[f"gap_{name}_behind"]
        return f"{name} {'blocked' if min(ga, gb) < 10 else 'tight' if min(ga, gb) < 25 else 'free'}"

    sides = ", ".join(x for x in (side("left"), side("right")) if x)
    follower = "" if m["gap_own_behind"] > 25 else f", follower close behind ({m['gap_own_behind']:.0f} m)"
    dens = "dense" if m["density"] >= 12 else "light" if m["density"] <= 5 else "moderate"
    return f"{lane}; {lead}; {sides}{follower}; {dens} traffic ({m['density']:.0f} veh within 50 m)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(BASE_DIR / "situations.csv"))
    ap.add_argument("--k", type=int, default=int(SIT.get("n_clusters", 16)))
    ap.add_argument("--scan", type=int, nargs=2, metavar=("KMIN", "KMAX"), default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    split = seed_split(df["seed"].to_numpy(), SIT["split"], int(SIT.get("split_seed", 0)))
    df["split"] = split
    X = df[FEATURES].to_numpy(float)
    tr = split == "train"
    mean, std = X[tr].mean(0), X[tr].std(0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    rng = np.random.default_rng(int(SIT.get("kmeans_seed", 0)))

    print("=" * 100)
    print("Step 2: situation clusters")
    print(f"Episodes : {len(df)} | split by seed: train {tr.sum()} / val {(split == 'val').sum()} / "
          f"test {(split == 'test').sum()}")
    print("=" * 100, flush=True)

    if args.scan:
        print(f"{'K':>3s} {'inertia/ep (train)':>19s} {'inertia/ep (val)':>17s} {'silhouette (train)':>19s} {'smallest cluster':>17s}")
        va = split == "val"
        for k in range(args.scan[0], args.scan[1] + 1):
            C, lab, inertia = kmeans(Z[tr], k, rng, n_init=5)
            dva = ((Z[va][:, None, :] - C[None]) ** 2).sum(-1).min(1).sum()
            print(f"{k:3d} {inertia / tr.sum():19.3f} {dva / va.sum():17.3f} {silhouette(Z[tr], lab):19.3f} "
                  f"{np.bincount(lab, minlength=k).min():17d}", flush=True)
        print("Pick K where the val inertia stops dropping much and clusters keep enough training episodes "
              f"(cluster_opt.min_train_episodes = {CONFIG['cluster_opt']['min_train_episodes']}).")
        return

    C, lab_tr, inertia = kmeans(Z[tr], args.k, rng)
    d_all = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
    lab = d_all.argmin(1)
    df["cluster"] = lab
    df["dist"] = np.sqrt(d_all.min(1))

    np.savez(BASE_DIR / "clusters.npz", centres=C, mean=mean, std=std, features=np.array(FEATURES))
    df[["episode_id", "seed", "split", "cluster", "dist"]].to_csv(BASE_DIR / "cluster_assignment.csv", index=False)

    rows = []
    for j in range(args.k):
        m = df.loc[(df.cluster == j) & tr, FEATURES].mean()
        rows.append({"cluster": j, "n_train": int(((df.cluster == j) & tr).sum()),
                     "n_val": int(((df.cluster == j) & (df.split == "val")).sum()),
                     "n_test": int(((df.cluster == j) & (df.split == "test")).sum()),
                     "description": describe(m, float(SIT["slot_range_m"])), **m.round(3).to_dict()})
    summ = pd.DataFrame(rows)
    summ.to_csv(BASE_DIR / "clusters_summary.csv", index=False)

    print(f"K = {args.k}, inertia/episode (train) {inertia / tr.sum():.3f}\n")
    print(f"{'cl':>3s} {'train':>5s} {'val':>4s} {'test':>4s}  description")
    for r in rows:
        print(f"{r['cluster']:3d} {r['n_train']:5d} {r['n_val']:4d} {r['n_test']:4d}  {r['description']}")
    small = [r["cluster"] for r in rows if r["n_train"] < CONFIG["cluster_opt"]["min_train_episodes"]]
    if small:
        print(f"\nClusters with < {CONFIG['cluster_opt']['min_train_episodes']} training episodes "
              f"(will fall back to v6): {small}")
    print(f"\nSaved: clusters.npz, cluster_assignment.csv, clusters_summary.csv")


if __name__ == "__main__":
    main()
