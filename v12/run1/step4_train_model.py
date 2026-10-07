"""
Step 4 -- model: traffic encoding at the takeover -> success probability of EVERY
library set (multi-label: K independent sigmoid outputs, they need not sum to 1).

Input : the 80-value encoding (encoding.FEATURES), preprocessed (common.preprocess:
        gaps -> closeness, drac artefact removed) and standardised.
Labels: p_k = success probability of set k over the futures (soft targets, binary
        cross-entropy per output).
Split : by seed. Scenarios used to BUILD the library (pipeline.seeds.library_train)
        only go to the training split -- the sets were tuned on them, so they would
        flatter every policy. Validation (early stopping) and test come from the rest.
Extra : left/right mirrored copies of the training scenarios (the controller is
        symmetric, so a mirrored scenario has the same labels); an ensemble of
        independently initialised networks, outputs averaged.

Evaluation on the test split, as expected segment success when the set is picked by
    set 0 (round 0 = best single set on the initial pool)
    best single set (highest mean p on the training split)
    the model (argmax of the predicted probabilities)
    the model with fallback: switch away from the best single set only if the
        predicted advantage exceeds a margin chosen on the validation split
        (a margin of 1 = always the best single set, so an unreliable model
        cannot do worse than the fixed set on validation)
    random set / oracle (best set in hindsight; an upper bound, slightly optimistic)
plus per-set AUC / Brier score and a calibration table.

Outputs:
    model.npz            weights, input scaling, meta (library, names, best single set)
    model_report.json    metrics

Usage:
    python step4_train_model.py
    python step4_train_model.py --no-mirror --ensemble 3
"""

import argparse
import json
import math

import numpy as np
import pandas as pd

from common import BASE_DIR, MODEL_FILE, PIPE, TAU, mirror, preprocess, seed_range, train_mlp
from encoding import FEATURES as ENC

MOD = PIPE["model"]
DATA = BASE_DIR / "dataset.csv"
LIBRARY = BASE_DIR / "library.json"
REPORT = BASE_DIR / "model_report.json"


def auc(score, label):
    """Area under the ROC curve (rank formula); NaN if only one class."""
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bce(P, Y):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return float(np.mean(-(Y * np.log(P) + (1 - Y) * np.log(1 - P))))


MARGINS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def pick(P, kbest, margin):
    """argmax of the predicted probabilities, but the best single set unless the winner beats it by > margin."""
    c = P.argmax(1)
    return np.where(P[np.arange(len(P)), c] - P[:, kbest] > margin, c, kbest)


def policies(P, Y, kbest, margin):
    """Expected success (mean p of the picked set) and solve rate of each selection rule."""
    choice = pick(P, kbest, margin)
    rows = {"set 0": np.zeros(len(Y), int), f"best single (set {kbest})": np.full(len(Y), kbest),
            "model (argmax)": P.argmax(1), f"model + fallback (m={margin:g})": choice}
    out = {name: {"success": float(Y[np.arange(len(Y)), c].mean()),
                  "solved": float((Y[np.arange(len(Y)), c] >= TAU).mean())} for name, c in rows.items()}
    out["random set"] = {"success": float(Y.mean()), "solved": float((Y >= TAU).mean())}
    out["oracle"] = {"success": float(Y.max(1).mean()), "solved": float((Y.max(1) >= TAU).mean())}
    d = Y[np.arange(len(Y)), choice] - Y[:, kbest]
    paired = {"mean_diff_vs_best_single": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
              "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "same": int((d == 0).sum())}
    return out, paired, np.bincount(choice, minlength=Y.shape[1])


def show(name, P, Y, kbest, prior, margin):
    out, paired, counts = policies(P, Y, kbest, margin)
    print(f"\n--- {name}: {len(Y)} scenarios ---")
    print(f"{'selection':<26s} {'mean success':>13s} {'solved':>8s}")
    for k, v in out.items():
        print(f"{k:<26s} {v['success']:13.3f} {v['solved']:8.1%}")
    print(f"model + fallback - best single: {paired['mean_diff_vs_best_single']:+.3f} +- {paired['se']:.3f} (SE)   "
          f"better on {paired['better']}, worse on {paired['worse']}, same on {paired['same']}")
    print("model + fallback picks: " + "  ".join(f"set {k}: {c}" for k, c in enumerate(counts)))
    print(f"{'set':>4s} {'AUC':>6s} {'Brier':>7s} {'BCE':>7s} {'BCE prior':>10s}")
    per = []
    for k in range(Y.shape[1]):
        a = auc(P[:, k], Y[:, k] >= TAU)
        b = float(np.mean((P[:, k] - Y[:, k]) ** 2))
        per.append({"auc": a, "brier": b, "bce": bce(P[:, k], Y[:, k]), "bce_prior": bce(np.full(len(Y), prior[k]), Y[:, k])})
        print(f"{k:4d} {a:6.3f} {b:7.3f} {per[-1]['bce']:7.3f} {per[-1]['bce_prior']:10.3f}")
    print("calibration (all sets): predicted -> observed")
    edges = np.linspace(0, 1, 6)
    cal = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & ((P < hi) if hi < 1 else (P <= hi))
        if m.any():
            cal.append({"bin": [lo, hi], "n": int(m.sum()), "pred": float(P[m].mean()), "obs": float(Y[m].mean())})
            print(f"  [{lo:.1f},{hi:.1f}]  n={m.sum():5d}  predicted {P[m].mean():.3f}  observed {Y[m].mean():.3f}")
    return {"policies": out, "paired": paired, "picks": counts.tolist(), "per_set": per, "calibration": cal}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", type=int, default=int(MOD["ensemble"]))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    use_mirror = bool(MOD.get("mirror", True)) and not args.no_mirror

    lib = json.loads(LIBRARY.read_text())
    df = pd.read_csv(DATA)
    K = len(lib["sets"])
    pcols = [f"p_{k}" for k in range(K)]
    if any(c not in df for c in pcols):
        raise SystemExit("dataset.csv does not match library.json -- rerun step 3")
    X, Y, seeds = df[ENC].to_numpy(float), df[pcols].to_numpy(float), df["seed"].to_numpy(int)

    rng = np.random.default_rng(args.seed)
    lib_train = set(seed_range(PIPE["seeds"]["library_train"]))
    others = np.array(sorted(set(seeds) - lib_train))
    rng.shuffle(others)
    n_val, n_test = round(MOD["split"][1] * len(df)), round(MOD["split"][2] * len(df))
    if n_val + n_test > len(others) or min(n_val, n_test) < 10:
        n_val = n_test = len(others) // 3
    val_s, test_s = set(others[:n_val]), set(others[n_val:n_val + n_test])
    sp = np.array(["val" if s in val_s else "test" if s in test_s else "train" for s in seeds])
    tr, va, te = sp == "train", sp == "val", sp == "test"

    Xtr_raw, Ytr = X[tr], Y[tr]
    if use_mirror:
        Xtr_raw, Ytr = np.vstack([Xtr_raw, mirror(Xtr_raw)]), np.vstack([Ytr, Ytr])
    Xtr = preprocess(Xtr_raw)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    norm = lambda A: (preprocess(A) - mu) / sd                       # noqa: E731
    Xtr, Xva, Xte = (Xtr - mu) / sd, norm(X[va]), norm(X[te])

    print("=" * 90)
    print(f"Step 4: multi-label model, {K} library sets, {len(df)} scenarios")
    print(f"Split by seed: train {tr.sum()} (incl. {np.isin(seeds, list(lib_train)).sum()} library-building scenarios)"
          f"{' + mirrored copies' if use_mirror else ''}, val {va.sum()}, test {te.sum()}")
    print(f"MLP {ENC.__len__()} -> {MOD['hidden']} -> {K} (sigmoid), Adam lr {MOD['lr']}, weight decay "
          f"{MOD['weight_decay']}, <= {MOD['epochs']} epochs (patience {MOD['patience']}), ensemble {args.ensemble}")
    print("=" * 90, flush=True)

    members = []
    for m in range(args.ensemble):
        net, ep, v = train_mlp(Xtr, Ytr, Xva, Y[va], MOD["hidden"], int(MOD["epochs"]), float(MOD["lr"]),
                               float(MOD["weight_decay"]), int(MOD["patience"]), np.random.default_rng(args.seed + 1 + m))
        members.append(net)
        print(f"  member {m}: best epoch {ep}, val loss {v:.4f}", flush=True)
    predict = lambda A: np.mean([n.proba(A) for n in members], axis=0)  # noqa: E731

    prior = Y[tr].mean(0)
    kbest = int(prior.argmax())
    Pva = predict(Xva)
    gains = [float(Y[va][np.arange(va.sum()), pick(Pva, kbest, m)].mean()) for m in MARGINS]
    margin = max(MARGINS, key=lambda m: (round(gains[MARGINS.index(m)], 9), m))   # ties -> more conservative
    print("\nFallback margin on validation: " + "  ".join(f"{m:g}: {g:.3f}" for m, g in zip(MARGINS, gains))
          + f"  -> {margin:g}")
    rep = {"margin": margin, "margin_scan": dict(zip(map(str, MARGINS), gains)),
           "val": show("VALIDATION", Pva, Y[va], kbest, prior, margin),
           "test": show("TEST", predict(Xte), Y[te], kbest, prior, margin)}
    print(f"\nTraining split (for overfitting check): model {Y[tr][np.arange(tr.sum()), predict(norm(X[tr])).argmax(1)].mean():.3f}"
          f" vs best single {Y[tr][:, kbest].mean():.3f}")

    meta = {"ensemble": len(members), "n_layers": len(members[0].W), "features": ENC, "K": K,
            "sets": [s["params"] for s in lib["sets"]], "best_single": kbest, "margin": margin,
            "prior": prior.tolist(),
            "tau": TAU, "mirror": use_mirror, "hidden": MOD["hidden"],
            "split": {"val": sorted(int(s) for s in val_s), "test": sorted(int(s) for s in test_s)}}
    arrays = {"mu": mu, "sd": sd, "meta": np.array(json.dumps(meta))}
    for m, net in enumerate(members):
        for i, (W, b) in enumerate(zip(net.W, net.b)):
            arrays[f"m{m}_W{i}"], arrays[f"m{m}_b{i}"] = W, b
    np.savez(MODEL_FILE, **arrays)
    REPORT.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved {MODEL_FILE} and {REPORT}")


if __name__ == "__main__":
    main()
