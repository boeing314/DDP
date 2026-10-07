"""
Small MLP that predicts the DIPF parameters from the traffic situation.

Input  (28): x_ego, y_ego, vx_ego, vy_ego and x/y/vx/vy relative to the ego of
             the 6 nearest vehicles (skewed distance), as written by
             make_param_dataset.py
Output  (8): lambda_obs, alpha, tau_x_base, tau_y_base, sx, delta_lane,
             obs_radius_ahead, y_margin -- the episode's best parameters from
             the per-episode differential evolution

Design
  - Targets are learned as log10(parameter) (they span several orders of
    magnitude), standardised; inputs are standardised with training-set
    statistics. Predictions are mapped back with 10**y and clipped to the
    search bounds (config.yaml -> episode_opt.bounds).
  - Train / validation / test split is by SEED (default 70/15/15): episodes
    from the same seed share traffic, so a row-wise split would leak.
  - Network: 28 -> 64 -> 64 -> 8, ReLU, plain NumPy (no torch / sklearn
    needed), Adam, L2 weight decay, mini-batches, early stopping on
    validation loss.
  - Reported: R^2 and RMSE (log10 units) per parameter on each split, next
    to the "predict the training mean" baseline (R^2 = 0).
  - --sim-eval: closed-loop test on the test-set episodes -- simulate each one
    (150 m segment, same as optimize_episodes.py) with (a) the MLP's
    predicted parameters, (b) the v6 parameters, and compare with (c) the
    episode's own optimum. This is the metric that matters: many different
    parameter sets solve an episode, so a low R^2 does not by itself mean
    the predictions are useless.

Output files (prefix --out, default param_mlp):
    param_mlp.npz            weights + normalisation + names + bounds
    param_mlp_metrics.json   split sizes, per-parameter R^2/RMSE, sim-eval results
    param_mlp_training.png   loss curves and predicted-vs-actual per parameter
    param_mlp_test_predictions.csv   test episodes: actual and predicted parameters

Usage:
    python train_param_mlp.py                          # train + report
    python train_param_mlp.py --sim-eval               # + closed-loop test on test episodes
    python train_param_mlp.py --hidden 128 128 --epochs 3000 --lr 5e-4
    python train_param_mlp.py --predict some.csv --pred-out predicted.csv   # use a trained model
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in CONFIG["episode_opt"]["bounds"].items()}

TARGETS = ["lambda_obs", "alpha", "tau_x_base", "tau_y_base", "sx", "delta_lane", "obs_radius_ahead", "y_margin"]
FEATURES = (["x_ego", "y_ego", "vx_ego", "vy_ego"]
            + [f"{q}_veh{n}" for n in range(1, 7) for q in ("x_rel", "y_rel", "vx_rel", "vy_rel")])


# =============================================================================
# NumPy MLP
# =============================================================================

class MLP:
    """loss='mse' (regression, linear outputs) or 'bce' (multi-label classification: outputs are logits)."""

    def __init__(self, sizes, rng, loss="mse"):
        self.sizes = list(sizes)
        self.loss = loss
        self.W = [rng.normal(0.0, math.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]   # He init
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z      # ReLU hidden, linear output
            acts.append(h)
        return h, acts

    def predict(self, X):
        return self.forward(X)[0]

    def loss_value(self, X, Y):
        out = self.predict(X)
        if self.loss == "bce":
            # mean binary cross-entropy on logits (numerically stable form)
            return float(np.mean(np.maximum(out, 0) - out * Y + np.log1p(np.exp(-np.abs(out)))))
        return float(np.mean((out - Y) ** 2))

    def gradients(self, X, Y):
        out, acts = self.forward(X)
        n = len(X)
        if self.loss == "bce":
            delta = (1.0 / (1.0 + np.exp(-out)) - Y) / (n * Y.shape[1])   # d(BCE)/d(logit)
        else:
            delta = 2.0 * (out - Y) / (n * Y.shape[1])                  # d(MSE)/d(out)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW, gb, float(np.mean((out - Y) ** 2))

    def params(self):
        return self.W + self.b

    def copy_weights(self):
        return [p.copy() for p in self.params()]

    def set_weights(self, ws):
        k = len(self.W)
        self.W, self.b = [w.copy() for w in ws[:k]], [w.copy() for w in ws[k:]]


def train(model, Xtr, Ytr, Xva, Yva, epochs, lr, batch, weight_decay, patience, rng, log_every):
    ps = model.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    best_val, best_w, best_epoch, wait = math.inf, model.copy_weights(), 0, 0
    hist = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            gW, gb, _ = model.gradients(Xtr[idx], Ytr[idx])
            grads = [g + weight_decay * W for g, W in zip(gW, model.W)] + gb   # L2 on weights only
            step += 1
            params = model.params()
            for k, (p, g) in enumerate(zip(params, grads)):
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                p -= lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
        tr = model.loss_value(Xtr, Ytr)
        va = model.loss_value(Xva, Yva)
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best_val - 1e-6:
            best_val, best_w, best_epoch, wait = va, model.copy_weights(), epoch, 0
        else:
            wait += 1
        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:5d}  train loss {tr:.4f}  val loss {va:.4f}  (best val {best_val:.4f} @ {best_epoch})",
                  flush=True)
        if wait >= patience:
            print(f"  early stop at epoch {epoch}: no val improvement for {patience} epochs", flush=True)
            break
    model.set_weights(best_w)
    return hist, best_epoch


# =============================================================================
# Saving / loading / predicting
# =============================================================================

def save_model(path, model, norm):
    arrays = {f"W{i}": W for i, W in enumerate(model.W)} | {f"b{i}": b for i, b in enumerate(model.b)}
    meta = {"sizes": model.sizes, "features": FEATURES, "targets": TARGETS, "bounds": BOUNDS,
            "target_transform": "log10, standardised"}
    np.savez(path, **arrays, x_mean=norm["x_mean"], x_std=norm["x_std"], y_mean=norm["y_mean"],
             y_std=norm["y_std"], meta=json.dumps(meta))


def load_model(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    model = MLP(meta["sizes"], np.random.default_rng(0), loss=meta.get("loss", "mse"))
    k = len(meta["sizes"]) - 1
    model.W = [z[f"W{i}"] for i in range(k)]
    model.b = [z[f"b{i}"] for i in range(k)]
    norm = {n: z[n] for n in ("x_mean", "x_std", "y_mean", "y_std")}
    return model, norm, meta


def predict_params(model, norm, df):
    """DataFrame with the 28 feature columns -> DataFrame of the 8 parameters (clipped to bounds)."""
    X = (df[FEATURES].to_numpy(float) - norm["x_mean"]) / norm["x_std"]
    logp = model.predict(X) * norm["y_std"] + norm["y_mean"]
    out = pd.DataFrame(10.0 ** logp, columns=TARGETS, index=df.index)
    for p, (lo, hi) in BOUNDS.items():
        out[p] = out[p].clip(lo, hi)
    return out


# =============================================================================
# Metrics
# =============================================================================

def r2_rmse(Y_true, Y_pred):
    res = {}
    for j, p in enumerate(TARGETS):
        t, y = Y_true[:, j], Y_pred[:, j]
        ss_res = float(np.sum((t - y) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        res[p] = {"r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                  "rmse_log10": math.sqrt(ss_res / len(t))}
    return res


# =============================================================================
# Closed-loop evaluation
# =============================================================================

def _sim_one(args):
    ep_id, params, distance, max_time = args
    import tempfile
    import dipf_controller as dc
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_episode(BASE_DIR / "episodes" / ep_id, params, distance, max_time, output_dir=tmp)
            return ep_id, m["status"], float(m["C"]), float(m["final_progress"])
        except Exception as exc:
            return ep_id, "error", math.nan, math.nan


def sim_eval(test_df, pred, v6_params, n_jobs):
    ep_cfg = CONFIG["episode_opt"]
    dist, tmax = float(ep_cfg.get("distance_m", 150.0)), float(ep_cfg.get("max_time_s", 60.0))
    jobs = []
    for i, ep_id in enumerate(test_df["episode_id"]):
        jobs.append((ep_id, {p: float(pred.iloc[i][p]) for p in TARGETS}, dist, tmax))
        jobs.append((ep_id, dict(v6_params), dist, tmax))
    print(f"\nClosed-loop evaluation: {len(test_df)} test episodes x 2 parameter sets ({len(jobs)} simulations)...",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        res = list(ex.map(_sim_one, jobs, chunksize=4))
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    mlp, v6 = res[0::2], res[1::2]

    def summary(rs):
        st = [r[1] for r in rs]
        C = [r[2] for r in rs if math.isfinite(r[2])]
        return {"success_rate": st.count("success") / len(st), "mean_C": float(np.mean(C)),
                "mean_progress": float(np.nanmean([r[3] for r in rs])), "errors": st.count("error")}

    opt = {"success_rate": float((test_df["best_status"] == "success").mean()),
           "mean_C": float(test_df["best_C"].mean())}
    out = {"mlp_predicted": summary(mlp), "v6_params": summary(v6), "per_episode_optimum": opt,
           "per_episode": [{"episode_id": a[0], "mlp_status": a[1], "mlp_C": a[2], "v6_status": b[1], "v6_C": b[2]}
                           for a, b in zip(mlp, v6)]}
    print(f"  {'parameters':28s} {'success':>8s} {'mean C':>8s}")
    print(f"  {'MLP prediction':28s} {out['mlp_predicted']['success_rate']:8.1%} {out['mlp_predicted']['mean_C']:8.3f}")
    print(f"  {'v6 best (one set for all)':28s} {out['v6_params']['success_rate']:8.1%} {out['v6_params']['mean_C']:8.3f}")
    print(f"  {'per-episode DE optimum':28s} {opt['success_rate']:8.1%} {opt['mean_C']:8.3f}")
    return out


# =============================================================================
# Plot
# =============================================================================

def plot(path, hist, Y_test, P_test, best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    ax = axes[0, 0]
    ax.plot(hist["train"], label="train")
    ax.plot(hist["val"], label="validation")
    ax.axvline(best_epoch - 1, color="0.5", ls=":", label=f"best epoch {best_epoch}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised log10)")
    ax.set_title("training"); ax.legend(fontsize=8)
    for j, p in enumerate(TARGETS):
        ax = axes.flat[j + 1]
        ax.scatter(Y_test[:, j], P_test[:, j], s=8, alpha=0.5)
        lo, hi = math.log10(BOUNDS[p][0]), math.log10(BOUNDS[p][1])
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{p} (test)", fontsize=10)
        ax.set_xlabel("actual log10"); ax.set_ylabel("predicted log10")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(BASE_DIR / "episode_param_dataset_ids.csv"),
                    help="CSV from make_param_dataset.py --with-id (episode_id needed for the seed split)")
    ap.add_argument("--out", default=str(BASE_DIR / "param_mlp"), help="output prefix")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-eval", action="store_true", help="closed-loop test on the test-set episodes")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--predict", default=None, help="CSV with the 28 feature columns: predict with a trained model")
    ap.add_argument("--pred-out", default="predicted_params.csv")
    args = ap.parse_args()

    if args.predict:
        model, norm, _ = load_model(f"{args.out}.npz")
        df = pd.read_csv(args.predict)
        pred = predict_params(model, norm, df)
        pred.to_csv(args.pred_out, index=False)
        print(f"Predicted {len(pred)} rows -> {args.pred_out}")
        return

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.data)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in {args.data}: {missing}")

    # ---- split by seed
    if "episode_id" in df.columns:
        groups = df["episode_id"].str.slice(1, 5).astype(int).to_numpy()
    else:
        print("WARNING: no episode_id column -> random row split (episodes of one seed may leak across splits)")
        groups = np.arange(len(df))
    uniq = rng.permutation(np.unique(groups))
    n_tr = int(round(args.split[0] * len(uniq)))
    n_va = int(round(args.split[1] * len(uniq)))
    g_tr, g_va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
    split = np.array(["train" if g in g_tr else "val" if g in g_va else "test" for g in groups])

    X = df[FEATURES].to_numpy(float)
    Y = np.log10(df[TARGETS].to_numpy(float))
    tr, va, te = split == "train", split == "val", split == "test"
    norm = {"x_mean": X[tr].mean(0), "x_std": X[tr].std(0) + 1e-9,
            "y_mean": Y[tr].mean(0), "y_std": Y[tr].std(0) + 1e-9}
    Xn = (X - norm["x_mean"]) / norm["x_std"]
    Yn = (Y - norm["y_mean"]) / norm["y_std"]

    sizes = [len(FEATURES)] + args.hidden + [len(TARGETS)]
    n_params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    print("=" * 90)
    print("Parameter-prediction MLP (NumPy)")
    print(f"Data     : {args.data}  ({len(df)} rows, {len(np.unique(groups))} seeds)")
    print(f"Split    : train {tr.sum()} / val {va.sum()} / test {te.sum()} rows (by seed)")
    print(f"Network  : {' -> '.join(map(str, sizes))}  ({n_params:,} weights), ReLU, Adam lr={args.lr}, "
          f"batch {args.batch}, L2 {args.weight_decay}")
    print(f"Targets  : log10 of {', '.join(TARGETS)}")
    print("=" * 90, flush=True)

    model = MLP(sizes, rng)
    t0 = time.time()
    hist, best_epoch = train(model, Xn[tr], Yn[tr], Xn[va], Yn[va], args.epochs, args.lr, args.batch,
                             args.weight_decay, args.patience, rng, log_every=100)
    print(f"Training done in {time.time() - t0:.0f}s, best epoch {best_epoch}")

    # ---- metrics (log10 space)
    def unnorm(Pn):
        return Pn * norm["y_std"] + norm["y_mean"]

    metrics = {"split_rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
               "network": sizes, "best_epoch": best_epoch, "args": vars(args)}
    for name, mask in (("train", tr), ("val", va), ("test", te)):
        metrics[name] = r2_rmse(Y[mask], unnorm(model.predict(Xn[mask])))
    metrics["test_baseline_train_mean"] = r2_rmse(Y[te], np.repeat(norm["y_mean"][None, :], te.sum(), axis=0))

    print(f"\n{'parameter':17s} {'R2 train':>9s} {'R2 val':>8s} {'R2 test':>8s} {'RMSE test':>10s} {'baseline RMSE':>14s}")
    for p in TARGETS:
        print(f"{p:17s} {metrics['train'][p]['r2']:9.3f} {metrics['val'][p]['r2']:8.3f} {metrics['test'][p]['r2']:8.3f} "
              f"{metrics['test'][p]['rmse_log10']:10.3f} {metrics['test_baseline_train_mean'][p]['rmse_log10']:14.3f}")
    print("(RMSE in log10 units: 1.0 = a factor of 10; baseline = always predict the training mean)")

    # ---- save
    save_model(f"{args.out}.npz", model, norm)
    test_df = df[te].reset_index(drop=True)
    pred_test = predict_params(model, norm, test_df)
    out_pred = test_df[["episode_id"]].copy() if "episode_id" in test_df else pd.DataFrame(index=test_df.index)
    for p in TARGETS:
        out_pred[f"actual_{p}"] = test_df[p]
        out_pred[f"pred_{p}"] = pred_test[p]
    out_pred.to_csv(f"{args.out}_test_predictions.csv", index=False)
    plot(f"{args.out}_training.png", hist, Y[te], unnorm(model.predict(Xn[te])), best_epoch)

    if args.sim_eval:
        if "episode_id" not in test_df:
            raise SystemExit("--sim-eval needs episode_id (use make_param_dataset.py --with-id)")
        from capture_states import load_v6_params
        v6, _ = load_v6_params((BASE_DIR / CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve())
        metrics["sim_eval"] = sim_eval(test_df, pred_test, {p: float(v6[p]) for p in TARGETS}, args.n_jobs)

    Path(f"{args.out}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out}.npz, {args.out}_metrics.json, {args.out}_training.png, {args.out}_test_predictions.csv")


if __name__ == "__main__":
    main()
