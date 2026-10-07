"""
Shared code for the v11 parameter-library pipeline.

  - config / scenario helpers
  - worker pool (Ctrl+C only reaches the main process; workers exit when it dies)
  - segment evaluation: success probability of a parameter set on a scenario,
    averaged over `futures` random futures (saved state reloaded with SUMO seeds 1..M)
  - DE operators (v4 algorithm, log10 space)
  - NumPy MLP with sigmoid outputs (multi-label, soft targets, binary cross-entropy)
  - model-input preprocessing of the 80-value encoding (+ left/right mirroring)
"""

import json
import math
import os
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
PIPE = CONFIG["pipeline"]
LIB = PIPE["library"]
SCEN_DIR = BASE_DIR / PIPE.get("scenarios_dir", "scenarios")
END_X = float(PIPE["end_x_m"])
MAX_TIME = float(PIPE["max_time_s"])
FUTURES = int(PIPE["futures"])
TAU = float(PIPE["success_threshold"])
ROAD_LENGTH = float(CONFIG["simulation"]["road_length_m"])
BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in LIB["bounds"].items()}
NAMES = list(BOUNDS)


# =============================================================================
# Scenarios
# =============================================================================

def seed_range(pair):
    return list(range(int(pair[0]), int(pair[1]) + 1))


def scen_id(seed):
    return f"s{int(seed):05d}"


def existing(seeds):
    """Scenario ids (in seed order) for the seeds whose scenario was created (same road as config.yaml)."""
    ids = [scen_id(s) for s in seeds if (SCEN_DIR / scen_id(s) / "ego.json").exists()]
    if ids:
        made_on = json.loads((SCEN_DIR / ids[0] / "meta.json").read_text(encoding="utf-8")).get("road_length_m")
        if made_on != ROAD_LENGTH:
            raise SystemExit(f"{SCEN_DIR.name}/ was made on a road of {made_on or 'unknown'} m, config.yaml says "
                             f"{ROAD_LENGTH:g} m -- move {SCEN_DIR.name}/ away and rerun step 1")
    return ids


def load_ego(sid):
    return json.loads((SCEN_DIR / sid / "ego.json").read_text(encoding="utf-8"))


# =============================================================================
# Worker pool
# =============================================================================

def _watchdog(parent):
    import threading
    import time

    def watch():
        while True:
            if os.getppid() != parent:
                os._exit(0)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


def _init(parent):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.argv = [sys.argv[0]]
    _watchdog(parent)


def make_pool(n_jobs=None):
    n = int(n_jobs or PIPE.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1))
    return ProcessPoolExecutor(max_workers=n, initializer=_init, initargs=(os.getpid(),)), n


# =============================================================================
# Segment evaluation
# =============================================================================

_START_X = {}


def _run(job):
    """One segment run: (scenario id, params, future seed) -> 1.0 success / 0.0 otherwise."""
    sid, params, fut = job
    import dipf_controller as dc
    if sid not in _START_X:
        _START_X[sid] = float(load_ego(sid)["centre_x"])
    distance = max(1.0, END_X - _START_X[sid])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_episode(SCEN_DIR / sid, params, distance, MAX_TIME, output_dir=tmp, sumo_seed=fut)
        except Exception:
            return sid, fut, float("nan")
    return sid, fut, 1.0 if m["status"] == "success" else 0.0


def run_outcomes(pool, param_list, sids, futures):
    """
    Outcomes of several parameter sets on several scenarios, all in one parallel batch.
    futures: list of SUMO seeds for the reloaded state (e.g. [1, 2, 3]).
    Returns one {scenario id: [1.0/0.0/NaN per future, in `futures` order]} per parameter set.
    """
    out = [{sid: [float("nan")] * len(futures) for sid in sids} for _ in param_list]
    if not sids or not param_list:
        return out
    jobs = [(sid, params, f) for params in param_list for sid in sids for f in futures]
    owner = [(i, sid, j) for i in range(len(param_list)) for sid in sids for j in range(len(futures))]
    for (i, sid, j), (_, _, ok) in zip(owner, pool.map(_run, jobs, chunksize=1)):
        out[i][sid][j] = ok
    return out


def prob(outcomes):
    """Success probability from a list of outcomes (errors ignored; all errors -> 0)."""
    v = [o for o in outcomes if o == o]
    return sum(v) / len(v) if v else 0.0


def success_probs_many(pool, param_list, sids, futures=FUTURES):
    """[{scenario id: success probability over futures 1..M}] for each parameter set."""
    runs = run_outcomes(pool, param_list, sids, list(range(1, futures + 1)))
    return [{sid: prob(o) for sid, o in r.items()} for r in runs]


def success_probs(pool, params, sids, futures=FUTURES):
    """{scenario id: success probability over futures 1..M} for one parameter set."""
    return success_probs_many(pool, [params], sids, futures)[0]


def config_params():
    return {k: float(CONFIG["dipf"][k]) for k in NAMES}


# =============================================================================
# DE (v4 algorithm, log10 space)
# =============================================================================

LO = np.array([math.log10(BOUNDS[k][0]) for k in NAMES])
HI = np.array([math.log10(BOUNDS[k][1]) for k in NAMES])
D = len(NAMES)


def to_params(v):
    return {k: float(10.0 ** v[i]) for i, k in enumerate(NAMES)}


def to_log(p):
    return np.clip([math.log10(p[k]) for k in NAMES], LO, HI)


def lhs(n, rng):
    u = np.empty((n, D))
    for d in range(D):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return LO + u * (HI - LO)


def mutate(pop, i, best, F, rng):
    r = rng.choice([k for k in range(len(pop)) if k not in (i, best)], size=4, replace=False)
    return np.clip(pop[best] + F * (pop[r[0]] - pop[r[1]] + pop[r[2]] - pop[r[3]]), LO, HI)


def crossover(theta, v, CR, rng):
    u = theta.copy()
    j, k = int(rng.integers(0, D)), 0
    while True:
        u[(j + k) % D] = v[(j + k) % D]
        k += 1
        if rng.random() >= CR or k == D:
            return u


# =============================================================================
# Model input: preprocessing of the 80-value encoding
# =============================================================================

from encoding import FEATURES as ENC, PER_SECTOR, SECTORS  # noqa: E402

MIRROR_PAIRS = [("front_left", "front_right"), ("left", "right"), ("rear_left", "rear_right")]


def preprocess(X):
    """
    X: (n, 80) raw encodings -> model inputs (same width):
      gap_long -> exp(-gap/20 m), gap_lat -> exp(-gap/2 m), both 0 for an empty sector
      (no jump between "far" and "empty"); drac zeroed in the left/right sectors (an
      artefact there: those vehicles overlap the ego along the road).
    """
    X = np.array(X, dtype=float, copy=True)
    for s in SECTORS:
        pres = X[:, ENC.index(f"{s}_present")]
        gl, gt = ENC.index(f"{s}_gap_long"), ENC.index(f"{s}_gap_lat")
        X[:, gl] = np.exp(-X[:, gl] / 20.0) * pres
        X[:, gt] = np.exp(-X[:, gt] / 2.0) * pres
        if s in ("left", "right"):
            X[:, ENC.index(f"{s}_drac")] = 0.0
    return X


def mirror(X):
    """Left/right mirror image of raw encodings (labels stay the same: the controller is symmetric)."""
    X = np.array(X, dtype=float, copy=True)
    Y = X.copy()
    for a, b in MIRROR_PAIRS:
        for f in PER_SECTOR:
            Y[:, ENC.index(f"{a}_{f}")], Y[:, ENC.index(f"{b}_{f}")] = X[:, ENC.index(f"{b}_{f}")], X[:, ENC.index(f"{a}_{f}")]
    for s in SECTORS:
        Y[:, ENC.index(f"{s}_dvy")] *= -1.0
    for g in ("vy", "heading", "y"):
        Y[:, ENC.index(g)] *= -1.0
    Y[:, ENC.index("clear_left")], Y[:, ENC.index("clear_right")] = X[:, ENC.index("clear_right")], X[:, ENC.index("clear_left")]
    return Y


# =============================================================================
# NumPy MLP (sigmoid outputs, BCE with soft targets)
# =============================================================================

class MLP:
    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.W = [rng.normal(0.0, math.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        acts, h = [X], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        return h, acts

    def proba(self, X):
        return 1.0 / (1.0 + np.exp(-self.forward(X)[0]))

    def loss(self, X, Y):
        z = self.forward(X)[0]
        return float(np.mean(np.maximum(z, 0) - z * Y + np.log1p(np.exp(-np.abs(z)))))

    def grads(self, X, Y):
        z, acts = self.forward(X)
        delta = (1.0 / (1.0 + np.exp(-z)) - Y) / (len(X) * Y.shape[1])
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i], gb[i] = acts[i].T @ delta, delta.sum(0)
            if i:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW, gb


def train_mlp(Xtr, Ytr, Xva, Yva, hidden, epochs, lr, wd, patience, rng, batch=64):
    m = MLP([Xtr.shape[1]] + list(hidden) + [Ytr.shape[1]], rng)
    params = m.W + m.b
    mom = [np.zeros_like(p) for p in params]
    var = [np.zeros_like(p) for p in params]
    step, best, best_w, wait, best_ep = 0, math.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            gW, gb = m.grads(Xtr[idx], Ytr[idx])
            g = [a + wd * W for a, W in zip(gW, m.W)] + gb
            step += 1
            for k, (p, gk) in enumerate(zip(m.W + m.b, g)):
                mom[k] = 0.9 * mom[k] + 0.1 * gk
                var[k] = 0.999 * var[k] + 0.001 * gk * gk
                p -= lr * (mom[k] / (1 - 0.9 ** step)) / (np.sqrt(var[k] / (1 - 0.999 ** step)) + 1e-8)
        v = m.loss(Xva, Yva)
        if v < best - 1e-6:
            best, best_w, wait, best_ep = v, [p.copy() for p in m.W + m.b], 0, ep
        else:
            wait += 1
            if wait >= patience:
                break
    k = len(m.W)
    m.W, m.b = best_w[:k], best_w[k:]
    return m, best_ep, best


# =============================================================================
# Trained model (step 4): load + predict
# =============================================================================

MODEL_FILE = BASE_DIR / "model.npz"


def load_model(path=MODEL_FILE):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    members = []
    for m in range(meta["ensemble"]):
        net = MLP.__new__(MLP)
        n = meta["n_layers"]
        net.W = [z[f"m{m}_W{i}"] for i in range(n)]
        net.b = [z[f"m{m}_b{i}"] for i in range(n)]
        members.append(net)
    return {"members": members, "mu": z["mu"], "sd": z["sd"], "meta": meta}


def predict_proba(model, X_raw):
    """(n, 80) raw encodings -> (n, K) success probability of every library set (ensemble mean)."""
    X = (preprocess(np.atleast_2d(X_raw)) - model["mu"]) / model["sd"]
    return np.mean([net.proba(X) for net in model["members"]], axis=0)
