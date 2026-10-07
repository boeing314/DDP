"""
Shared code for v13 (adaptive DIPF parameters, alternative A).

  - config / scenario helpers (same scenarios as v11/v12)
  - worker pool (Ctrl+C only reaches the main process; workers exit when it dies)
  - closed-loop evaluation of a (base parameters, policy) pair on scenarios x futures
  - CMA-ES (Hansen's (mu/mu_w, lambda)-CMA-ES, NumPy only)
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

import policy as pol

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
PIPE = CONFIG["pipeline"]
ADA = PIPE["adaptive"]
CMA = PIPE["cmaes"]
SCEN_DIR = BASE_DIR / PIPE.get("scenarios_dir", "scenarios")
END_X = float(PIPE["end_x_m"])
MAX_TIME = float(PIPE["max_time_s"])
FUT = {k: int(v) for k, v in PIPE["futures"].items()}
ROAD_LENGTH = float(CONFIG["simulation"]["road_length_m"])
PARAMS = list(ADA["params"])
BASE = {k: float(v) for k, v in ADA["base_params"].items()}


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


def split_ids(name):
    ids = existing(seed_range(PIPE["seeds"][name]))
    if not ids:
        raise SystemExit(f"No {name} scenarios -- run step 1 first")
    return ids


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
# Policies
# =============================================================================

def make_policy(x, bias_only=False):
    """Optimiser vector -> policy dict (sign constraints applied)."""
    W, b = pol.unpack(x, PARAMS, ADA.get("signs"), bias_only)
    return pol.make_policy(W, b, PARAMS, ADA["limits"], ADA["clip"], ADA["smooth_s"])


def zero_policy():
    """W = 0, b = 0: behaves exactly like the fixed base parameters."""
    return make_policy(np.zeros(pol.n_dims(len(PARAMS))))


def bias_part(policy):
    """Same policy without the feature weights (a fixed re-tuning of the base: p = p0 * exp(clip(b)))."""
    out = dict(policy)
    out["W"] = np.zeros_like(np.asarray(policy["W"])).tolist()
    return out


# =============================================================================
# Closed-loop evaluation
# =============================================================================

_START_X = {}


def _run(job):
    """One segment run: (scenario id, policy or None, future seed) -> (success 1/0 or NaN, progress 0..1)."""
    sid, policy, fut = job
    import dipf_controller as dc
    if sid not in _START_X:
        _START_X[sid] = float(load_ego(sid)["centre_x"])
    distance = max(1.0, END_X - _START_X[sid])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_episode(SCEN_DIR / sid, BASE, distance, MAX_TIME, output_dir=tmp, sumo_seed=fut,
                               policy=policy)
        except Exception:
            return float("nan"), float("nan")
    ok = 1.0 if m["status"] == "success" else 0.0
    return ok, 1.0 if ok else max(0.0, min(1.0, float(m.get("final_progress", 0.0))))


def evaluate(pool, policies, sids, n_futures):
    """
    Run every policy (None = fixed base parameters) on every scenario x futures 1..n, in one batch.
    Returns per policy {scenario id: [(success, progress) per future]}.
    """
    futs = list(range(1, n_futures + 1))
    jobs = [(sid, p, f) for p in policies for sid in sids for f in futs]
    out = [{sid: [] for sid in sids} for _ in policies]
    for (i, sid, _), r in zip(((i, sid, f) for i in range(len(policies)) for sid in sids for f in futs),
                              pool.map(_run, jobs, chunksize=1)):
        out[i][sid].append(r)
    return out


def summarize(runs, w_progress=float(CMA.get("w_progress", 0.1))):
    """Per-policy result dict -> success rate, mean progress, J = success + w * progress."""
    ok = [o for v in runs.values() for o, _ in v if o == o]
    pr = [p for v in runs.values() for _, p in v if p == p]
    s = float(np.mean(ok)) if ok else 0.0
    g = float(np.mean(pr)) if pr else 0.0
    return {"success": s, "progress": g, "J": s + w_progress * g, "n": len(ok)}


def per_scenario(runs):
    """{scenario id: success probability over the futures}."""
    return {sid: (float(np.mean([o for o, _ in v if o == o])) if any(o == o for o, _ in v) else 0.0)
            for sid, v in runs.items()}


# =============================================================================
# CMA-ES  (minimisation; N. Hansen, "The CMA Evolution Strategy: A Tutorial", 2016)
# =============================================================================

class CMAES:
    def __init__(self, x0, sigma0, popsize=None, seed=0):
        N = len(x0)
        self.N = N
        self.lam = int(popsize or 4 + int(3 * math.log(N)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs
        self.chiN = math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N))
        self.m = np.array(x0, float)
        self.sigma = float(sigma0)
        self.C = np.eye(N)
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        z = self.rng.standard_normal((self.lam, self.N))
        return self.m + self.sigma * (z * D) @ B.T

    def tell(self, X, f):
        X, f = np.asarray(X), np.asarray(f)
        idx = np.argsort(f, kind="stable")[:self.mu]
        y = (X[idx] - self.m) / self.sigma
        yw = self.w @ y
        D2, B = np.linalg.eigh(self.C)
        Cinvsqrt = B @ np.diag(1.0 / np.sqrt(np.maximum(D2, 1e-20))) @ B.T
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinvsqrt @ yw)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.N + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = 0.5 * (self.C + self.C.T)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1

    def state(self):
        st = self.rng.bit_generator.state
        return {"m": self.m.tolist(), "sigma": self.sigma, "C": self.C.tolist(), "pc": self.pc.tolist(),
                "ps": self.ps.tolist(), "gen": self.gen, "lam": self.lam,
                "rng": {"state": str(st["state"]["state"]), "inc": str(st["state"]["inc"]),
                        "has_uint32": st["has_uint32"], "uinteger": st["uinteger"]}}

    def load(self, st):
        self.m, self.sigma, self.C = np.array(st["m"]), float(st["sigma"]), np.array(st["C"])
        self.pc, self.ps, self.gen = np.array(st["pc"]), np.array(st["ps"]), int(st["gen"])
        r = st["rng"]
        self.rng.bit_generator.state = {"bit_generator": "PCG64",
                                        "state": {"state": int(r["state"]), "inc": int(r["inc"])},
                                        "has_uint32": r["has_uint32"], "uinteger": r["uinteger"]}
