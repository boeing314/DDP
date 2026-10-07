"""
Shared evaluation machinery (optimize_de.py, optimize_optuna.py, evaluate.py).

One evaluation = full-road runs (dipf_controller.run_seed: in-process, fresh
race-free SUMO per run) of one parameter set over a list of seeds, in a pool of
worker processes. Workers ignore Ctrl+C (only the main process reacts) and stop
by themselves if the main process disappears.

Objective (lower is better):
    J = (1 - success_rate) + w_progress * (1 - mean_progress)
"""

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
OPT = CONFIG["optimization"]

_PARENT = None


def _watchdog(parent_pid):
    """Exit this worker as soon as the main process is gone -- even while idle."""
    import threading
    import time

    def watch():
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)                 # its SUMO (if any) dies with it (PR_SET_PDEATHSIG)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


def _init(parent_pid):
    global _PARENT
    _PARENT = parent_pid
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.argv = [sys.argv[0]]
    _watchdog(parent_pid)


def _run(job):
    seed, params, max_time = job
    if _PARENT is not None and os.getppid() != _PARENT:
        raise SystemExit("main process gone")
    import dipf_controller as dc
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_seed(seed, params, output_dir=tmp, max_time=max_time)
        except Exception as exc:
            return {"seed": seed, "status": "error", "progress": 0.0, "C": math.nan, "error": str(exc)[:200]}
    return {"seed": seed, "status": m["status"], "progress": float(m["final_progress"]), "C": float(m["C"])}


def make_pool(n_jobs=None):
    n = int(n_jobs or OPT.get("n_jobs") or max(1, (os.cpu_count() or 2) - 1))
    return ProcessPoolExecutor(max_workers=n, initializer=_init, initargs=(os.getpid(),)), n


def seed_range(pair):
    return list(range(int(pair[0]), int(pair[1]) + 1))


def summarize(rows, w_progress=None):
    w = float(OPT.get("w_progress", 0.2) if w_progress is None else w_progress)
    st = [r["status"] for r in rows]
    n = len(rows)
    succ = st.count("success") / n
    prog = float(np.mean([min(1.0, max(0.0, r["progress"])) for r in rows]))
    return {"J": (1.0 - succ) + w * (1.0 - prog), "success": succ, "progress": prog,
            "collision": st.count("collision") / n, "timeout": (st.count("timeout") + st.count("stuck")) / n,
            "errors": st.count("error"), "n": n,
            "mean_C": float(np.nanmean([r["C"] for r in rows])) if any(r["C"] == r["C"] for r in rows) else math.nan}


def evaluate(pool, params, seeds, max_time=None):
    """Run params on every seed; returns (summary, per-seed rows)."""
    mt = float(OPT.get("max_time_s", 300.0) if max_time is None else max_time)
    rows = list(pool.map(_run, [(s, params, mt) for s in seeds], chunksize=1))
    return summarize(rows), rows


def fmt(s):
    return (f"J={s['J']:.4f} success={s['success']:.1%} progress={s['progress']:.3f} "
            f"collision={s['collision']:.1%} timeout={s['timeout']:.1%}"
            + (f" errors={s['errors']}" if s["errors"] else ""))


def config_params():
    """The tunable dipf.* values currently in config.yaml."""
    return {k: float(CONFIG["dipf"][k]) for k in OPT["bounds"]}


def apply_to_config(best):
    """Write best values into config.yaml's dipf section in place (old value kept as a comment)."""
    f = BASE_DIR / "config.yaml"
    lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
    in_dipf, done = False, set()
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(" "):
            in_dipf = line.startswith("dipf:")
            continue
        if not in_dipf:
            continue
        for k, v in best.items():
            if line.strip().startswith(f"{k}:") and k not in done:
                old = line.split(":", 1)[1].split("#", 1)[0].strip()
                lines[i] = f"  {k}: {v!r}".ljust(40) + f"# was {old} (before optimisation)\n"
                done.add(k)
    missing = set(best) - done
    if missing:
        raise KeyError(f"not found in config.yaml dipf section: {sorted(missing)}")
    f.write_text("".join(lines), encoding="utf-8")
