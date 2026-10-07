"""
DIPF controller whose potential coefficients come from a trained MLP at every
time step. Works with either model:

  regression (train_param_mlp.py -> param_mlp.npz): predicts the 8 values directly
  classifier (train_param_classifier.py -> param_classifier.npz): predicts, for
             each parameter set in its library, the probability that the set
             succeeds here, and uses the most likely set (with optional
             hysteresis --margin so it does not flip back and forth)

Same controller as dipf_controller.py -- same potentials, limits, collision
handling, episode mode, metrics -- with one addition: before each control step
the MLP (train_param_mlp.py -> param_mlp.npz) predicts

    lambda_obs, alpha, tau_x_base, tau_y_base, sx, delta_lane, obs_radius_ahead, y_margin

from the current traffic situation, and those values are used for that step's
potential gradient. All other dipf.* values come from config.yaml.

The model input is built exactly like the training data (make_param_dataset.py):
    x_ego, y_ego         ego body centre
    vx_ego, vy_ego       controller-held ego velocity
    6 nearest vehicles   skewed distance d = hypot(dx, dy), x 0.33 if the
                         vehicle's centre is ahead of the ego; per vehicle
                         x/y/vx/vy relative to the ego (vehicle - ego), with
                         vx = SUMO speed and vy = SUMO lateral speed
Predictions are mapped back from log10 and clipped to the search bounds.

Implementation: imports dipf_controller and wraps its ego_control_step, so no
controller code is duplicated. The ego's context subscription is given a radius
covering the whole road, so the nearest-6 search sees every vehicle at no
extra TraCI cost; the obstacle list itself still applies the exact sensing
window, so obstacle handling is unchanged.

Per-step parameters are written to <results>/params.csv next to
trajectory.csv and metrics.json.

Usage:
    python mlp_dipf_controller.py --seed 42
    python mlp_dipf_controller.py --seed 42 --gui
    python mlp_dipf_controller.py --episode episodes/s0007_c0_start            # 150 m episode
    python mlp_dipf_controller.py --seed 42 --update-every 5 --smooth 0.5      # less jumpy coefficients
    python mlp_dipf_controller.py --eval-test                                  # MLP-per-step on the test episodes
    python mlp_dipf_controller.py --model param_classifier.npz --seed 42       # classifier picks a set each step
    python mlp_dipf_controller.py --model param_classifier.npz --eval-test --margin 0.1

Full-road (1000 m) test over many seeds -- model choosing coefficients every
step, compared with the fixed v6 parameters on the same seeds:
    python mlp_dipf_controller.py --model param_classifier.npz --eval-seeds 1000 1099
    python mlp_dipf_controller.py --model param_classifier.npz --eval-seeds 1000 1099 --margin 0.1 --update-every 5
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import dipf_controller as dc
from train_param_mlp import BOUNDS, FEATURES, TARGETS, load_model

BASE_DIR = Path(__file__).resolve().parent
traci, tc = dc.traci, dc.tc

K_NEAREST = 6
AHEAD_FACTOR = 0.33
FEATURE_RADIUS = 5000.0        # context radius (m) > road length: every vehicle is visible
FILL_REL = (200.0, 0.0, 0.0, 0.0)   # used only if fewer than 6 other vehicles exist (never in the data)


class MLPCoefficients:
    """Holds the model and turns the current SUMO state into DIPF parameters."""

    def __init__(self, model_path, update_every=1, smooth=0.0, margin=0.0, start_set=None):
        self.model, self.norm, self.meta = load_model(model_path)
        self.kind = self.meta.get("kind", "regression")
        if self.meta["features"] != FEATURES:
            raise ValueError(f"{model_path} was trained with different feature columns")
        if self.kind == "regression" and self.meta["targets"] != TARGETS:
            raise ValueError(f"{model_path} was trained with different target columns")
        if self.kind == "classifier":
            self.library = [{p: float(lib["params"][p]) for p in TARGETS} for lib in self.meta["library"]]
        self.update_every = max(1, int(update_every))
        self.smooth = float(smooth)          # regression: EMA weight on the previous log10 params (0 = none)
        self.margin = float(margin)          # classifier: switch only if new prob > current prob + margin
        self.start_set = start_set           # classifier: set in use before the first prediction (None = best)
        self.reset()

    def reset(self):
        self.step = 0
        self.log_params = None               # regression: current params, log10 space
        self.current_set = getattr(self, "start_set", None)   # classifier: index of the library set in use
        self.current_prob = math.nan
        self.log = []

    def features(self, x, y, vx, vy):
        ctx = traci.vehicle.getContextSubscriptionResults(dc.EGO_ID) or {}
        cands = []
        for vid, r in ctx.items():
            if vid == dc.EGO_ID:
                continue
            l_actual = dc._vehicle_static(vid)[3]
            xf, yf = r[tc.VAR_POSITION]
            xo, yo = dc.front_to_center(xf, yf, l_actual, r[tc.VAR_ANGLE])
            d = math.hypot(xo - x, yo - y)
            cands.append(((AHEAD_FACTOR * d if xo > x else d), vid, xo, yo, r[tc.VAR_SPEED], r[tc.VAR_SPEED_LAT]))
        cands.sort(key=lambda c: (c[0], c[1]))
        feat = [x, y, vx, vy]
        for n in range(K_NEAREST):
            if n < len(cands):
                _, _, xo, yo, vxo, vyo = cands[n]
                feat += [xo - x, yo - y, vxo - vx, vyo - vy]
            else:
                feat += list(FILL_REL)
        return np.asarray(feat, dtype=float), len(cands)

    def update(self, x, y, vx, vy):
        """Predict (every update_every steps) and return the params dict to use now."""
        if self.kind == "classifier":
            return self._update_classifier(x, y, vx, vy)
        n_veh = None
        if self.log_params is None or self.step % self.update_every == 0:
            feat, n_veh = self.features(x, y, vx, vy)
            xn = (feat - self.norm["x_mean"]) / self.norm["x_std"]
            logp = self.model.predict(xn[None, :])[0] * self.norm["y_std"] + self.norm["y_mean"]
            lo = np.log10([BOUNDS[p][0] for p in TARGETS])
            hi = np.log10([BOUNDS[p][1] for p in TARGETS])
            logp = np.clip(logp, lo, hi)
            if self.log_params is not None and self.smooth > 0:
                logp = self.smooth * self.log_params + (1.0 - self.smooth) * logp
            self.log_params = logp
        self.step += 1
        params = {p: float(10.0 ** v) for p, v in zip(TARGETS, self.log_params)}
        self.log.append({"time": traci.simulation.getTime(), **params,
                         "updated": n_veh is not None, "n_vehicles": n_veh if n_veh is not None else ""})
        return params

    def _update_classifier(self, x, y, vx, vy):
        n_veh = None
        if self.step == 0 or self.current_set is None or self.step % self.update_every == 0:
            feat, n_veh = self.features(x, y, vx, vy)
            xn = (feat - self.norm["x_mean"]) / self.norm["x_std"]
            prob = 1.0 / (1.0 + np.exp(-self.model.predict(xn[None, :])[0]))
            best = int(np.argmax(prob))
            if self.current_set is None or prob[best] > prob[self.current_set] + self.margin:
                self.current_set = best
            self.current_prob = float(prob[self.current_set])
        self.step += 1
        params = self.library[self.current_set]
        self.log.append({"time": traci.simulation.getTime(), **params, "set": self.current_set,
                         "p_set": self.current_prob, "updated": n_veh is not None,
                         "n_vehicles": n_veh if n_veh is not None else ""})
        return params


SOURCE = None                      # MLPCoefficients, set by install()
_original_step = dc.ego_control_step


def mlp_ego_control_step(dt, hist):
    """dipf_controller.ego_control_step with MLP-chosen coefficients for this step."""
    x, y = dc.vehicle_center(dc.EGO_ID, dc.EGO_LENGTH)
    params = SOURCE.update(x, y, mlp_ego_control_step.vx, mlp_ego_control_step.vy)
    dc.set_dipf_params(params)
    return _original_step(dt, hist)


def install(model_path, update_every=1, smooth=0.0, margin=0.0, start_set=None):
    """Hook the MLP into dipf_controller (idempotent per process)."""
    global SOURCE
    SOURCE = MLPCoefficients(model_path, update_every, smooth, margin, start_set)
    # The controller keeps its state (vx, vy, prev_ax, ...) as attributes on the
    # function object it calls by the global name `ego_control_step`; after the
    # swap that name -- inside dipf_controller too -- refers to the wrapper.
    for k in ("prev_ax", "prev_ay", "vx", "vy", "angle"):
        setattr(mlp_ego_control_step, k, getattr(dc.ego_control_step, k))
    dc.ego_control_step = mlp_ego_control_step
    dc.context_radius = lambda: FEATURE_RADIUS


def write_param_log(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "params.csv").open("w", newline="", encoding="utf-8") as f:
        extra = ["set", "p_set"] if SOURCE.kind == "classifier" else []
        w = csv.DictWriter(f, fieldnames=["time", *TARGETS, *extra, "updated", "n_vehicles"])
        w.writeheader()
        w.writerows(SOURCE.log)


# =============================================================================
# Batch evaluation on episodes (in-process, fresh SUMO per episode)
# =============================================================================

def _eval_worker_init(model_path, update_every, smooth, margin):
    sys.argv = [sys.argv[0]]
    install(model_path, update_every, smooth, margin)


def _eval_one(ep_id):
    import tempfile
    SOURCE.reset()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m = dc.run_episode(BASE_DIR / "episodes" / ep_id, {}, EVAL_DISTANCE, EVAL_MAX_TIME, output_dir=tmp)
            return ep_id, m["status"], float(m["C"])
        except Exception:
            return ep_id, "error", math.nan


def _seed_worker_init(model_path, update_every, smooth, margin, start_set=None):
    sys.argv = [sys.argv[0]]
    if model_path is not None:
        install(model_path, update_every, smooth, margin, start_set)


def _run_seed(job):
    seed, params, max_time, out_root, use_model = job
    if use_model:
        SOURCE.reset()
    out = Path(out_root) / f"seed_{seed}"
    try:
        m = dc.run_seed(seed, params, output_dir=out, max_time=max_time)
    except Exception as exc:
        return {"seed": seed, "status": "error", "progress": math.nan, "C": math.nan, "error": str(exc)[:200]}
    row = {"seed": seed, "status": m["status"], "progress": float(m["final_progress"]), "C": float(m["C"]),
           "final_x": float(m["final_x"]), "avg_speed": float(m["avg_speed"])}
    if use_model:
        write_param_log(out / f"results_{seed}")
        sets = [r.get("set") for r in SOURCE.log if r.get("set") is not None]
        row["n_steps"] = len(SOURCE.log)
        row["set_switches"] = sum(a != b for a, b in zip(sets, sets[1:])) if sets else ""
    return row


def eval_seeds(model_path, seeds, update_every, smooth, margin, max_time, n_jobs, out_dir, start_set=None):
    from capture_states import load_v6_params
    v6, v6_src = load_v6_params((BASE_DIR / dc.CONFIG["capture"].get("v6_dir", "../../v6/run1")).resolve())
    v6 = {p: float(v6[p]) for p in TARGETS}
    kind = MLPCoefficients(model_path).kind
    out_dir = Path(out_dir)
    print("=" * 90)
    print(f"Full-road test (goal x={dc.EGO_GOAL_X:g} m) on seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} runs each)")
    print(f"Model   : {model_path} [{kind}], coefficients re-chosen every {update_every} step(s)"
          + (f", margin {margin}" + (f", starting from set {start_set}" if start_set is not None else "")
             if kind == "classifier" else f", smooth {smooth}"))
    print(f"Baseline: fixed v6 parameters ({v6_src})")
    print(f"Limit   : {max_time:g} s after the ego departs | output: {out_dir}")
    print("=" * 90, flush=True)

    results = {}
    for label, model, params in (("model", model_path, {}), ("v6", None, v6)):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_seed_worker_init,
                                 initargs=(model, update_every, smooth, margin, start_set)) as ex:
            jobs = [(sd, params, max_time, out_dir / label, model is not None) for sd in seeds]
            results[label] = list(ex.map(_run_seed, jobs, chunksize=1))
        print(f"  {label:5s} runs done in {time.time() - t0:.0f}s", flush=True)

    def summ(rows):
        st = [r["status"] for r in rows]
        return {"success": st.count("success") / len(st), "collision": st.count("collision") / len(st),
                "timeout/stuck": (st.count("timeout") + st.count("stuck")) / len(st), "error": st.count("error"),
                "progress": float(np.nanmean([r["progress"] for r in rows])),
                "C": float(np.nanmean([r["C"] for r in rows]))}

    sm, sv = summ(results["model"]), summ(results["v6"])
    print(f"\n{'':22s} {'success':>8s} {'collision':>10s} {'timeout':>8s} {'progress':>9s} {'mean C':>8s}")
    for lab, sx in (("model every step", sm), ("fixed v6", sv)):
        print(f"{lab:22s} {sx['success']:8.1%} {sx['collision']:10.1%} {sx['timeout/stuck']:8.1%} "
              f"{sx['progress']:9.3f} {sx['C']:8.3f}" + (f"   ({sx['error']} errors)" if sx["error"] else ""))
    both = sum(a["status"] == "success" and b["status"] == "success" for a, b in zip(results["model"], results["v6"]))
    only_m = sum(a["status"] == "success" and b["status"] != "success" for a, b in zip(results["model"], results["v6"]))
    only_v = sum(a["status"] != "success" and b["status"] == "success" for a, b in zip(results["model"], results["v6"]))
    print(f"\nper seed: both succeed {both}, only model {only_m}, only v6 {only_v}")
    if kind == "classifier":
        sw = [r["set_switches"] for r in results["model"] if r.get("set_switches") != ""]
        if sw:
            print(f"set switches per run: mean {np.mean(sw):.1f}, max {max(sw)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "model_status", "model_progress", "model_C", "v6_status", "v6_progress", "v6_C"])
        for a, b in zip(results["model"], results["v6"]):
            w.writerow([a["seed"], a["status"], a["progress"], a["C"], b["status"], b["progress"], b["C"]])
    (out_dir / "summary.json").write_text(json.dumps({"model": sm, "v6": sv, "model_path": str(model_path),
                                                      "update_every": update_every, "margin": margin,
                                                      "smooth": smooth, "max_time": max_time,
                                                      "seeds": [seeds[0], seeds[-1]]}, indent=2))
    print(f"Per-seed results: {out_dir / 'summary.csv'}  (trajectories + params.csv under {out_dir}/model/)")


EVAL_DISTANCE = float(dc.CONFIG.get("episode_opt", {}).get("distance_m", 150.0))
EVAL_MAX_TIME = float(dc.CONFIG.get("episode_opt", {}).get("max_time_s", 60.0))


def eval_test(model_path, update_every, smooth, margin, n_jobs):
    pred_file = Path(str(model_path).replace(".npz", "_test_predictions.csv"))
    ep_ids = [r["episode_id"] for r in csv.DictReader(pred_file.open(encoding="utf-8"))]
    print(f"MLP-per-step controller on {len(ep_ids)} test episodes ({EVAL_DISTANCE:g} m each)...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_eval_worker_init,
                             initargs=(model_path, update_every, smooth, margin)) as ex:
        res = list(ex.map(_eval_one, ep_ids, chunksize=4))
    st = [r[1] for r in res]
    C = [r[2] for r in res if math.isfinite(r[2])]
    print(f"  done in {time.time() - t0:.0f}s: success {st.count('success')}/{len(st)} = "
          f"{st.count('success') / len(st):.1%}, mean C {np.mean(C):.3f}, errors {st.count('error')}")
    metrics_file = Path(str(model_path).replace(".npz", "_metrics.json"))
    if metrics_file.exists() and "classifier" in Path(model_path).name:
        r = json.loads(metrics_file.read_text())["results"]["test"]
        print(f"  for comparison (same episodes, set chosen once at the start, by lookup): classifier "
              f"{r['classifier']['success']:.1%}, v6 {r['v6_only']['success']:.1%}, "
              f"oracle {r['oracle_library']['success']:.1%}")
    elif metrics_file.exists():
        se = json.loads(metrics_file.read_text()).get("sim_eval")
        if se:
            print(f"  for comparison (same episodes): MLP once per episode {se['mlp_predicted']['success_rate']:.1%}, "
                  f"v6 params {se['v6_params']['success_rate']:.1%}, "
                  f"per-episode optimum {se['per_episode_optimum']['success_rate']:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(BASE_DIR / "param_mlp.npz"))
    ap.add_argument("--seed", type=int, default=None, help="SUMO traffic seed (default: config.yaml)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--episode", default=None, help="episode folder (state.xml + ego.json) to start from")
    ap.add_argument("--distance", type=float, default=150.0, help="episode mode: metres to cover")
    ap.add_argument("--max-time", type=float, default=60.0, help="episode mode: timeout (s)")
    ap.add_argument("--update-every", type=int, default=1, help="re-predict every N control steps (default 1)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="EMA on the coefficients in log10 space, weight of the previous value (0..1, default 0)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="classifier: switch set only if its probability beats the current set's by this much")
    ap.add_argument("--start-set", type=int, default=None,
                    help="classifier: library set in use before the first prediction (0 = v6); with --margin the "
                         "controller only leaves it when another set is more likely by that margin")
    ap.add_argument("--eval-test", action="store_true", help="evaluate on the model's test episodes and exit")
    ap.add_argument("--eval-seeds", type=int, nargs=2, metavar=("START", "END"), default=None,
                    help="full-road test on seeds START..END (inclusive): model every step vs fixed v6")
    ap.add_argument("--eval-max-time", type=float, default=300.0,
                    help="--eval-seeds: stop a run this many seconds after the ego departs (default 300)")
    ap.add_argument("--eval-out", default=None, help="--eval-seeds output folder (default eval_seeds_<model>)")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    if args.eval_seeds:
        seeds = list(range(args.eval_seeds[0], args.eval_seeds[1] + 1))
        out = args.eval_out or str(BASE_DIR / f"eval_seeds_{Path(args.model).stem}")
        eval_seeds(args.model, seeds, args.update_every, args.smooth, args.margin, args.eval_max_time,
                   args.n_jobs, out, args.start_set)
        return

    if args.eval_test:
        eval_test(args.model, args.update_every, args.smooth, args.margin, args.n_jobs)
        return

    install(args.model, args.update_every, args.smooth, args.margin, args.start_set)
    if args.seed is not None:
        dc.SEED = args.seed
    dc.SUMO_GUI = args.gui
    if args.episode:
        dc.EPISODE_DIR = Path(args.episode).resolve()
        dc.EPISODE_EGO = json.loads((dc.EPISODE_DIR / "ego.json").read_text(encoding="utf-8"))
        dc.EPISODE_MAX_TIME = float(args.max_time)
        dc.SEED = dc.EPISODE_EGO.get("seed")
        dc.EGO_START_X = float(dc.EPISODE_EGO["centre_x"])
        dc.EGO_GOAL_X = min(dc.EGO_START_X + float(args.distance), dc.ROAD_LENGTH_M - dc.GOAL_X_OFFSET)
        dc.SEGMENT_LENGTH_M = dc.EGO_GOAL_X - dc.EGO_START_X
        dc.RUN_NAME = f"mlp_results_{dc.EPISODE_DIR.name}"
    else:
        dc.RUN_NAME = f"mlp_results_{dc.SEED}"

    print(f"MLP coefficients: {args.model} [{SOURCE.kind}]  (re-predict every {args.update_every} step(s), "
          f"{'margin=' + str(args.margin) if SOURCE.kind == 'classifier' else 'smooth=' + str(args.smooth)})")
    dc.run_simulation()

    results_dir = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results"))) / dc.RUN_NAME
    write_param_log(results_dir)
    print(f"Per-step coefficients: {results_dir / 'params.csv'}")


if __name__ == "__main__":
    main()
