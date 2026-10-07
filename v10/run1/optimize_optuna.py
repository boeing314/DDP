"""
v10: Optuna (TPE) search for ONE fixed parameter set -- same controller, search
space and objective as v9's differential evolution, so the two optimisers can be
compared directly.

    J = (1 - success_rate) + w_progress * (1 - mean_progress)      (minimise)

over full-road runs on the training seeds (optimization.train_seeds); the same
seeds for every trial. Each dipf.* key in optimization.bounds is suggested
log-uniformly; tau_x_base stays fixed.

Sampler : TPE, multivariate (models parameter interactions), first trial = the
          values currently in config.yaml, then n_startup random trials.
Pruning : the training seeds are run in `pruning_chunks` chunks; after each
          chunk the J so far is reported and a trial that is clearly worse
          than the median of earlier trials at the same point is stopped
          (MedianPruner) -- saves most of the cost of bad trials.
Storage : optuna_v10.db (SQLite). Rerunning continues the same study until it
          has n_trials finished trials; trials interrupted by Ctrl+C are marked
          failed and do not count.
At the end the best trial is scored on the VALIDATION seeds next to the start values.

Outputs:
    optuna_v10.db       the study (inspect with optuna-dashboard or optuna.load_study)
    optuna_trials.csv   one row per trial
    optuna_best.json    best parameters, train + validation results

Usage:
    python optimize_optuna.py                       # settings from config.yaml -> optimization
    python optimize_optuna.py --n-trials 80 --n-jobs 7
    python optimize_optuna.py --no-pruning
    python optimize_optuna.py --finish --apply      # no new trials: validate the best so far, write config
"""

import argparse
import json
import math
import time

import numpy as np
import optuna

from runner import BASE_DIR, OPT, apply_to_config, config_params, evaluate, fmt, make_pool, seed_range, summarize

BOUNDS = {k: (float(v[0]), float(v[1])) for k, v in OPT["bounds"].items()}
NAMES = list(BOUNDS)
if "tau_x_base" in BOUNDS:
    raise SystemExit("tau_x_base is fixed -- remove it from optimization.bounds")
DB = BASE_DIR / "optuna_v10.db"
STUDY = "v10_baseline"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=int(OPT.get("n_trials", 150)),
                    help="total finished trials wanted in the study (default optimization.n_trials)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--no-pruning", action="store_true")
    ap.add_argument("--finish", action="store_true", help="run no new trials, only validate / save the best")
    ap.add_argument("--apply", action="store_true", help="write the best set into config.yaml at the end")
    ap.add_argument("--train-seeds", type=int, nargs=2, default=None)
    ap.add_argument("--val-seeds", type=int, nargs=2, default=None)
    args = ap.parse_args()

    train = seed_range(args.train_seeds or OPT["train_seeds"])
    val = seed_range(args.val_seeds or OPT["validation_seeds"])
    n_chunks = 1 if args.no_pruning else max(1, int(OPT.get("pruning_chunks", 4)))
    chunks = [list(c) for c in np.array_split(train, n_chunks)]
    start = config_params()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{DB}"
    optuna.create_study(study_name=STUDY, storage=storage, direction="minimize", load_if_exists=True)
    n_existing = len(optuna.load_study(study_name=STUDY, storage=storage).trials)
    # seed offset by the number of existing trials: a resumed study must not regenerate
    # the same "random" startup trials it already ran
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15,
                                         seed=int(OPT.get("rng_seed", 0)) + n_existing)
    pruner = (optuna.pruners.NopPruner() if args.no_pruning
              else optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    study = optuna.load_study(study_name=STUDY, storage=storage, sampler=sampler, pruner=pruner)

    # trials left RUNNING by an interrupted run -> failed (they don't count)
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if not study.trials:
        study.enqueue_trial(start)

    finished = lambda: len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,
                                                                    optuna.trial.TrialState.PRUNED)))
    remaining = 0 if args.finish else max(0, args.n_trials - finished())
    pool, n_jobs = make_pool(args.n_jobs)

    print("=" * 100)
    print("v10 Optuna (TPE, multivariate) -- one fixed parameter set (baseline)")
    print(f"Objective : J = (1 - success) + {OPT.get('w_progress', 0.2)} * (1 - progress), full road")
    print(f"Seeds     : train {train[0]}..{train[-1]} ({len(train)}, in {n_chunks} chunk(s) for pruning), "
          f"validation {val[0]}..{val[-1]} ({len(val)})")
    print(f"Study     : {DB.name} -- {finished()} finished trials so far, {remaining} to run, {n_jobs} parallel sims")
    print("Search (log-uniform)          start value")
    for k in NAMES:
        print(f"  {k:<16s} [{BOUNDS[k][0]:g}, {BOUNDS[k][1]:g}]   {start[k]:.6g}")
    print("=" * 100, flush=True)

    t0 = time.time()

    def objective(trial):
        params = {k: trial.suggest_float(k, BOUNDS[k][0], BOUNDS[k][1], log=True) for k in NAMES}
        rows = []
        for step, chunk in enumerate(chunks):
            _, r = evaluate(pool, params, chunk)
            rows += r
            s = summarize(rows)
            trial.report(s["J"], step)
            if step < len(chunks) - 1 and trial.should_prune():
                trial.set_user_attr("summary", s)
                print(f"  trial {trial.number:4d} PRUNED after {len(rows)} seeds: {fmt(s)}  [{time.time() - t0:.0f}s]",
                      flush=True)
                raise optuna.TrialPruned()
        trial.set_user_attr("summary", s)
        try:
            best_before = study.best_value
        except ValueError:                      # no completed trial yet
            best_before = math.inf
        tag = "  <- best so far" if s["J"] < best_before else ""
        print(f"  trial {trial.number:4d} {fmt(s)}{tag}  [{time.time() - t0:.0f}s]", flush=True)
        return s["J"]

    try:
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=False)
        best = study.best_trial
        best_params = {k: float(best.params[k]) for k in NAMES}
        print(f"\nBest trial #{best.number}: {fmt(best.user_attrs['summary'])}")
        print(f"Validation on {len(val)} unseen seeds ...", flush=True)
        v_best, _ = evaluate(pool, best_params, val)
        v_start, _ = evaluate(pool, start, val)
    except KeyboardInterrupt:
        print("\nInterrupted -- finished trials are in the study; rerun to continue.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()

    # trial table + importances
    rows = []
    for t in study.get_trials(deepcopy=False):
        s = t.user_attrs.get("summary", {})
        rows.append({"number": t.number, "state": t.state.name, "value": t.value, **t.params,
                     **{f"s_{k}": s.get(k) for k in ("success", "progress", "collision", "n")}})
    import csv
    with (BASE_DIR / "optuna_trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "number", k)))
        w.writeheader()
        w.writerows(rows)
    try:
        imp = optuna.importance.get_param_importances(study)
        print("Parameter importance (fANOVA): " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    except Exception as exc:
        imp = {}
        print(f"(parameter importance unavailable: {exc})")

    out = {"best_params": best_params, "best_trial": best.number, "train": best.user_attrs["summary"],
           "validation": v_best, "validation_start_values": v_start, "start_params": start,
           "importance": imp, "train_seeds": [train[0], train[-1]], "validation_seeds": [val[0], val[-1]]}
    (BASE_DIR / "optuna_best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(f"Best (train)       : {fmt(best.user_attrs['summary'])}")
    print(f"Best (validation)  : {fmt(v_best)}")
    print(f"Start (validation) : {fmt(v_start)}")
    for k in NAMES:
        print(f"  {k:<16s} = {best_params[k]!r}")
    print("Saved optuna_best.json, optuna_trials.csv")
    if args.apply:
        apply_to_config(best_params)
        print("Best parameters written to config.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
