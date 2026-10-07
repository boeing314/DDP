"""
Step 2 -- parameter library: a small set of DIPF parameter sets that, between
them, solve as many scenarios as possible (sequential covering).

Round 0: Optuna finds the best single set on the initial pool (first trial = config.yaml values).
Round r: the RESIDUAL = pool scenarios that no library set solves (success prob < tau).
  1. top-up: if the residual is smaller than min_pool, reserve scenarios are added to
     the pool; every library set is run on them first and only the ones none of them
     solves enter the residual (the others still count as covered pool scenarios).
  2. an Optuna study (TPE, multivariate, log-uniform ranges) on the residual maximises
     the COVERAGE GAIN over the current library
         gain(theta) = mean_{s in residual} max(0, p_theta(s) - best_library_p(s))
     i.e. it rewards a set only for what the library cannot do yet (partial
     successes count, so the search has a signal even when nothing is fully solved).
     Trials are evaluated in batches (constant liar) so all workers stay busy. With
     pruning, a trial first runs future 1 on every residual scenario and is dropped
     if its gain so far is below the median of earlier trials at that stage; only
     the survivors run the remaining futures (a pruned trial costs 1/3).
     A round ends after n_trials, after `patience` trials without a better gain, or
     when the residual is fully solved.
  3. the round's best set is run on the whole pool and on the validation scenarios,
     and is KEPT only if it raises the validation success with selection
         V(L) = mean_s p_{k*(s)}(s),  k*(s) chosen on futures 1..3, scored on 4..6
                (and vice versa; picking and scoring on the same runs would reward noise)
     by at least min_val_gain. The first set is always kept.
  4. residual scenarios the round's best set did not solve get a strike; after
     drop_after_rounds strikes a scenario is set aside as unsolvable, so later
     rounds do not keep chasing it.
Stops at max_sets, max_rounds, max_rejects rejected rounds in a row, or when fewer
than min_residual unsolved scenarios remain and the reserve is used up.

Every trial is scored on the same scenarios x futures (common random numbers).
Everything is saved as it goes (state after each round / top-up, every trial in the
Optuna database); rerunning continues where it stopped.

Outputs:
    library.json          the sets (params) + per-set summaries   (rewritten after every round)
    library_state.json    full state incl. every set's success probs on pool + validation runs
    library_optuna.db     one Optuna study per round ("round0", "round1", ...)
    library_log.jsonl     every trial

Usage:
    python step2_build_library.py                 # start or resume
    python step2_build_library.py --n-jobs 7
    python step2_build_library.py --report        # print the summary / correlations of the current library
    python step2_build_library.py --restart       # move old results to library_old_<time>/ and start over
"""

import argparse
import json
import math
import shutil
import time
import warnings

import numpy as np
import optuna
from optuna.trial import TrialState

from common import (BASE_DIR, BOUNDS, CONFIG, END_X, LIB, NAMES, PIPE, ROAD_LENGTH, TAU, config_params, existing,
                    make_pool, prob, run_outcomes, seed_range, success_probs_many)

OPT = LIB["optuna"]
LF = int(OPT.get("futures", PIPE["futures"]))         # futures for every pool evaluation in this step
VF = int(LIB.get("val_futures", 2 * LF))               # validation futures (split in two halves)
H = VF // 2

STATE = BASE_DIR / "library_state.json"
DB = BASE_DIR / "library_optuna.db"
LOG = BASE_DIR / "library_log.jsonl"
OUT = BASE_DIR / "library.json"


# =============================================================================
# State
# =============================================================================

def new_state(train_ids, val_ids):
    n0 = int(LIB["initial_pool"])
    return {"pool": train_ids[:n0], "reserve": train_ids[n0:], "val": val_ids, "dropped": [], "fails": {},
            "sets": [], "rejected": [], "round": 0, "rejects_in_row": 0, "current": None, "stopped": None,
            "futures": LF, "val_futures": VF, "tau": TAU, "end_x": END_X, "road_length_m": ROAD_LENGTH}


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    tmp.replace(path)


def log(**kw):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


def best_lib(state, sid):
    return max((s["train"][sid] for s in state["sets"]), default=0.0)


def residual(state):
    dropped = set(state["dropped"])
    return [s for s in state["pool"] if s not in dropped and best_lib(state, s) < TAU]


def val_score(vals, sids):
    """Validation success with per-scenario selection, cross-fitted over the two halves of the futures."""
    if not vals or not sids:
        return 0.0
    tot = 0.0
    for sid in sids:
        a = [prob(v[sid][:H]) for v in vals]
        b = [prob(v[sid][H:]) for v in vals]
        ka, kb = int(np.argmax(a)), int(np.argmax(b))         # ties -> earlier set
        tot += 0.5 * (prob(vals[ka][sid][H:]) + prob(vals[kb][sid][:H]))
    return tot / len(sids)


def val_any(vals, sids):
    """Fraction of validation scenarios some set solves (all futures; optimistic)."""
    if not vals or not sids:
        return 0.0
    return float(np.mean([max(prob(v[s]) for v in vals) >= TAU for s in sids]))


def write_library(state):
    sets = [{"id": k, "round": s["round"], "params": s["params"],
             "train_alone": s["train_alone"], "val_alone": s["val_alone"], "val_gain": s["val_gain"],
             "val_score_after": s["val_score_after"]} for k, s in enumerate(state["sets"])]
    OUT.write_text(json.dumps({"names": NAMES, "futures": LF, "tau": TAU, "end_x": END_X, "sets": sets,
                               "val_score": sets[-1]["val_score_after"] if sets else 0.0,
                               "unsolvable": len(state["dropped"]), "stopped": state["stopped"]}, indent=2))


# =============================================================================
# Round pieces
# =============================================================================

def top_up(state, pool):
    need = int(LIB["min_pool"])
    R = residual(state)
    while len(R) < need and state["reserve"]:
        live = [s for s in state["pool"] if s not in set(state["dropped"])]
        frac = max(len(R) / max(len(live), 1), 0.1)          # unsolved fraction so far
        n = min(len(state["reserve"]), max(10, math.ceil((need - len(R)) / frac)))
        batch, state["reserve"] = state["reserve"][:n], state["reserve"][n:]
        t0 = time.time()
        probs = success_probs_many(pool, [s["params"] for s in state["sets"]], batch, LF)
        for st, pr in zip(state["sets"], probs):
            st["train"].update(pr)
        state["pool"] += batch
        before = len(R)
        R = residual(state)
        print(f"  top-up: +{len(batch)} reserve scenarios, {len(R) - before} not solved by any set -> "
              f"residual {len(R)} (reserve left {len(state['reserve'])}) [{time.time() - t0:.0f}s]", flush=True)
        save(state, STATE)
    return R


def run_optuna(state, pool, R, base, rnd, t0):
    """Optuna study on the residual; returns (best params, its probs on the residual, its gain)."""
    NT, B = int(OPT["n_trials"]), max(1, int(OPT.get("batch", 4)))
    patience, prune = int(OPT.get("patience", NT)), bool(OPT.get("prune", True)) and LF > 1
    max_gain = float(np.mean([1.0 - base[s] for s in R]))

    def gain(pr):
        return float(np.mean([max(0.0, pr[s] - base[s]) for s in R]))

    def describe(pr):
        return f"gain {gain(pr):.4f}  solves {sum(pr[s] >= TAU for s in R):3d}/{len(R)}  mean p {np.mean([pr[s] for s in R]):.3f}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    storage, name = f"sqlite:///{DB}", f"round{rnd}"
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize", load_if_exists=True)
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)):
        study.tell(t.number, state=TrialState.FAIL)          # interrupted before it finished
    n_existing = len(study.trials)
    # seed offset by the number of existing trials: a resumed study must not redraw the same points
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True,
                                         n_startup_trials=int(OPT.get("n_startup_trials", 16)),
                                         seed=int(OPT.get("rng_seed", 0)) + 1000 * rnd + n_existing)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=int(OPT.get("prune_startup", 10)), n_warmup_steps=0)
              if prune else optuna.pruners.NopPruner())
    study = optuna.load_study(study_name=name, storage=storage, sampler=sampler, pruner=pruner)
    if rnd == 0 and n_existing == 0:
        study.enqueue_trial(config_params())

    def finished():
        return sorted((t for t in study.get_trials(deepcopy=False) if t.state in (TrialState.COMPLETE, TrialState.PRUNED)),
                      key=lambda t: t.number)

    def best_and_stall(fin):
        comp = [t for t in fin if t.state == TrialState.COMPLETE]
        if not comp:
            return None, 0
        best = max(comp, key=lambda t: (t.value, -t.number))            # earliest of equal gains
        return best, sum(t.number > best.number for t in fin)

    fin = finished()
    if fin:
        best, _ = best_and_stall(fin)
        print(f"  resumed round {rnd}: {len(fin)} trials done"
              + (f", best gain {best.value:.4f} (trial {best.number})" if best else ""), flush=True)

    while True:
        fin = finished()
        best, stall = best_and_stall(fin)
        if len(fin) >= NT:
            why = f"{NT} trials done"
        elif best is not None and stall >= patience:
            why = f"no better gain in {stall} trials"
        elif best is not None and best.value >= max_gain - 1e-9:
            why = "residual fully solved"
        else:
            why = None
        if why:
            break
        trials = [study.ask() for _ in range(min(B, NT - len(fin)))]
        pp = [{k: tr.suggest_float(k, lo, hi, log=True) for k, (lo, hi) in BOUNDS.items()} for tr in trials]
        first = [1] if prune else list(range(1, LF + 1))
        o1 = run_outcomes(pool, pp, R, first)
        live = []
        for i, tr in enumerate(trials):
            if prune:
                pr1 = {s: prob(o1[i][s]) for s in R}
                tr.report(gain(pr1), 0)
                if tr.should_prune():
                    study.tell(tr, state=TrialState.PRUNED)
                    log(round=rnd, trial=tr.number, params=pp[i], state="pruned", gain_future1=gain(pr1))
                    print(f"  trial {tr.number:3d} PRUNED after future 1: gain {gain(pr1):.4f}  "
                          f"solves {sum(pr1[s] >= TAU for s in R):3d}/{len(R)}", flush=True)
                    continue
            live.append(i)
        rest = run_outcomes(pool, [pp[i] for i in live], R, list(range(2, LF + 1))) if prune and live else None
        best_v = best.value if best is not None else -math.inf
        for j, i in enumerate(live):
            pr = {s: prob(o1[i][s] + (rest[j][s] if rest else [])) for s in R}
            trials[i].set_user_attr("probs", pr)
            study.tell(trials[i], gain(pr))
            is_best = gain(pr) > best_v
            log(round=rnd, trial=trials[i].number, params=pp[i], state="complete", gain=gain(pr),
                solves=sum(pr[s] >= TAU for s in R))
            note = "   (config.yaml values)" if rnd == 0 and trials[i].number == 0 else ""
            print(f"  trial {trials[i].number:3d} {describe(pr)}{'  NEW BEST' if is_best else ''}{note}", flush=True)
            best_v = max(best_v, gain(pr))
        fin = finished()
        best, stall = best_and_stall(fin)
        n_pruned = sum(t.state == TrialState.PRUNED for t in fin)
        print(f"  -> {len(fin)}/{NT} trials ({n_pruned} pruned), "
              + (f"best gain {best.value:.4f} (trial {best.number}), {stall} since last improvement, " if best else "")
              + f"max possible {max_gain:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    print(f"  round {rnd} search finished: {why}", flush=True)
    best = study.best_trial
    return {k: float(best.params[k]) for k in NAMES}, dict(best.user_attrs["probs"]), float(best.value)


def do_round(state, pool):
    rnd = state["round"]
    t0 = time.time()
    print("\n" + "=" * 100)
    print(f"ROUND {rnd}   library {len(state['sets'])} sets   pool {len(state['pool'])}   "
          f"reserve {len(state['reserve'])}   unsolvable {len(state['dropped'])}")
    print("=" * 100, flush=True)

    cur = state["current"]
    if cur is None or cur["round"] != rnd:
        R = top_up(state, pool)
        if len(R) < int(LIB["min_residual"]):
            state["stopped"] = f"only {len(R)} unsolved scenarios left and the reserve is used up"
            return False
        cur = state["current"] = {"round": rnd, "residual": R, "base": {s: best_lib(state, s) for s in R}}
        save(state, STATE)
    R, base = cur["residual"], cur["base"]
    print(f"Residual: {len(R)} scenarios no library set solves "
          f"(of them {sum(b > 0 for b in base.values())} partly, p>0)", flush=True)

    theta, pr_R, g = run_optuna(state, pool, R, base, rnd, t0)

    # whole pool + validation
    rest = [s for s in state["pool"] if s not in pr_R]
    train = dict(pr_R)
    train.update(success_probs_many(pool, [theta], rest, LF)[0])
    val = run_outcomes(pool, [theta], state["val"], list(range(1, VF + 1)))[0]
    old_vals = [s["val"] for s in state["sets"]]
    v_old, v_new = val_score(old_vals, state["val"]), val_score(old_vals + [val], state["val"])
    keep = not state["sets"] or v_new - v_old >= float(LIB["min_val_gain"])

    solved_R = [s for s in R if pr_R[s] >= TAU]
    for s in R:
        if pr_R[s] < TAU:
            state["fails"][s] = state["fails"].get(s, 0) + 1
            if state["fails"][s] >= int(LIB["drop_after_rounds"]) and s not in state["dropped"]:
                state["dropped"].append(s)
    entry = {"round": rnd, "params": theta, "train": train, "val": val, "residual": len(R),
             "residual_solved": len(solved_R), "residual_gain": g,
             "train_alone": float(np.mean([train[s] >= TAU for s in state["pool"]])),
             "val_alone": float(np.mean([prob(v) for v in val.values()])),
             "val_gain": v_new - v_old, "val_score_after": v_new if keep else v_old}
    log(round=rnd, result="kept" if keep else "rejected", params=theta, val_gain=v_new - v_old,
        val_score=v_new, residual_solved=len(solved_R), residual=len(R))
    if keep:
        state["sets"].append(entry)
        state["rejects_in_row"] = 0
    else:
        state["rejected"].append({k: v for k, v in entry.items() if k not in ("train", "val")})
        state["rejects_in_row"] += 1

    print("-" * 100)
    print(f"Round {rnd}: best set solves {len(solved_R)}/{len(R)} residual scenarios (gain {g:.4f}); "
          f"alone it solves {entry['train_alone']:.1%} of the pool, mean validation success {entry['val_alone']:.3f}")
    print(f"Validation success with selection: {v_old:.4f} -> {v_new:.4f} ({v_new - v_old:+.4f})  -> "
          f"{'KEPT as set ' + str(len(state['sets']) - 1) if keep else 'REJECTED (gain < ' + str(LIB['min_val_gain']) + ')'}")
    covered = sum(best_lib(state, s) >= TAU for s in state["pool"])
    print(f"Pool coverage now {covered}/{len(state['pool'])}   unsolvable so far {len(state['dropped'])}   "
          f"[{time.time() - t0:.0f}s]", flush=True)

    state["round"] += 1
    state["current"] = None
    save(state, STATE)
    write_library(state)
    return True


# =============================================================================
# Report
# =============================================================================

def report(state):
    sets, val_ids, pool_ids = state["sets"], state["val"], state["pool"]
    print("\n" + "=" * 100)
    print(f"LIBRARY: {len(sets)} sets   (stopped: {state['stopped'] or 'not finished'})")
    print(f"pool {len(pool_ids)} scenarios, validation {len(val_ids)}, unsolvable set aside {len(state['dropped'])}, "
          f"rejected rounds {len(state['rejected'])}")
    print("=" * 100)
    if not sets:
        return
    print(f"{'set':>3s} {'round':>5s} {'pool solved alone':>18s} {'val mean p alone':>17s} {'val gain':>9s} "
          f"{'val w/ selection':>17s} {'only this set solves':>21s}")
    P = np.array([[s["train"][sid] for sid in pool_ids] for s in sets])
    solved = P >= TAU
    for k, s in enumerate(sets):
        only = int(np.sum(solved[k] & (solved.sum(0) == 1)))
        print(f"{k:3d} {s['round']:5d} {s['train_alone']:18.1%} {s['val_alone']:17.3f} {s['val_gain']:+9.4f} "
              f"{s['val_score_after']:17.4f} {only:21d}")
    vals = [s["val"] for s in sets]
    print(f"\nValidation: best single set {max(s['val_alone'] for s in sets):.3f}, with per-scenario selection "
          f"{val_score(vals, val_ids):.3f} (cross-fitted), some set solves {val_any(vals, val_ids):.1%}")
    print(f"Pool: some set solves {np.mean(solved.any(0)):.1%}")

    V = np.array([[prob(s["val"][sid]) for sid in val_ids] for s in sets])
    M = np.hstack([P, V])
    print("\nOutcome correlation between sets (success prob per scenario, pool + validation) -- low = complementary:")
    print("     " + "".join(f"{k:>7d}" for k in range(len(sets))))
    for a in range(len(sets)):
        row = []
        for b in range(len(sets)):
            if M[a].std() == 0 or M[b].std() == 0:
                row.append("    n/a")
            else:
                row.append(f"{np.corrcoef(M[a], M[b])[0, 1]:7.2f}")
        print(f"{a:5d}" + "".join(row))
    print("\nParameters (log10):")
    print("     " + "".join(f"{n[:10]:>11s}" for n in NAMES))
    for k, s in enumerate(sets):
        print(f"{k:5d}" + "".join(f"{math.log10(s['params'][n]):11.3f}" for n in NAMES))


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="only print the summary of the current library")
    ap.add_argument("--restart", action="store_true", help="move old results away and start over")
    args = ap.parse_args()

    if args.restart:
        old = BASE_DIR / f"library_old_{time.strftime('%Y%m%d_%H%M%S')}"
        old.mkdir()
        for f in (STATE, DB, LOG, OUT):
            if f.exists():
                shutil.move(str(f), old / f.name)
        print(f"Old library results moved to {old}")

    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state.get("futures") != LF or state.get("val_futures") != VF or state.get("tau") != TAU \
                or state.get("end_x") != END_X or state.get("road_length_m") != ROAD_LENGTH:
            raise SystemExit("library_state.json was made with other futures / threshold / end_x / road length "
                             "-- use --restart")
        resumed = True
    else:
        train_ids = existing(seed_range(PIPE["seeds"]["library_train"]))
        val_ids = existing(seed_range(PIPE["seeds"]["library_val"]))
        if len(train_ids) < int(LIB["initial_pool"]) or not val_ids:
            raise SystemExit(f"Only {len(train_ids)} library_train / {len(val_ids)} library_val scenarios -- run step 1 first")
        state = new_state(train_ids, val_ids)
        resumed = False

    if args.report:
        report(state)
        return

    pool, n_jobs = make_pool(args.n_jobs)
    print("=" * 100)
    print("Step 2: parameter library (sequential covering, Optuna)" + ("   -- RESUMED" if resumed else ""))
    print(f"Segments: takeover (x~{CONFIG['ego']['convert_x_m']} m) -> x={END_X:g} m, limit {PIPE['max_time_s']} s; "
          f"a set solves a scenario if success >= {TAU} over {LF} futures")
    print(f"Scenarios: pool {len(state['pool'])} + reserve {len(state['reserve'])}, validation {len(state['val'])} "
          f"({VF} futures)")
    print(f"Optuna per round: TPE (multivariate, batches of {OPT.get('batch', 4)}), <= {OPT['n_trials']} trials "
          f"({OPT.get('n_startup_trials', 16)} random first), patience {OPT.get('patience')}, "
          f"pruning {'on (future 1 first)' if OPT.get('prune', True) else 'off'};  "
          f"a full trial = residual x {LF} runs;  {n_jobs} parallel")
    print(f"Library: <= {LIB['max_sets']} sets, keep if validation gain >= {LIB['min_val_gain']}, "
          f"<= {LIB['max_rounds']} rounds, stop after {LIB['max_rejects']} rejections in a row")
    print("=" * 100, flush=True)
    if not resumed:
        save(state, STATE)

    try:
        while state["stopped"] is None:
            if len(state["sets"]) >= int(LIB["max_sets"]):
                state["stopped"] = f"max_sets ({LIB['max_sets']}) reached"
            elif state["round"] >= int(LIB["max_rounds"]):
                state["stopped"] = f"max_rounds ({LIB['max_rounds']}) reached"
            elif state["rejects_in_row"] >= int(LIB["max_rejects"]):
                state["stopped"] = f"{state['rejects_in_row']} rejected rounds in a row"
            else:
                do_round(state, pool)
        save(state, STATE)
        write_library(state)
    except KeyboardInterrupt:
        print("\nInterrupted -- progress is checkpointed; rerun the same command to continue.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    print(f"\nStopped: {state['stopped']}")
    report(state)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
