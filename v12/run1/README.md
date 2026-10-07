# v12: parameter library + multi-label selector (Optuna search)

v12 = v11 with one change: each library round is searched with **Optuna** (TPE sampler,
multivariate, log-uniform ranges, median pruning) instead of differential evolution.
Everything else (scenarios, coverage-gain objective, top-up, unsolvable scenarios, validation
gate, labels, model, fallback, tests) is identical, so v11 and v12 libraries can be compared
directly. Road: 500 m (`simulation.road_length_m`).

Goal: a small library of DIPF parameter sets that solve *different* traffic situations,
and a model that predicts, from the traffic around the ego at the takeover, the
probability that each set gets through the segment. The probabilities are independent
(they need not sum to 1). The ego then drives the segment with the most promising set.

Segment: the car nearest x = 100 m at t = 30 s becomes the ego (it keeps its speed) and must
reach x = 400 m (`pipeline.end_x_m`) without a collision within 60 s.

## Run

```bash
./run_pipeline.sh 7          # all steps, 7 parallel simulations
# or step by step (each one resumes where it stopped):
python step1_make_scenarios.py --n-jobs 7
python step2_build_library.py  --n-jobs 7      # longest step; Ctrl+C and rerun to continue (trials are in library_optuna.db)
python step2_build_library.py  --report        # current library, coverage, outcome correlations
python step3_label_dataset.py  --n-jobs 7
python step4_train_model.py
python step5_evaluate_policy.py --n-jobs 7
python step5_evaluate_policy.py --show s05003  # predicted probs for one scenario + a sumo-gui command
```

Rough times with 7 jobs (about 1.1 s per segment run per worker): step 1 ~10 min,
step 2 4-7 h (each round: <= 80 trials x residual x 3 runs; a pruned trial costs 1/3,
rounds stop early after 30 trials without improvement),
step 3 ~3 h (3000 scenarios x K sets x 3 runs), step 4 minutes, step 5 ~20 min.

All settings are in `config.yaml -> pipeline`.

## Steps

| step | what it does | output |
|---|---|---|
| 1 | one saved SUMO state per seed at the takeover + the 80-value encoding (model input) | `scenarios/sNNNNN/` |
| 2 | builds the library by sequential covering, one Optuna study per round | `library.json`, `library_optuna.db` |
| 3 | runs every set on every dataset scenario | `dataset.csv` |
| 4 | trains the multi-label MLP ensemble | `model.npz`, `model_report.json` |
| 5 | closed-loop test on fresh scenarios | `test_results.csv`, `test_summary.json` |

## Optuna settings (`pipeline.library.optuna`)

| key | default | meaning |
|---|---|---|
| `n_trials` | 80 | trials per round (completed + pruned) |
| `n_startup_trials` | 16 | random trials before TPE starts modelling (round 0: trial 0 = config.yaml values) |
| `patience` | 30 | end the round after this many trials without a better gain |
| `batch` | 4 | trials evaluated together (constant liar), keeps all workers busy |
| `prune` | true | a trial runs future 1 first and is dropped if below the median of earlier trials |
| `prune_startup` | 10 | completed trials before pruning starts |

Inspect a round: `optuna.load_study(study_name="round0", storage="sqlite:///library_optuna.db")`.

## How v11/v12 differ from "remove solved seeds, re-optimise, repeat"

1. **Success probability, not one run.** Each scenario is a saved state, replayed with
   3 random futures (SUMO seeds 1..3). A set "works" if it succeeds in >= 2 of 3
   (`success_threshold` 0.66). One lucky run no longer counts as solved.
2. **Coverage-gain objective.** Round r's search maximises
   `mean over residual of max(0, p_new - p_best_in_library)`: partial successes count,
   so the search has something to climb even when nothing is fully solved yet, and it is never
   rewarded for re-solving what the library already handles.
3. **Your top-up rule.** When the residual drops below `min_pool` (60), reserve scenarios
   are added; every library set is run on them first and only scenarios none of them
   solves join the residual.
4. **Unsolvable scenarios are set aside.** A residual scenario that the round's best set
   still fails gets a strike; after 2 strikes it is dropped, so later rounds stop chasing
   situations no parameter set can handle (e.g. a car cutting in right at the takeover).
5. **Validation gate.** A new set is kept only if it raises the success on 100 held-out
   scenarios with per-scenario selection by >= 0.01. Selection is cross-fitted: the set
   is picked on futures 1-3 and scored on futures 4-6 (and vice versa), because picking
   and scoring on the same runs rewards noise. This keeps sets that generalise and
   stops sets that only fit their own residual.
6. **Complete labels.** After the library is fixed, every set is run on every dataset
   scenario. The model learns each set's success probability separately (sigmoid
   outputs, binary cross-entropy on the soft labels), so "several sets work here" is
   learned too.
7. **One decision per segment.** The set is chosen at the takeover and held to x = 400 m,
   which is exactly what the labels measure (earlier versions lost performance by
   switching sets mid-run).
8. **Fallback margin.** The ego leaves the best single set only if the model predicts the
   other set is better by more than a margin chosen on the validation split. If the model
   is unreliable, the margin becomes 1 (always the best single set), so the policy cannot
   do worse than the fixed set on validation.
9. **Input preprocessing + mirroring.** Gaps become closeness values (no jump between
   "far" and "empty sector"); the drac artefact in the left/right sectors is removed;
   mirrored left/right copies double the training data.
10. **Honest split.** Scenarios used to build the library only go to the training split.
    Validation, test and the step-5 seeds (5000-5299) are never seen by the search.

## Reading the results

* Step 2 `--report`: *only this set solves* = how many scenarios need that set; the
  outcome correlation matrix should show low values (complementary sets).
* Step 4/5: compare **model + fallback** with **best single set**. The **oracle** is an
  upper bound (the best set in hindsight). The gap between best single and oracle is all
  a selector can gain. The McNemar p-value in step 5 says whether the difference
  over individual runs is significant.
