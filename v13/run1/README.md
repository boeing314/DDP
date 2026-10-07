# v13: adaptive DIPF parameters (learned gain scheduling)

The DIPF potential still produces every acceleration (a = -grad U, then the physical limits).
What changes: four of its parameters are no longer constants. Every control step (0.1 s) a
small learned policy reads the traffic around the ego and rescales them:

```
f_t  = 9 dynamics features in [0, 1]          (policy.py: time-to-collision, required braking, gaps, lateral room, speed, density)
z*_t = clip(W f_t + b, -ln10, ln10)           one exponent per adapted parameter
z_t  = z_{t-1} + (dt/T)(z*_t - z_{t-1})       smoothed, T = 1 s
p_t  = p0 * exp(z_t)                          lambda_obs, tau_y_base, alpha, obs_radius_ahead
```

W = 0, b = 0 is exactly the fixed controller with the base parameters p0 (v12 library set 2).
W and b (40 numbers) are trained with CMA-ES on closed-loop segment success, with no labels.

Road 500 m, segment: the car nearest x = 100 m at t = 30 s becomes the ego and must reach x = 400 m
within 60 s. Same scenarios as v11/v12 for the same seeds.

## Run

```bash
./run_pipeline.sh 7
# or step by step (every step resumes where it stopped):
python step1_make_scenarios.py --n-jobs 7          # 500 scenarios (or copy v12/run1/scenarios -- same road, same seeds)
python step2_train_policy.py   --n-jobs 7          # ~8 h; Ctrl+C and rerun to continue; --finish to stop early
python step2_train_policy.py   --bias-only --generations 15 --n-jobs 7   # control experiment, ~2 h
python step3_evaluate.py       --n-jobs 7          # fresh test scenarios: base vs adaptive vs controls
python step4_analyze.py        --n-jobs 7          # figures/
# watch a test scenario with the trained policy:
python dipf_controller.py --episode scenarios/s05003 --distance 300 --max-time 60 --policy policy.json --gui
```

All settings: `config.yaml -> pipeline` (`adaptive` = the policy, `cmaes` = the search).

## Files

| file | role |
|---|---|
| `policy.py` | features, the parameter mapping, packing of W/b for the optimiser |
| `dipf_controller.py` | `adapt_parameters()` runs before each control step when a policy is set |
| `common.py` | scenarios, worker pool, closed-loop evaluation, CMA-ES (NumPy) |
| `step2_train_policy.py` | CMA-ES training with validation checkpoints -> `policy.json` |
| `step3_evaluate.py` | test: base, adaptive, its bias part, trained bias-only control; McNemar tests |
| `step4_analyze.py` | training curve, weight map, parameter traces, multiplier distributions |

## Repeatable runs (important, also for v11/v12)

SUMO orders some internal work by memory address. With the default address-space layout
randomisation (ASLR), the *same* run (state, seed, commands) could draw its random numbers in a
different order on each launch, and an identical segment run could end in success once and
in a collision the next time. v13 starts SUMO with ASLR disabled (`setarch -R` equivalent, in
`_kill_with_parent`). Every run is now exactly repeatable, so all candidates really meet the
same traffic, and W = 0 reproduces the fixed controller bit for bit.

## Reading the results

* Step 3 `adaptive vs base` is the total gain. `adaptive vs bias part` is the part that comes from
  adapting during the run. `bias part vs base` is the part that only comes from better fixed
  values. With the trained bias-only control, `adaptive vs bias-only` is the clean test of adaptation.
* Step 2 selects the final policy on validation. If no checkpoint beats the base, `policy.json` is the
  base (W = 0) and the run says so.
* `fig_weights.png`: each cell is the log-multiplier per unit feature. For example, +0.7 for
  (lambda_obs, drac_front) means obstacle repulsion is about doubled when hard braking is needed.
