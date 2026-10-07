# v7 — starting-condition capture

Runs the v6 controller with v6's **optimised parameters** over a range of seeds and saves the full
simulation state at four moments of each episode:

| Capture | Time |
|---|---|
| `start` | ego's first control step |
| `between_1`, `between_2` | equally spaced between `start` and the last capture |
| `end_minus_10s` | 10 s before the ego finishes or crashes |

## Run

```bash
python generate_sumo.py            # only if simulation/ego/vehicle_types/flows changed
python capture_states.py           # seeds from config.yaml -> capture (default 0..49)
python capture_states.py --seeds 0 9 --n-jobs 4
```

Settings (`config.yaml` → `capture:`): `v6_dir`, `seed_start`/`seed_end`, `lead_time_s` (10),
`n_between` (2), `output_dir` (`states`), `n_jobs`.

## Where the parameters come from

From `capture.v6_dir`, the first of these that exists:

1. `best_params.json`, from a finished `optimize.py` run.
2. `optimization_log.jsonl`: the lowest-cost accepted evaluation. That's the current population
   best of an unfinished run, the same member as `de_checkpoint.npz`'s best.
3. v6 `config.yaml`: its `dipf.*` values, e.g. after `optimize.py --apply`.

Only the tuned `dipf.*` keys are taken from v6; everything else comes from this folder's
`config.yaml`. The script warns if the two configs differ in anything that affects an episode.

## How it works

For each seed:

1. **Pass 1:** run the episode and record `t_start` and `t_end`.
2. **Pick the capture times:** `t_last = t_end − lead_time`, plus `n_between` times equally spaced
   between `t_start` and `t_last`, all snapped to the 0.1 s step grid.
3. **Pass 2:** rerun the same seed (SUMO and the controller are deterministic) with
   `--capture-times …`. At each time, *before* the controller acts that step, it writes:
   - `state_t<time>.xml`: the full SUMO state (`traci.simulation.saveState`) with RNG state
     (`--save-state.rng`) and 6-decimal precision. All vehicles, routes, flows.
   - `ego_t<time>.json`: the controller-held ego state that the SUMO state file does **not**
     contain: `vx`, `vy`, `prev_ax`/`prev_ay` (jerk limiter), `heading_deg`. Also the ego's centre
     and front positions, lane, and the `dipf` parameters used.
4. **Check:** pass 2 must end with the same status at the same time as pass 1. Otherwise the seed
   is flagged `deterministic: false`.

If an episode is shorter than `lead_time_s` (e.g. a crash 8 s after the start), only `start` is
captured and the seed is flagged `short_episode`.

## Output

```
states/
  params_used.json        dipf values used + their source + v6/v7 config differences
  index.csv               one row per seed: outcome, t_start, t_end, capture labels/times/files
  seed_<s>/info.json      the same, per seed
  seed_<s>/state_t*.xml   SUMO states
  seed_<s>/ego_t*.json    controller states
```

Each state uses about 30 KB.

## Reloading a state

```bash
sumo -n straight3lane.net.xml --load-state states/seed_0/state_t58.6.xml --lateral-resolution 0.25
```

Leave out `-r`: the state file contains the vehicle types and the flow state, so traffic keeps
arriving as in the original run. To continue the *ego* from a capture, the controller's `vx`, `vy`,
`prev_ax`, `prev_ay` and heading must be restored from the matching `ego_t*.json`.

---

# Episode dataset and per-episode optimization

## 1. Build the episode folder

```bash
python build_episodes.py
```

Each captured state is one **episode**, so a seed with 4 captures gives 4 episodes. They all go into
`episodes/<episode_id>/` (`state.xml`, `ego.json`, `meta.json`), plus one `episodes/index.csv`.
`episode_id = s<seed>_c<k>_<label>`, e.g. `s0042_c2_between_2`.

`meta.json` holds:
- the source: seed, capture label, time, and the original outcome
- the ego's start (`start_x`, `start_vx`, …)
- the segment to drive: `target_x = min(start_x + 150, 985)`, plus `segment_capped` and `usable`.
  An episode is usable if at least `min_distance_m` of road is left.

## 2. Episode mode of the controller

```bash
python dipf_controller.py --episode episodes/s0042_c2_between_2 --distance 150 --max-time 60
```

- **Where it starts:** loads the saved SUMO state (without the route file) and restores the
  controller's ego state from `ego.json` (`vx`, `vy`, last accelerations, heading).
- **Timing:** acts first, then advances SUMO, because the capture was taken before the controller
  acted at that step.
- **Success:** the ego's centre reaches `start_x + distance`, i.e. the target or beyond. It can also
  end by collision, or by timeout after `--max-time` simulated seconds.
- **The cost C is computed over the segment:** progress and the time term are relative to the
  segment length.

**Reproducibility:** reloading an episode gives *identical* results every time. A reload does
**not** exactly continue the original run, though: SUMO's state reload doesn't restore the drivers'
random speed fluctuations bit-for-bit, even with `--save-state.rng`. So an episode's outcome can
differ from `original_outcome`, which is why the optimizer re-scores the v6 baseline on every
episode.

## 3. Per-episode differential evolution

```bash
python optimize_episodes.py --limit 5 --verbose    # quick look first
python optimize_episodes.py                        # all usable episodes
python optimize_episodes.py --summarize            # rebuild summary.csv only
```

Settings (`config.yaml` → `episode_opt`): `distance_m`, `max_time_s`, `bounds`, `pop_size`,
`n_generations`, `F`, `CR`, `patience`, `include_baseline`, `n_jobs`. Most can also be overridden
on the command line (`--pop-size`, `--n-generations`, `--patience`, `--n-jobs`, `--match`, `--limit`).

- **Algorithm:** per episode, the v4/v6 DE runs unchanged: log10 space, Latin hypercube start,
  DE/best/2, exponential crossover, greedy selection. Population member 0 is the v6 best parameters,
  so each episode's result is never worse than v6.
- **One evaluation is one simulation of the episode.** Episodes are deterministic, so one run
  scores a candidate.
- **Parallel:** one episode per worker.
- **Resumable:** episodes that already have `<episode_id>.json` are skipped; use `--rerun` to
  redo them.

What it prints:
- a header with settings, counts and the evaluation budget
- for each episode: its start line, the baseline result, and one line per generation
  (best C / outcome / progress / accepted, elapsed time)
- a `DONE` line per episode, overall progress and ETA after every episode, and a final summary

Add `--verbose` to also print every candidate.

Output: `episode_opt_results/<episode_id>.json` holds the baseline and best parameters with their
results and the per-generation history. `episode_opt_results/summary.csv` has one row per episode.

**Speed.** Two changes make an evaluation about 11× faster (4.8 s → 0.42 s on its own) without changing
any result. Trajectories were verified byte-identical against the original code, for a normal seed run
and for episodes:

1. **The obstacle list uses one TraCI context subscription on the ego.** SUMO sends
   position/angle/speed/lateral speed of all vehicles near the ego in one message per step. The old
   code made ~7 separate TraCI calls for *every* vehicle on the road every step. Each vehicle's type
   and sizes are looked up once.
2. **Evaluations run in-process** (`dipf_controller.run_episode`). Each optimizer worker imports the
   controller once, and every evaluation starts a **fresh** SUMO with a 10 ms connection poll instead
   of `traci.start`'s fixed 1 s wait. Keeping one SUMO alive and switching episodes with
   `traci.simulation.loadState()` was tested and rejected: results then depend on earlier episodes.
   `--subprocess` restores the old one-process-per-evaluation path, which gives identical results
   but is slower.

**Budget:** each episode runs `pop_size × (1 + n_generations)` evaluations. Measured throughput with
11 episodes in parallel is **0.057 s per evaluation**:

| Setting | Evaluations per episode | All 1,618 episodes |
|---|---|---|
| pop 10, 10 generations (default) | 110 | **~2.8 h** |
| pop 10, 5 generations | 60 | ~1.5 h |
| pop 8, 6 generations | 56 | ~1.4 h |
| pop 6, 5 generations | 36 | ~0.9 h |

Candidates that run to the 60 s timeout take longer, so allow some margin (about 3–4 h for the default).

---

# Learning to choose parameters

## Regression MLP (`train_param_mlp.py`)
Predicts the 8 parameters directly from 28 features: ego state plus the 6 nearest vehicles by
skewed distance (`make_param_dataset.py --with-id`). On the test episodes R² ≈ 0: each episode's
DE optimum is one of many working sets, so the labels are not learnable. Kept as a baseline.

## Parameter-set classifier
1. `python build_param_library.py` builds the candidates: v6 plus the medoids of 24 k-means
   clusters of the per-episode best sets. It simulates every candidate on every episode and writes
   `param_library/eval_matrix.csv` (success and C). This takes about 40 min (1,618 × 25 runs at
   ~0.06 s each with 11 parallel). It's resumable, and `--n-clusters 16` makes it about 30% faster.
2. `python train_param_classifier.py` trains the classifier:
   - **Library:** chosen greedily on the training episodes, v6 first, then the set that solves the
     most still-unsolved episodes, up to `--k` sets (default 8).
   - **Model:** MLP 28 → 64 → 64 → K with sigmoid outputs, giving the probability that each set
     succeeds in this situation. It picks the most likely set.
   - **Evaluation:** test success is looked up in the matrix (exact) and compared with v6, the
     best single set, a random library set, and the oracle.
3. `python mlp_dipf_controller.py --model param_classifier.npz --seed 42` uses the classifier in
   the controller. Each step it picks a set; `--margin 0.1` makes switching sticky and
   `--update-every N` re-predicts less often. `--eval-test` runs it on the classifier's test
   episodes.
