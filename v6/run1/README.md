# v6 — v5 controller + differential evolution tuning (v4 algorithm)

The controller (`dipf_controller.py`) is the v5 single-run Pure-DIPF controller.
v6 adds `optimize.py`, the v4 Differential Evolution driver, set up for the v5 model.
`config.yaml` is the single source of truth for every parameter, including the search.

## Files

- `config.yaml` — simulation, road, ego, traffic, DIPF parameters, cost, and the `optimization:` section.
- `generate_sumo.py` — generates the SUMO network, routes and `.sumocfg` from `config.yaml`.
- `dipf_controller.py` — runs one episode.
- `optimize.py` — differential evolution over the `dipf.*` parameters listed in `optimization.bounds`.

## Run one episode

```bash
pip install -r requirements.txt
python generate_sumo.py                  # after changing simulation/ego/vehicle_types/flows
python dipf_controller.py                # seed = simulation.seed
python dipf_controller.py --seed 7 --gui
```

## Optimize

```bash
python optimize.py                       # all settings from config.yaml -> optimization
python optimize.py --n-generations 5     # override pop-size / n-generations / F / CR / n-jobs / rng-seed
python optimize.py --resume              # continue from de_checkpoint.npz
python optimize.py --apply               # also write the best values into config.yaml
```

The algorithm is the same as in v4:

1. Latin hypercube initial population in log10 space.
2. For each member: DE/best/2 mutation `v = best + F·(r1 − r2 + r3 − r4)`, clipped to the bounds,
   then exponential crossover with probability `CR`.
3. Score the trial vector as its average episode cost `C` over the fixed `train_seeds`.
4. Greedy selection: the trial replaces its parent if it has lower `C`.
5. Checkpoint after every generation.

v6 additions:

- **Parameters and ranges come from `optimization.bounds`:** add or remove any `dipf.*` key there.
- **The best member is checked on unseen seeds:** it is re-scored on `optimization.validation_seeds`
  (both `C` and success rate), because small seed lists are easy to overfit.
- **`--apply`** writes the best values into `config.yaml` at full precision and keeps the old value
  as a comment. Results are sensitive to tiny changes, so don't round them.
- **Candidate values never edit `config.yaml`.** Each run receives them as JSON in the `DIPF_PARAMS`
  env var, and `DIPF_OUTPUT_DIR` sends its results to a scratch folder.

Outputs: `optimization_log.jsonl` (every evaluation), `de_checkpoint.npz`, `best_params.json`.

Cost of a run ≈ `len(train_seeds) × pop_size × (1 + n_generations)` episodes. The defaults are
24 × 15 × 21 ≈ 7,600 episodes, about 30–60 min on 11 cores.
