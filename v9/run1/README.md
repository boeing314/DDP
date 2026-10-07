# v9: baseline (one fixed parameter set), optimized with DE

The v8 controller (pure DIPF: no safety filter), with:
- **Encoding** (`encoding.py`): 80 laneless, type-aware values describing the surrounding traffic. It covers
  8 sectors relative to the ego's body and path (front / front-left / front-right / left / right / rear /
  rear-left / rear-right). For the most critical vehicle in each sector it records presence, longitudinal and
  lateral box gaps, relative velocity, inverse TTC, required braking, and length and width (which stand in for
  the vehicle type). It also has 8 global values: ego velocity, heading, lateral position, edge clearances,
  density, and the mean relative speed ahead. `python dipf_controller.py --seed 7 --log-encoding` writes it for
  every step to `encoding.csv`, for future models. It isn't used for control.
- **`tau_x_base` fixed at 1.0.**
- **New search ranges** (`optimization.bounds`, 7 parameters), widened where earlier optima sat at a limit:
  `sx`, `delta_lane`, `y_margin`, `obs_radius_ahead`.

## Optimize (DE, v4 algorithm)
```bash
python optimize_de.py --n-jobs 7              # 60 train seeds x pop 12 x 16 generations
python optimize_de.py --resume                # after an interruption (checkpoint per generation)
python optimize_de.py --train-seeds 0 119 --n-generations 10      # more seeds, fewer generations
python optimize_de.py --resume --n-generations 10 --apply         # stop at gen 10, validate, write config.yaml
```
The objective is `J = (1 − success) + 0.2·(1 − progress)` over full-road runs on the training seeds. Member 0 of
the starting population is the current `config.yaml`. At the end the best set is scored on 200 validation seeds
next to the starting values. The script writes `de_log.jsonl`, `de_checkpoint.npz` and `de_best.json`. Ctrl+C
stops cleanly; the last finished generation is kept.

## Evaluate on fresh seeds (paired, with an exact McNemar test)
```bash
python evaluate.py --sets best config v6_old --seeds 1000 1499
```
