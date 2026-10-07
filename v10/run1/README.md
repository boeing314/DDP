# v10: the v9 baseline, optimized with Optuna instead of DE

The controller (pure DIPF, no safety filter), encoding, search space (`config.yaml → optimization.bounds`, `tau_x_base`
fixed) and objective are identical to v9, so the two optimizers can be compared directly. See v9's README
for the controller.

```bash
pip install optuna                               # already installed in the project venv
python optimize_optuna.py --n-jobs 7             # up to 150 trials x 60 train seeds (less with pruning)
python optimize_optuna.py                        # rerun = continue the same study (optuna_v10.db)
python optimize_optuna.py --finish --apply       # no new trials: validate the best, write config.yaml
python evaluate.py --sets best config v6_old --seeds 1000 1499
```
- **Sampler:** TPE, multivariate, so it models how the parameters interact. Trial 0 is the current
  `config.yaml`, followed by 15 random startup trials. On resume the sampler's seed is offset, so random trials
  are never repeated.
- **Pruning:** the training seeds run in 4 chunks. After the first 10 trials, a trial clearly worse than the
  median at the same point is stopped early (MedianPruner). Use `--no-pruning` to turn this off.
- **Outputs:** `optuna_v10.db` (the study), `optuna_trials.csv` (one row per trial), and `optuna_best.json`
  (best parameters, training and validation results, fANOVA parameter importance).
