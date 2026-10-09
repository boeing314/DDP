# v14/base_delta_v_new: best fixed set with an approach-speed numerator

Identical to `../base` except for the obstacle potential:

    ../base               : U_obs = lambda / k'
    ../base_delta_v       : U_obs = lambda * (|dv| / v_max) / k'
    base_delta_v_new      : U_obs = lambda * (1 + max(dv_approach / v_max, 0)) / k'

    dv_approach = -(dx*dvx + dy*dvy) / |r|   rate at which the centre distance shrinks
                                             (> 0 approaching, < 0 moving apart), v_max = 34 m/s

So every vehicle repels at least as in ../base (numerator >= 1), and approaching vehicles repel more
(x2 at an approach speed of 34 m/s); vehicles moving apart are treated like ../base.
(`config.yaml -> dipf.approach_numerator: true`, `dipf.dv_max: 34.0`.)

Everything else is the same as ../base: scenarios (`../base/scenarios`), seeds, futures, objective, CMA-ES over all
7 parameters from v12 set 2, budget (15 x 30), validation-based selection, test set.

```bash
python step2_train_base.py --n-jobs 7       # ~6 h, resumable; --finish stops early
python step3_evaluate.py --n-jobs 7         # 300 fresh scenarios: start set vs trained set
```
