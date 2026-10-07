# v14/base_delta_v: best fixed set with the relative-speed numerator

Identical to `../base` except for one term of the obstacle potential:

    ../base          : U_obs = lambda / k'
    base_delta_v     : U_obs = lambda * (|dv| / v_max) / k'      |dv| = |v_obstacle - v_ego| (relative speed)
                                                                   v_max = 34 m/s (the max speed, constant)

(`config.yaml -> dipf.use_dv_numerator: true`, `dipf.dv_max: 34.0`). Everything else is the same: scenarios
(`../base/scenarios`), seeds, futures, objective, CMA-ES over all 7 parameters from the same start (v12 set 2),
budget (15 x 30), validation-based selection, test set. So step 3 here is directly comparable with `../base`.

Note: with the numerator, a vehicle moving at exactly the ego's velocity exerts no repulsion, however close;
the search can compensate only through lambda and the other parameters.

```bash
python step2_train_base.py --n-jobs 7       # ~6 h, resumable; --finish stops early
python step3_evaluate.py --n-jobs 7         # 300 fresh scenarios: start set vs trained set (both with the numerator)
```

Step 1 is not needed: the scenarios are `../base/scenarios`.
