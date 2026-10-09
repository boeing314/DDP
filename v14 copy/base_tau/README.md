# v14/base_tau: same as v14/base, but with a different tau per k' region

With tau1 = `tau_x_base`, tau2 = `tau_y_base` (see `dipf_controller.k_prime`):

    k' = (1 - w/|Dx|) . sqrt( (Dx/tau(Dx,Dvx,tau2))^2 + (Dy/tau(Dy,Dvy,tau1))^2 )   if |Dy| <= l|Dx|/w
       = (1 - l/|Dy|) . sqrt( (Dx/tau(Dx,Dvx,tau1))^2 + (Dy/tau(Dy,Dvy,tau2))^2 )   otherwise

Everything else is identical to `../base`.

Overview and order of runs: ../README.md

CMA-ES over all 7 tuned DIPF parameters (`config.yaml -> pipeline.base_search.bounds`, log-uniform),
starting from v12 library set 2, scored on segment success (takeover at x ~ 100 m -> x = 400 m) over
100 train scenarios x 2 futures, 450 candidates, final set chosen on 100 validation scenarios x 3 futures.

```bash
python step1_make_scenarios.py --n-jobs 7   # or: cp -r ../../v13/run1/scenarios .   (same road, same seeds)
python step2_train_base.py --n-jobs 7       # ~6 h, resumable; --finish stops early
python step3_evaluate.py --n-jobs 7         # 300 fresh scenarios: v12 set 2 vs trained base
```

Output `base_best.json` is p0 for `../adaptive`.
