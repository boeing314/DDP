# v14: adaptive DIPF parameters vs a properly trained fixed baseline

v13 showed the adaptive policy beating its base (v12 library set 2) by +16 points, but that base was a
specialist from the v12 library, never tuned as an all-round fixed set. v14 makes the comparison fair:

| folder | what it trains | output |
|---|---|---|
| `base/` | the best **fixed** set: CMA-ES over all 7 tuned parameters, full log ranges, start = v12 set 2 | `base_best.json` |
| `adaptive/` | the v13 adaptive policy with **p0 = the trained base** (W = 0 reproduces it exactly) | `policy.json` |

Same for both: scenarios (made once in `base/scenarios`, read by `adaptive/`), train/val/test seeds,
futures, objective J = success + 0.1 progress, search budget (popsize 15 x 32 generations = 480
candidates), validation-based selection, test on 300 fresh scenarios x 3 futures, repeatable runs
(SUMO without ASLR). The obstacle-radius floor is 10 m in both (it was pinned at 30 m in v13).

Because the adaptive policy starts from the trained base, `adaptive vs base` in `adaptive/step3_evaluate.py`
measures only what adapting adds on top of the best fixed set the same search could find.

## Run (in this order)

```bash
cd v14/base     && ./run_pipeline.sh 7     # ~6 h   (step 1 can be skipped: cp -r ../../v13/run1/scenarios .)
cd ../adaptive  && ./run_pipeline.sh 7     # ~7.5 h
```

Retraining the base invalidates the adaptive run (its p0 changes). `adaptive/step2_train_policy.py`
detects this and asks for `--restart`.
