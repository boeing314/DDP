# v14/adaptive: adaptive parameters on top of the trained base

Overview and order of runs: ../README.md

Same policy as v13 (policy.py): every 0.1 s, 8 traffic features rescale lambda_obs, alpha, tau_y_base and
delta_lane, p = p0 * exp(clip(W f + b)), smoothed with T = 1 s. Here p0 = `../base/base_best.json`, so
W = 0 is exactly the trained fixed set, and any gain comes from adapting. Scenarios: `../base/scenarios`.

```bash
python step2_train_policy.py --n-jobs 7     # ~7 h, resumable; --finish stops early
python step3_evaluate.py --n-jobs 7         # 300 fresh scenarios: trained base, adaptive, bias part, base start
python step4_analyze.py --n-jobs 7          # figures/
python dipf_controller.py --episode ../base/scenarios/s05003 --distance 300 --max-time 60 --policy policy.json --gui
```

Note for `--gui`: the controller CLI uses config.yaml's dipf values as fixed parameters; pass the trained base with
`DIPF_PARAMS="$(python -c 'import json;print(json.dumps(json.load(open("../base/base_best.json"))["params"]))')"`.
