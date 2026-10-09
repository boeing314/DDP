# v14/longroad: base vs adaptive on a ~1000 m drive (evaluation only)

The trained base (`../base/base_best.json`) and adaptive policy (`../adaptive/policy.json`) were trained on
the 500 m road (segment x ~100 -> 400 m). Here they drive ~1000 m without retraining: road 1250 m, takeover
of the car nearest x = 100 m at t = 30 s, finish at x = 1100 m, limit 200 s.

```bash
python step1_make_scenarios.py --n-jobs 7      # once: 300 test scenarios on the 1250 m road (~1 min; already done)
python evaluate_longroad.py --n-jobs 7         # ~35 min: 3 policies x 300 scenarios x 3 futures
python evaluate_longroad.py --n-jobs 7 --n-scenarios 100 --futures 2   # quicker (~8 min)
```

Output: success over ~1000 m with paired McNemar tests, and the fraction of runs that covered >= 100 ... 900 m
without a collision (`longroad_summary.json`, `longroad_profile.png`). The 300 m row is comparable with the
500 m road results; the ratio column shows whether the adaptive advantage grows or shrinks with distance.
