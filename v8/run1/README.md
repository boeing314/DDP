# v8: situation-cluster adaptive potential field

The v7 controller, plus parameters that depend on the traffic situation:
1. Cluster traffic situations.
2. Find one optimized DIPF parameter set per cluster.
3. While driving, use the parameter set of the cluster the current situation is closest to.

**`tau_x_base` is fixed at 1.0** (`dipf.tau_x_base`) and not tuned. The 7 tuned keys are in
`cluster_opt.bounds`.

## Pipeline

```bash
./run_pipeline.sh 7          # everything, 7 parallel simulations
```
Or step by step:

| Step | Command | Output | Time |
|---|---|---|---|
| 1 | `python make_situations.py` | `situations.csv`: slot features of every episode's start state | ~1 min |
| 2 | `python cluster_situations.py [--scan 4 28] [--k 16]` | `clusters.npz`, `cluster_assignment.csv`, `clusters_summary.csv` | seconds |
| 3 | `python optimize_clusters.py` | `cluster_opt_results/cluster_<j>.json`, `cluster_params.json` | ~3–5 h (resumable) |
| 4 | `python cluster_controller.py --eval-test` / `--eval-seeds 1000 1499` | `eval_test/`, `eval_seeds_*/summary.csv` | minutes |

Driving with it: `python cluster_controller.py --seed 1000 [--gui] [--update-every 10] [--hysteresis 0.1]`.
It writes `results/cluster_results_<seed>/params.csv`, the cluster and parameters at every step.

## Details
- **Situation features (`situation.py`, 21 of them):** for the vehicle ahead and behind in the
  own, left and right lane, the bumper gap and relative speed. A free slot has gap = 100 m; a lane
  that doesn't exist counts as a wall (gap = 0). Plus `vx`, `vy`, offset from the lane centre,
  clearance to each road edge, which side lanes exist, TTC to the leader, and the vehicle count
  within ±50 m. The same function is used offline and while driving, and live features were
  checked to match the offline ones exactly.
- **Split:** by seed, 70 / 15 / 15. Clusters are fitted on training episodes only.
- **Clusters:** k-means with K = 16 (`--scan` shows inertia and silhouette per K). Silhouette is
  only about 0.2, so situations form a continuum rather than crisp groups. Each cluster gets a
  readable description, e.g. "left lane; slower leader medium ahead (28 m, TTC 7 s); right
  tight".
- **DE per cluster:** the v4 algorithm. The objective is (1 − success) + 1e-4·C averaged over the
  cluster's training episodes, each simulated for `distance_m` (300 m) from its saved state.
  Population member 0 is v6 (with τx = 1). `futures > 1` averages over random futures (the state
  reloaded with different SUMO seeds) instead of one replayed future.
- **Fallbacks:** clusters with fewer than `min_train_episodes` training episodes use v6.
  **Validation gate:** a cluster's optimized set is used only if it's at least as good as v6 on
  that cluster's validation episodes. Small clusters overfit: in the smoke test, 81% vs 69% on
  16 training episodes became 57% vs 71% on validation.
- **Controller:** re-identifies the cluster every 10 steps (1 s). It switches only if the new
  centre is at least 10% closer (`--hysteresis`). Evaluation is paired against fixed v6
  parameters on the same seeds or episodes, with an exact McNemar p-value.
