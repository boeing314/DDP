# v5 — single-run Pure DIPF + SUMO

One SUMO episode driven by the Pure-DIPF controller. No optimization.
`config.yaml` is the single source of truth for every parameter.

## Files

- `config.yaml` — simulation, road, ego, traffic, vehicle-type, DIPF and cost parameters.
- `generate_sumo.py` — generates `straight3lane.{nod,edg,rou}.xml`, `.sumocfg` and `.net.xml` (via `netconvert`).
- `dipf_controller.py` — runs one episode.

## Run

```bash
pip install -r requirements.txt
python generate_sumo.py          # after any change to simulation/ego/vehicle_types/flows
python dipf_controller.py        # seed = simulation.seed in config.yaml
python dipf_controller.py --seed 7
python dipf_controller.py --seed 7 --gui
```

Output: `results/results_<seed>/trajectory.csv` and `metrics.json`.

## Model changes vs. v4

- **Obstacle potential numerator used as written:**
  `U_o = λ · (|Δv| / Δv_max) / k'`, with `|Δv| = hypot(Δv_x, Δv_y)` and `Δv_max = dipf.dv_max`.
  There is no closing-speed projection, no clamp and no zeroing for a constant or growing gap.
  It is still 0 when both vehicles have identical velocity, because |Δv| = 0.
- **Time-based collision measure:** each step logs `ttc`, the time until the ego's and the
  nearest obstacle's bodies would overlap if both kept their current velocities
  (0 = overlapping, inf = never). The cost uses the episode minimum:
  `C_collision = (e^(1 − t/T) − 1)/(e − 1)`, `t = min(min_ttc, T)`, `T = cost.ttc_threshold_s`.
  An actual collision is still detected by SUMO or by road departure.
- **Ego max velocity:** `ego.max_speed_mps` sets the controller clamp and the SUMO egoType `maxSpeed`.
  The egoType `speedFactor` is raised so the 20 m/s lane limit doesn't cap it. The ego departs at
  `ego.start_speed_mps` (34 m/s).
- **Lane width fix:** the edge now uses SUMO's `width` attribute, so lanes really are
  `lane_width_m` (v4's `laneWidth` was ignored and gave 3.2 m lanes).
- **Ego velocity is controller state:** `vx` and `vy` are integrated by the controller and carried
  from step to step, not read back from SUMO. `getLateralSpeed()` returns 0 for a `moveToXY`-driven
  vehicle, which previously wiped out the sideways velocity every step.
- `movement.*` config keys are now actually read.
