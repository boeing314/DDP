# Config-driven Pure DIPF + SUMO

This project uses **`config.yaml` as the single source of truth**.

## Files

- `config.yaml` — all simulation, road, ego, traffic, vehicle-type, and DIPF parameters.
- `generate_sumo.py` — generates:
  - `straight3lane.nod.xml`
  - `straight3lane.edg.xml`
  - `straight3lane.rou.xml`
  - `straight3lane.sumocfg`
  - `straight3lane.net.xml` using SUMO `netconvert`
- `dipf_controller.py` — reads the same `config.yaml` and runs the Pure-DIPF controller.
- `requirements.txt` — Python dependencies.

## Install

Make sure SUMO is installed and `SUMO_HOME` is set.

```bash
pip install -r requirements.txt
```

## Generate SUMO files

```bash
python generate_sumo.py
```

The generated `.sumocfg` contains the network, route file, simulation begin/end,
step length, and random seed from `config.yaml`.

The `.net.xml` is produced by `netconvert` from the generated node and edge XML.
This is the standard SUMO network-generation workflow. SUMO configuration files
refer to the generated network and route files. 

## Run

```bash
python dipf_controller.py
```

## Change parameters

Edit only `config.yaml`. For example:

```yaml
simulation:
  lanes: 4
  lane_width_m: 3.5
  step_length: 0.05
```

Then regenerate:

```bash
python generate_sumo.py
python dipf_controller.py
```

The controller does not independently hard-code these values.

## Important

`config.yaml` contains the **model parameters** and **simulation parameters**.
The generated XML files are derived artifacts and should not be manually edited
if you want configuration reproducibility.
