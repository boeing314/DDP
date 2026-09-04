import os
import traci
from pathlib import Path


# ---------------------------------------------------------
# SUMO setup
# ---------------------------------------------------------

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("SUMO_HOME is not set.")

SUMO_HOME = Path(os.environ["SUMO_HOME"])
HERE = Path(__file__).resolve().parent


def exe(name):
    suffix = ".exe" if os.name == "nt" else ""
    return SUMO_HOME / "bin" / (name + suffix)

cfg = HERE / "follow.sumocfg"

traci.start([
    str(exe("sumo-gui")),
    "-c", str(cfg),
    "--seed", "42",
    "--collision.action", "warn"
])

ego_id = "ego"
lead_id = "lead"
traci.vehicle.setLaneChangeMode(lead_id, 0)
traci.vehicle.setSpeedMode(ego_id, 0)
traci.vehicle.setLaneChangeMode(ego_id, 0)
desired_gap = 2.0       # desired distance from lead [m]
target_time_gap = 0.5    # desired time gap [s]

K_gap = 0.5              # gap control gain
K_speed = 0.8            # speed control gain


# ---------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    if ego_id not in vehicles or lead_id not in vehicles:
        continue

    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    ego_speed = traci.vehicle.getSpeed(ego_id)

    lead_x, lead_y = traci.vehicle.getPosition(lead_id)
    lead_speed = traci.vehicle.getSpeed(lead_id)

    ego_length = traci.vehicle.getLength(ego_id)

    gap = lead_x - ego_x - ego_length

    desired_gap_dynamic = (desired_gap +target_time_gap * ego_speed)

    gap_error = gap-desired_gap_dynamic
    speed_error = lead_speed-ego_speed

    acceleration = (K_gap * gap_error +K_speed * speed_error)
    if acceleration > 2.5:
        acceleration = 2.5
    elif acceleration < -4.5:
        acceleration = -4.5

    traci.vehicle.setAcceleration(ego_id,acceleration, 0.1)

    if round(traci.simulation.getTime(), 1) % 1.0 == 0:

        print(
            f"t={traci.simulation.getTime():.1f} | "
            f"ego_v={ego_speed:.2f} | "
            f"lead_v={lead_speed:.2f} | "
            f"gap={gap:.2f} | "
            f"a={acceleration:.2f}"
        )


traci.close()