import os
import traci
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("SUMO_HOME is not set.")

SUMO_HOME = Path(os.environ["SUMO_HOME"])
HERE = Path(__file__).resolve().parent

def exe(name):
    suffix = ".exe" if os.name == "nt" else ""
    return SUMO_HOME / "bin" / (name + suffix)


# ---------------------------------------------------------
# Start SUMO-GUI through TraCI
# ---------------------------------------------------------

cfg = HERE / "ego_only.sumocfg"

traci.start([
    str(exe("sumo-gui")),
    "-c", str(cfg),
    "--seed", "42"
])


# ---------------------------------------------------------
# Ego controller
# ---------------------------------------------------------

ego_id = "ego"

target_speed = 15.0       # m/s
accel = 2.0               # m/s²
decel = 3.0               # m/s²


while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    # Make sure ego has entered the simulation
    if ego_id not in traci.vehicle.getIDList():
        continue

    x, y = traci.vehicle.getPosition(ego_id)
    speed = traci.vehicle.getSpeed(ego_id)

    if speed < target_speed:
        traci.vehicle.setAcceleration(ego_id,accel,0.1)
    else:
        traci.vehicle.setAcceleration(ego_id,0.0,0.1)

    # -----------------------------------------------------
    # Minimal output
    # -----------------------------------------------------

    if int(traci.simulation.getTime() * 10) % 10 == 0:

        print(
            f"t={traci.simulation.getTime():.1f}s | "
            f"x={x:.2f} m | "
            f"y={y:.2f} m | "
            f"v={speed:.2f} m/s"
        )


traci.close()