import os
import shutil
import sys
from pathlib import Path

import traci
import traci.constants as tc


BASE_DIR = Path(__file__).resolve().parent
CFG = BASE_DIR / "traffic3lane.sumocfg"

# ---------------------------------------------------------
# Safety-net tuning
# ---------------------------------------------------------
LANE_WIDTH = 3.5            # must match traffic3lane_edg.xml
LONGITUDINAL_MARGIN = 1.0   # extra buffer beyond vehicle lengths, in meters
LATERAL_MARGIN = 0.15       # extra buffer beyond vehicle widths, in meters
CHECK_WINDOW = 15.0         # only compare vehicles within this longitudinal distance
BRAKE_SPEED_FACTOR = 0.5    # fraction of current speed to slow to when intervening


def find_sumo_gui():

    exe = shutil.which("sumo-gui")

    if exe:
        return exe

    candidates = []

    if os.environ.get("SUMO_HOME"):
        candidates += [
            Path(os.environ["SUMO_HOME"]) / "bin" / "sumo-gui.exe",
            Path(os.environ["SUMO_HOME"]) / "bin" / "sumo-gui",
        ]

    candidates += [
        Path(r"C:\Program Files\Eclipse SUMO\bin\sumo-gui.exe"),
        Path(r"C:\Program Files (x86)\Eclipse SUMO\bin\sumo-gui.exe"),
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    return None


def lane_index(lane_id):
    # e.g. "e0_0" -> 0, "e0_1" -> 1
    try:
        return int(lane_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def subscribe_vehicle(veh_id):
    traci.vehicle.subscribe(
        veh_id,
        (
            tc.VAR_LANE_ID,
            tc.VAR_LANEPOSITION,
            tc.VAR_LANEPOSITION_LAT,
            tc.VAR_SPEED,
        ),
    )


def check_and_avoid_collisions(static_info, delta_t):
    """
    Pulls all subscribed vehicles' latest state in a single TraCI call,
    computes proximity locally (no extra round-trips), and brakes any
    vehicle that's about to breach the safety margins.
    Returns the number of interventions made this step.
    """

    results = traci.vehicle.getAllSubscriptionResults()

    vehicles = []

    for veh_id, data in results.items():

        if veh_id not in static_info:
            continue  # not yet cached (shouldn't normally happen)

        lane = data.get(tc.VAR_LANE_ID)
        x = data.get(tc.VAR_LANEPOSITION)
        lat = data.get(tc.VAR_LANEPOSITION_LAT)
        speed = data.get(tc.VAR_SPEED)

        if lane is None or x is None or lat is None:
            continue

        length, width = static_info[veh_id]
        lidx = lane_index(lane)
        y = lidx * LANE_WIDTH + (LANE_WIDTH / 2.0) + lat

        vehicles.append((veh_id, x, y, speed, length, width))

    # Sort by longitudinal position so we only compare nearby vehicles
    vehicles.sort(key=lambda v: v[1])

    interventions = 0
    n = len(vehicles)

    for i in range(n):

        veh_id, x, y, speed, length, width = vehicles[i]

        for j in range(i + 1, n):

            other_id, ox, oy, ospeed, olength, owidth = vehicles[j]

            dx = ox - x
            if dx > CHECK_WINDOW:
                break  # sorted by x, no closer pairs beyond this

            required_x_gap = (length + olength) / 2.0 + LONGITUDINAL_MARGIN
            required_y_gap = (width + owidth) / 2.0 + LATERAL_MARGIN

            actual_x_gap = abs(dx)
            actual_y_gap = abs(oy - y)

            if actual_x_gap < required_x_gap and actual_y_gap < required_y_gap:

                # Brake whichever vehicle is behind (smaller x) to open the gap
                behind_id, behind_speed = (
                    (veh_id, speed) if x <= ox else (other_id, ospeed)
                )

                new_speed = behind_speed * BRAKE_SPEED_FACTOR

                try:
                    traci.vehicle.slowDown(behind_id, new_speed, delta_t)
                    interventions += 1
                except traci.TraCIException:
                    pass

    return interventions


def main():

    sumo_gui = find_sumo_gui()

    if sumo_gui is None:
        print("ERROR: SUMO-GUI was not found.")
        print("Add SUMO/bin to PATH or set SUMO_HOME.")
        sys.exit(1)

    print("Starting 3-lane heterogeneous traffic simulation...")

    traci.start([
        sumo_gui,
        "-c",
        str(CFG),
        "--start"
    ])

    print("Connected to SUMO through TraCI.")

    delta_t = traci.simulation.getDeltaT() / 1000.0  # ms -> s

    static_info = {}   # veh_id -> (length, width), fetched once per vehicle
    collision_count = 0
    total_interventions = 0

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            # -------------------------------------------------
            # Subscribe newly departed vehicles, drop arrived ones
            # -------------------------------------------------

            for veh_id in traci.simulation.getDepartedIDList():
                subscribe_vehicle(veh_id)
                static_info[veh_id] = (
                    traci.vehicle.getLength(veh_id),
                    traci.vehicle.getWidth(veh_id),
                )

            for veh_id in traci.simulation.getArrivedIDList():
                static_info.pop(veh_id, None)

            # -------------------------------------------------
            # Safety net: check proximity, brake at-risk vehicles
            # -------------------------------------------------

            total_interventions += check_and_avoid_collisions(static_info, delta_t)

            # -------------------------------------------------
            # Collision detection (unchanged)
            # -------------------------------------------------

            colliding = traci.simulation.getCollidingVehiclesIDList()

            if colliding:

                collision_count += 1

                sim_time = traci.simulation.getTime()

                print("\n" + "=" * 70)
                print(
                    f"COLLISION #{collision_count} "
                    f"at simulation time = {sim_time:.2f} s"
                )
                print("=" * 70)

                for veh_id in colliding:

                    try:
                        x, y = traci.vehicle.getPosition(veh_id)
                        speed = traci.vehicle.getSpeed(veh_id)
                        lane = traci.vehicle.getLaneID(veh_id)
                        lane_pos = traci.vehicle.getLanePosition(veh_id)
                        veh_type = traci.vehicle.getTypeID(veh_id)

                        print(f"Vehicle ID : {veh_id}")
                        print(f"Type       : {veh_type}")
                        print(f"Position   : X = {x:.2f} m, Y = {y:.2f} m")
                        print(f"Lane       : {lane}")
                        print(f"Lane pos   : {lane_pos:.2f} m")
                        print(f"Speed      : {speed:.2f} m/s")
                        print("-" * 70)

                    except traci.TraCIException:
                        print(
                            f"Could not retrieve information "
                            f"for vehicle {veh_id}"
                        )

    except KeyboardInterrupt:

        print("\nSimulation interrupted by user.")

    finally:

        print("\n" + "=" * 70)
        print(f"Total collision events detected: {collision_count}")
        print(f"Total safety-net interventions:  {total_interventions}")
        print("=" * 70)

        traci.close()


if __name__ == "__main__":
    main()