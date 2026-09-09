
import os
import shutil
import sys
from pathlib import Path

import traci


BASE_DIR = Path(__file__).resolve().parent
CFG = BASE_DIR / "traffic3lane.sumocfg"


def find_sumo_gui():

    # Check PATH
    exe = shutil.which("sumo-gui")

    if exe:
        return exe

    candidates = []

    # Check SUMO_HOME
    if os.environ.get("SUMO_HOME"):
        candidates += [
            Path(os.environ["SUMO_HOME"]) / "bin" / "sumo-gui.exe",
            Path(os.environ["SUMO_HOME"]) / "bin" / "sumo-gui",
        ]

    # Common Windows installation locations
    candidates += [
        Path(r"C:\Program Files\Eclipse SUMO\bin\sumo-gui.exe"),
        Path(r"C:\Program Files (x86)\Eclipse SUMO\bin\sumo-gui.exe"),
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    return None


def main():

    sumo_gui = find_sumo_gui()

    if sumo_gui is None:
        print("ERROR: SUMO-GUI was not found.")
        print("Add SUMO/bin to PATH or set SUMO_HOME.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Ask which vehicle should be highlighted
    # ---------------------------------------------------------

    highlight_id = input(
        "\nEnter vehicle ID to highlight "
        "(e.g. auto_flow.40): "
    ).strip()

    if highlight_id:
        print(f"Vehicle selected for highlighting: {highlight_id}")
    else:
        print("No vehicle selected for highlighting.")

    # ---------------------------------------------------------
    # Start SUMO-GUI through TraCI
    # ---------------------------------------------------------

    print("\nStarting 3-lane heterogeneous traffic simulation...")

    traci.start([
        sumo_gui,
        "-c",
        str(CFG),
        "--start"
    ])

    print("Connected to SUMO through TraCI.")

    collision_count = 0
    highlighted = False

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            # Advance simulation
            traci.simulationStep()

            # -------------------------------------------------
            # Highlight selected vehicle
            # -------------------------------------------------

            if highlight_id:

                vehicles = traci.vehicle.getIDList()

                if highlight_id in vehicles:

                    # Bright magenta
                    traci.vehicle.setColor(
                        highlight_id,
                        (255, 0, 255, 255)
                    )

                    if not highlighted:
                        print(
                            f"\nVehicle {highlight_id} "
                            f"has entered the simulation."
                        )
                        highlighted = True

            # -------------------------------------------------
            # Collision detection
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
                        print(
                            f"Position   : "
                            f"X = {x:.2f} m, "
                            f"Y = {y:.2f} m"
                        )
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
        print(
            f"Total collision events detected: "
            f"{collision_count}"
        )
        print("=" * 70)

        traci.close()


if __name__ == "__main__":
    main()

