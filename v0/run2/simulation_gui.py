import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------
# Check SUMO installation
# ---------------------------------------------------------

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("SUMO_HOME is not set.")

SUMO_HOME = Path(os.environ["SUMO_HOME"])


# ---------------------------------------------------------
# Get the folder containing this Python file
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Define input/output files
# ---------------------------------------------------------

nod = HERE / "straight3lane.nod.xml"
edg = HERE / "straight3lane.edg.xml"
net = HERE / "straight3lane.net.xml"
cfg = HERE / "straight3lane.sumocfg"


# ---------------------------------------------------------
# Find SUMO executables
# ---------------------------------------------------------

def exe(name):
    suffix = ".exe" if os.name == "nt" else ""
    return SUMO_HOME / "bin" / (name + suffix)


# ---------------------------------------------------------
# Generate the network if it does not already exist
# ---------------------------------------------------------

if not net.exists():

    subprocess.run([
        str(exe("netconvert")),
        "-n", str(nod),
        "-e", str(edg),
        "-o", str(net)
    ], check=True)


# ---------------------------------------------------------
# Start SUMO-GUI
# ---------------------------------------------------------

subprocess.run([
    str(exe("sumo-gui")),
    "-c", str(cfg),
    "--seed", "42",
    "--collision.action", "warn"
], check=True)