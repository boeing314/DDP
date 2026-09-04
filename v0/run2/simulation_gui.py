import os
import subprocess
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("SUMO_HOME is not set.")

SUMO_HOME = Path(os.environ["SUMO_HOME"])
HERE = Path(__file__).resolve().parent
nod = HERE / "straight3lane.nod.xml"
edg = HERE / "straight3lane.edg.xml"
net = HERE / "straight3lane.net.xml"
cfg = HERE / "straight3lane.sumocfg"
def exe(name):
    suffix = ".exe" if os.name == "nt" else ""
    return SUMO_HOME / "bin" / (name + suffix)

subprocess.run([
    str(exe("sumo-gui")),
    "-c", str(cfg),
    "--seed", "42",
    "--collision.action", "warn"
], check=True)