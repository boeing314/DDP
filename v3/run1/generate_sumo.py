
from pathlib import Path
import subprocess
import shutil
import yaml
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_xml(root, path):
    indent(root, space="    ")
    ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def make_nodes(cfg):
    sim = cfg["simulation"]

    root = Element("nodes")
    SubElement(
        root, "node",
        id="n0",
        x="0",
        y="0",
        type="priority",
    )
    SubElement(
        root, "node",
        id="n1",
        x=str(sim["road_length_m"]),
        y="0",
        type="priority",
    )

    write_xml(root, BASE_DIR / f'{sim["net_name"]}.nod.xml')


def make_edges(cfg):
    sim = cfg["simulation"]

    root = Element("edges")
    SubElement(
        root,
        "edge",
        id="e0",
        **{
            "from": "n0",
            "to": "n1",
            "numLanes": str(sim["lanes"]),
            "speed": str(sim["edge_speed_mps"]),
            "laneWidth": str(sim["lane_width_m"]),
            "spreadType": "center",
        },
    )

    write_xml(root, BASE_DIR / f'{sim["net_name"]}.edg.xml')


def add_vtype(parent, type_id, v):
    SubElement(
        parent,
        "vType",
        id=type_id,
        length=str(v["length_m"]),
        width=str(v["width_m"]),
        guiShape=v["gui_shape"],
        color=v["color"],
        accel=str(v["accel_mps2"]),
        decel=str(v["decel_mps2"]),
        maxSpeed=str(v["max_speed_mps"]),
        sigma=str(v["sigma"]),
        minGap=str(v["min_gap_m"]),
        speedDev=str(v["speed_dev"]),
        speedFactor=str(v["speed_factor"]),
        lcModel=v["lc_model"],
        lcSublane=str(v["lc_sublane"]),
        lcPushy=str(v["lc_pushy"]),
        lcAssertive=str(v["lc_assertive"]),
        lcImpatience=str(v["lc_impatience"]),
        latAlignment=v["lat_alignment"],
        lcStrategic=str(v["lc_strategic"]),
        lcCooperative=str(v["lc_cooperative"]),
        lcSpeedGain=str(v["lc_speed_gain"]),
        lcKeepRight=str(v["lc_keep_right"]),
    )


def make_routes(cfg):
    sim = cfg["simulation"]
    root = Element("routes")

    for type_id, v in cfg["vehicle_types"].items():
        add_vtype(root, type_id, v)

    SubElement(root, "route", id="r_forward", edges="e0")

    for type_id, flow in cfg["flows"].items():
        SubElement(
            root,
            "flow",
            id=f"fwd_{type_id}",
            type=type_id,
            route="r_forward",
            begin=str(sim["begin"]),
            end=str(sim["end"]),
            vehsPerHour=str(flow["vehs_per_hour"]),
            departLane=flow["depart_lane"],
            departPos=str(flow["depart_pos"]),
            departSpeed=flow["depart_speed"],
        )

    ego = cfg["ego"]
    SubElement(
        root,
        "vehicle",
        id=ego["id"],
        type=ego["type_id"],
        route="r_forward",
        depart=str(ego["depart_time"]),
        departLane=str(ego["start_lane"]),
        departPos=str(ego["start_x"]),
        departSpeed=str(ego["start_speed_mps"]),
    )

    write_xml(root, BASE_DIR / f'{sim["net_name"]}.rou.xml')


def make_sumocfg(cfg):
    sim = cfg["simulation"]
    name = sim["net_name"]

    root = Element("configuration")

    input_el = SubElement(root, "input")
    SubElement(input_el, "net-file", value=f"{name}.net.xml")
    SubElement(input_el, "route-files", value=f"{name}.rou.xml")

    time_el = SubElement(root, "time")
    SubElement(time_el, "begin", value=str(sim["begin"]))
    SubElement(time_el, "end", value=str(sim["end"]))
    SubElement(time_el, "step-length", value=str(sim["step_length"]))

    processing_el = SubElement(root, "processing")
    SubElement(processing_el, "time-to-teleport", value="-1")
    SubElement(processing_el, "lateral-resolution", value="0.25")
    SubElement(processing_el, "collision.action", value="warn")
    SubElement(processing_el, "collision.mingap-factor", value="0")
    random_number = SubElement(root, "random_number")
    SubElement(random_number, "seed", value=str(sim["seed"]))

    write_xml(root, BASE_DIR / f"{name}.sumocfg")


def build_network(cfg):
    sim = cfg["simulation"]

    netconvert = shutil.which("netconvert")
    if netconvert is None:
        sumo_home = shutil.which("sumo")
        if sumo_home:
            # Usually netconvert is alongside sumo.
            candidate = Path(sumo_home).with_name("netconvert")
            if candidate.exists():
                netconvert = str(candidate)

    if netconvert is None:
        raise RuntimeError(
            "netconvert was not found. Set SUMO_HOME or add SUMO/bin to PATH."
        )

    name = sim["net_name"]

    cmd = [
        netconvert,
        "-n", str(BASE_DIR / f"{name}.nod.xml"),
        "-e", str(BASE_DIR / f"{name}.edg.xml"),
        "-o", str(BASE_DIR / f"{name}.net.xml"),
    ]

    subprocess.run(cmd, check=True, cwd=BASE_DIR)


def main():
    cfg = load_config()

    make_nodes(cfg)
    make_edges(cfg)
    make_routes(cfg)
    make_sumocfg(cfg)
    build_network(cfg)

    print("Generated:")
    for suffix in ["nod.xml", "edg.xml", "rou.xml", "sumocfg", "net.xml"]:
        print(f"  {cfg['simulation']['net_name']}.{suffix}")


if __name__ == "__main__":
    main()
