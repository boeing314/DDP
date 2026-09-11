import os
import sys
import math
import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

OUT_DIR = Path(__file__).resolve().parent / "Simulator" / "Test_APF" / "Pure_DIPF_Optimized"
NET_NAME = "straight3lane"

ROAD_LENGTH_M = 1000
LANES = 3
LANE_WIDTH_M = 3.5
SPEED_MPS = 20.0

SIM_BEGIN = 0
SIM_END = 3600
STEP_LENGTH = 0.1

USE_GUI_AFTER_HEADLESS = False

# Batch settings
N_RUNS = 100
SEEDS = list(range(42, 42 + N_RUNS))

# Output folders
ANALYSIS_DIR = OUT_DIR / "Pure_DIPF"
RUNS_DIR = ANALYSIS_DIR / "runs"
PLOTS_DIR = ANALYSIS_DIR / "plots"
SELECTED_DIR = ANALYSIS_DIR / "selected_runs"

for d in [OUT_DIR, ANALYSIS_DIR, RUNS_DIR, PLOTS_DIR, SELECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


FLOWS = [
    ("bike", 2160),
    ("auto", 1080),
    ("car", 1260),
    ("bus", 144),
    ("truck", 216),
]
BV_START_X = 15.0

CAL = {
    "auto_lcAssertive": 0.207888,
    "auto_sigma": 0.334265,
    "bus_lcImpatience": 0.300214,
    "bike_minGap": 0.383637,
    "bus_lcPushy": 0.950682,
    "bus_sigma": 0.423239,
    "auto_lcImpatience": 0.800325,
    "bus_speedDev": 0.281946,
    "car_minGap": 0.3933065,
    "car_accel": 2.495427,
}

LATERAL_RESOLUTION = 0.25
PREVIEW_STEPS = 9000
DEBUG_INTERVAL = 2.0


# =============================================================================
# EGO SETTINGS
# =============================================================================

EGO_ID = "ego"
EGO_DEPART_TIME = 20.0
EGO_START_X = 1.0

EGO_START_LANE = 2
GOAL_LANE_INDEX = 0

# Updated ego size
EGO_LENGTH = 4.583
EGO_WIDTH = 1.748

EGO_MAX_SPEED = 30.0
EGO_A_MAX = 3.2
EGO_AY_MAX = 3.0
EGO_VY_MAX = 3.0

EGO_GOAL_X = ROAD_LENGTH_M - 15.0
GOAL_TOL_X = 1.5
GOAL_TOL_Y = 1.2
SUCCESS_ANY_LANE = True
COLLISION_RADIUS = 0.25



# =============================================================================
# PURE DIPF PARAMETERS
# =============================================================================


LAMBDA_OBS = 244.96347872189534
ALPHA = 1.6422052478480245
TAU_X_BASE = 4.721871478710501
TAU_Y_BASE = 1.8648315932156687
SX = 6.99092303528134
SY = 6.94763321383086
DELTA_LANE = 1.0831433701155517
W_DEST = 1.0
W_OBS = 5.83019897534365
W_LANE = 0.744766733104413


W_FRONT = 3.5671088935820676
W_REAR = 0.07796364736680284
W_SIDE = 1.081823868233479
DV_MAX_FRONT = 11.074266277655797
DV_MAX_REAR = 24.840465484499298
DV_MAX_LAT = 14.322195407952098

EGO_W = EGO_WIDTH
EGO_L = EGO_LENGTH

OBS_RADIUS_AHEAD = 61.79779257046734
OBS_RADIUS_BEHIND = 39.74567719579933
LATERAL_BAND = 11.37407177396119
GRAD_EPS = 0.215559818453697

Y_MARGIN = 0.14873687420594128
NEAR_MISS_THRESH = 1.0

# Stuck logic
STUCK_SPEED_THRESH = 0.35
STUCK_WINDOW_SEC = 8.0
STUCK_PROGRESS_THRESH = 2.0

# Robust moveToXY settings
MOVE_MATCH_THRESHOLD = 1000.0
MOVE_KEEP_ROUTE = 2
X_MIN_SAFE = 0.5
X_MAX_SAFE = ROAD_LENGTH_M - 0.5
# =============================================================================
# SUMO IMPORTS
# =============================================================================

def ensure_sumo_tools():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME is not set. Set it to your SUMO installation folder.\n"
            r'Example: setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"'
        )
    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    if str(tools) not in sys.path:
        sys.path.append(str(tools))


ensure_sumo_tools()
import sumolib  # noqa: E402
import traci  # noqa: E402
import traci.constants as tc  # noqa: E402


# =============================================================================
# FILE PATHS
# =============================================================================

nodes_file = OUT_DIR / f"{NET_NAME}.nod.xml"
edges_file = OUT_DIR / f"{NET_NAME}.edg.xml"
net_file = OUT_DIR / f"{NET_NAME}.net.xml"
rou_file = OUT_DIR / f"{NET_NAME}.rou.xml"
cfg_file = OUT_DIR / f"{NET_NAME}.sumocfg"


# =============================================================================
# HELPERS
# =============================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lane_centers_y():
    centers = []
    mid = (LANES - 1) / 2.0
    for i in range(LANES):
        centers.append((i - mid) * LANE_WIDTH_M)
    return centers


def road_bounds_y():
    half = (LANES * LANE_WIDTH_M) / 2.0
    return (-half, +half)


def road_bounds_y_safe_for_ego():
    yL, yR = road_bounds_y()
    half_ego = 0.5 * EGO_W
    buf = 0.05
    y_min = yL + half_ego + Y_MARGIN + buf
    y_max = yR - half_ego - Y_MARGIN - buf
    return y_min, y_max, yL, yR


def nearest_lane_index(y):
    centers = lane_centers_y()
    best_i, best_d = 0, float("inf")
    for i, yc in enumerate(centers):
        d = abs(y - yc)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def safe_mean(arr, default=np.nan):
    return default if len(arr) == 0 else float(np.mean(arr))


def clamp_x_to_road(x):
    return clamp(x, X_MIN_SAFE, X_MAX_SAFE)


def project_to_lane_center(y):
    lane_i = nearest_lane_index(y)
    centers = lane_centers_y()
    return lane_i, centers[lane_i]


def success_goal_reached(x, y):
    reached_x = abs(x - EGO_GOAL_X) <= GOAL_TOL_X
    if SUCCESS_ANY_LANE:
        return reached_x
    goal_y = lane_centers_y()[GOAL_LANE_INDEX]
    return reached_x and (abs(y - goal_y) <= GOAL_TOL_Y)


def final_lane_name(y):
    return f"Lane {nearest_lane_index(y) + 1}"


def safe_move_ego(x_cmd, y_cmd, lane_i_hint=None):
    x_cmd = clamp_x_to_road(x_cmd)

    y_min, y_max, _, _ = road_bounds_y_safe_for_ego()
    y_cmd = clamp(y_cmd, y_min, y_max)

    if lane_i_hint is None:
        lane_i_hint = nearest_lane_index(y_cmd)

    invalid_angle = tc.INVALID_DOUBLE_VALUE

    try:
        traci.vehicle.moveToXY(
            EGO_ID, "e0", lane_i_hint, x_cmd, y_cmd,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD
        )
        return True
    except Exception:
        pass

    lane_i2, y2 = project_to_lane_center(y_cmd)
    try:
        traci.vehicle.moveToXY(
            EGO_ID, "e0", lane_i2, x_cmd, y2,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD
        )
        return True
    except Exception:
        pass

    try:
        x_cur, y_cur = traci.vehicle.getPosition(EGO_ID)
        x3 = clamp_x_to_road(x_cur)
        lane_i3, y3 = project_to_lane_center(y_cur)
        traci.vehicle.moveToXY(
            EGO_ID, "e0", lane_i3, x3, y3,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD
        )
        return True
    except Exception:
        return False


# =============================================================================
# NETWORK / ROUTES
# =============================================================================

def write_nodes_edges():
    nodes_xml = f"""<nodes>
    <node id="n0" x="0" y="0" type="priority"/>
    <node id="n1" x="{ROAD_LENGTH_M}" y="0" type="priority"/>
</nodes>
"""
    edges_xml = f"""<edges>
    <edge id="e0" from="n0" to="n1" numLanes="{LANES}" speed="{SPEED_MPS}"
          laneWidth="{LANE_WIDTH_M}" spreadType="center"/>
</edges>
"""
    nodes_file.write_text(nodes_xml, encoding="utf-8")
    edges_file.write_text(edges_xml, encoding="utf-8")


def build_net():
    netconvert = sumolib.checkBinary("netconvert")
    subprocess.run(
        [netconvert, "-n", str(nodes_file), "-e", str(edges_file), "-o", str(net_file)],
        check=True
    )


def _add_vtype(root, *, vid, length, width, guiShape, color,
               accel=None, decel=None, maxSpeed=None,
               sigma=None, minGap=None, speedDev=None, speedFactor=None,
               lcModel="SL2015", lcSublane=8, lcPushy=None, lcAssertive=None, lcImpatience=None,
               latAlignment="nice",
               lcStrategic=None, lcCooperative=None, lcSpeedGain=None, lcKeepRight=None):
    attrs = {
        "id": vid,
        "length": str(length),
        "width": str(width),
        "guiShape": guiShape,
        "color": color,
        "lcModel": lcModel,
        "lcSublane": str(lcSublane),
        "latAlignment": latAlignment,
    }
    if accel is not None:
        attrs["accel"] = str(accel)
    if decel is not None:
        attrs["decel"] = str(decel)
    if maxSpeed is not None:
        attrs["maxSpeed"] = str(maxSpeed)
    if sigma is not None:
        attrs["sigma"] = str(sigma)
    if minGap is not None:
        attrs["minGap"] = str(minGap)
    if speedDev is not None:
        attrs["speedDev"] = str(speedDev)
    if speedFactor is not None:
        attrs["speedFactor"] = str(speedFactor)

    if lcPushy is not None:
        attrs["lcPushy"] = str(lcPushy)
    if lcAssertive is not None:
        attrs["lcAssertive"] = str(lcAssertive)
    if lcImpatience is not None:
        attrs["lcImpatience"] = str(lcImpatience)

    if lcStrategic is not None:
        attrs["lcStrategic"] = str(lcStrategic)
    if lcCooperative is not None:
        attrs["lcCooperative"] = str(lcCooperative)
    if lcSpeedGain is not None:
        attrs["lcSpeedGain"] = str(lcSpeedGain)
    if lcKeepRight is not None:
        attrs["lcKeepRight"] = str(lcKeepRight)

    ET.SubElement(root, "vType", **attrs)


def write_routes_flows_with_ego():
    root = ET.Element("routes")

    _add_vtype(root, vid="bike",
               length=1.8, width=0.6, guiShape="motorcycle", color="0,1,0",
               accel=3.0, decel=7.0, maxSpeed=35,
               sigma=0.65, minGap=CAL["bike_minGap"],
               speedDev=0.45, speedFactor=1.10,
               lcModel="SL2015", lcSublane=8,
               lcPushy=0.85, lcAssertive=0.80, lcImpatience=0.95,
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.1, lcSpeedGain=2.0, lcKeepRight=0)

    _add_vtype(root, vid="auto",
               length=2.6, width=1.4, guiShape="passenger", color="1,0.6,0",
               accel=2.2, decel=5.0, maxSpeed=26,
               sigma=CAL["auto_sigma"], minGap=0.6,
               speedDev=0.35, speedFactor=1.05,
               lcModel="SL2015", lcSublane=8,
               lcPushy=0.85, lcAssertive=CAL["auto_lcAssertive"], lcImpatience=CAL["auto_lcImpatience"],
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.1, lcSpeedGain=2.0, lcKeepRight=0)

    _add_vtype(root, vid="car",
               length=5.0, width=2.0, guiShape="passenger", color="1,1,0",
               accel=CAL["car_accel"], decel=4.5, maxSpeed=34,
               sigma=0.45, minGap=CAL["car_minGap"],
               speedDev=0.40, speedFactor=1.00,
               lcModel="SL2015", lcSublane=8,
               lcPushy=0.55, lcAssertive=0.35, lcImpatience=0.65,
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.15, lcSpeedGain=1.8, lcKeepRight=0)

    _add_vtype(root, vid="bus",
               length=12.0, width=2.5, guiShape="bus", color="1,0,0",
               accel=1.2, decel=4.0, maxSpeed=24,
               sigma=CAL["bus_sigma"], minGap=1.0,
               speedDev=CAL["bus_speedDev"], speedFactor=0.90,
               lcModel="SL2015", lcSublane=8,
               lcPushy=CAL["bus_lcPushy"], lcAssertive=0.18, lcImpatience=CAL["bus_lcImpatience"],
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.2, lcSpeedGain=1.2, lcKeepRight=0)

    _add_vtype(root, vid="truck",
               length=12.0, width=2.5, guiShape="truck", color="0,0,1",
               accel=1.0, decel=4.0, maxSpeed=22,
               sigma=0.50, minGap=1.2,
               speedDev=0.25, speedFactor=0.85,
               lcModel="SL2015", lcSublane=8,
               lcPushy=0.85, lcAssertive=0.12, lcImpatience=0.35,
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.2, lcSpeedGain=1.2, lcKeepRight=0)

    _add_vtype(root, vid="egoType",
               length=EGO_LENGTH, width=EGO_WIDTH, guiShape="passenger", color="1,0,1",
               accel=3.0, decel=6.0, maxSpeed=EGO_MAX_SPEED,
               sigma=0.0, minGap=0.5, speedDev=0.0, speedFactor=1.0,
               lcModel="SL2015", lcSublane=8,
               lcPushy=0.0, lcAssertive=0.0, lcImpatience=0.0,
               latAlignment="nice",
               lcStrategic=0, lcCooperative=0.0, lcSpeedGain=0.0, lcKeepRight=0)

    ET.SubElement(root, "route", id="r_forward", edges="e0")

    for vtype, vehph in FLOWS:
        ET.SubElement(
            root, "flow",
            id=f"fwd_{vtype}",
            type=vtype,
            route="r_forward",
            begin=str(SIM_BEGIN),
            end=str(SIM_END),
            vehsPerHour=str(vehph),
            departLane="random",
            departPos=str(BV_START_X),
            departSpeed="random",
        )

    ET.SubElement(
        root, "vehicle",
        id=EGO_ID,
        type="egoType",
        route="r_forward",
        depart=str(EGO_DEPART_TIME),
        departLane=str(EGO_START_LANE),
        departPos=str(EGO_START_X),
        departSpeed="0"
    )

    ET.ElementTree(root).write(rou_file, encoding="utf-8", xml_declaration=True)


def write_sumocfg():
    cfg = f"""<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="{rou_file.name}"/>
    </input>
    <time>
        <begin value="{SIM_BEGIN}"/>
        <end value="{SIM_END}"/>
        <step-length value="{STEP_LENGTH}"/>
    </time>
</configuration>
"""
    cfg_file.write_text(cfg, encoding="utf-8")


# =============================================================================
# DIPF / APF HELPERS
# =============================================================================

def tau(delta_pos, delta_v, base):
    z = delta_pos * delta_v
    scale = 0.5 * ((1.0 + ALPHA * delta_v) + (ALPHA * delta_v - 1.0) * math.tanh(z))
    return base * scale


def k_prime_exact(dx, dy, dvx, dvy, w_eff, l_eff):
    tau_x = tau(dx, dvx, TAU_X_BASE)
    tau_y = tau(dy, dvy, TAU_Y_BASE)

    eps = 1e-3
    abs_dx = abs(dx) if abs(dx) > eps else eps
    abs_dy = abs(dy) if abs(dy) > eps else eps

    if abs_dy <= (l_eff * abs_dx) / w_eff:
        term1 = 1.0 - (w_eff / abs_dx) if abs_dx > w_eff else 0.1
        term1 = max(0.1, term1)
        kp = term1 * math.sqrt((dx / tau_x) ** 2 + (dy / tau_y) ** 2)
    else:
        term2 = 1.0 - (l_eff / abs_dy) if abs_dy > l_eff else 0.1
        term2 = max(0.1, term2)
        kp = term2 * math.sqrt((dx / tau_x) ** 2 + (dy / tau_y) ** 2)

    return kp + 0.05


def U_dest(xh, yh, xd, yd):
    return math.sqrt(SX * (xh - xd) ** 2 + SY * (yh - yd) ** 2)


def U_lane(yh, y_left, y_right):
    eps = 1e-3
    return DELTA_LANE * (
        (1.0 / max(eps, (yh - y_left))) ** 2 +
        (1.0 / max(eps, (y_right - yh))) ** 2
    )


def classify_region(dx, dy):
    if abs(dx) >= abs(dy):
        return "front" if dx >= 0.0 else "rear"
    return "left" if dy >= 0.0 else "right"


def compute_closing_speed_by_region(dx, dy, dvx, dvy, region):
    if region == "front":
        return max(0.0, -dvx)
    if region == "rear":
        return max(0.0, dvx)
    if region == "left":
        return max(0.0, -dvy)
    if region == "right":
        return max(0.0, dvy)
    return 0.0


def f_closing_region_aware(dx, dy, dvx, dvy):
    region = classify_region(dx, dy)
    closing_speed = compute_closing_speed_by_region(dx, dy, dvx, dvy, region)

    if region == "front":
        f_base = closing_speed / max(1e-6, DV_MAX_FRONT)
        weight = W_FRONT
    elif region == "rear":
        f_base = closing_speed / max(1e-6, DV_MAX_REAR)
        weight = W_REAR
    else:
        f_base = closing_speed / max(1e-6, DV_MAX_LAT)
        weight = W_SIDE

    f_base = clamp(f_base, 0.0, 1.0)
    return weight * f_base


def U_obs_exact(xh, yh, vxh, vyh, xo, yo, vxo, vyo, w_eff=2.0, l_eff=5.0):
    dx = xo - xh
    dy = yo - yh
    dvx = vxo - vxh
    dvy = vyo - vyh

    if dx > OBS_RADIUS_AHEAD or dx < -OBS_RADIUS_BEHIND:
        return 0.0
    if abs(dy) > LATERAL_BAND:
        return 0.0

    f_dv = f_closing_region_aware(dx, dy, dvx, dvy)
    if f_dv <= 0.0:
        return 0.0

    w_combined = (EGO_W + w_eff) / 2.0
    l_combined = (EGO_L + l_eff) / 2.0
    kp = k_prime_exact(dx, dy, dvx, dvy, w_combined, l_combined)

    return LAMBDA_OBS * f_dv / kp


def numerical_grad_U_total(state, obstacles, xd, yd, y_left, y_right):
    x, y, vx, vy = state

    def U_total(xx, yy):
        U = W_DEST * U_dest(xx, yy, xd, yd)
        U += W_LANE * U_lane(yy, y_left, y_right)
        for (xo, yo, vxo, vyo, w_eff, l_eff) in obstacles:
            U += W_OBS * U_obs_exact(xx, yy, vx, vy, xo, yo, vxo, vyo, w_eff=w_eff, l_eff=l_eff)
        return U

    eps = GRAD_EPS
    dUx = (U_total(x + eps, y) - U_total(x - eps, y)) / (2 * eps)
    dUy = (U_total(x, y + eps) - U_total(x, y - eps)) / (2 * eps)
    return dUx, dUy


def calculate_total_potential(x, y, vx, vy, obstacles, goal_x, goal_y):
    _, _, y_left, y_right = road_bounds_y_safe_for_ego()
    total = W_DEST * U_dest(x, y, goal_x, goal_y)
    total += W_LANE * U_lane(y, y_left, y_right)
    for (xo, yo, vxo, vyo, w_eff, l_eff) in obstacles:
        total += W_OBS * U_obs_exact(x, y, vx, vy, xo, yo, vxo, vyo, w_eff=w_eff, l_eff=l_eff)
    return total


def check_local_minima(x, y, vx, vy, obstacles, goal_x, goal_y, current_potential=None):
    y_min, y_max, _, _ = road_bounds_y_safe_for_ego()

    if current_potential is None:
        current_potential = calculate_total_potential(x, y, vx, vy, obstacles, goal_x, goal_y)

    sample_radius = 2.0
    sample_points = 8
    nearby_potentials = []

    for i in range(sample_points):
        angle = 2 * math.pi * i / sample_points
        sx = x + sample_radius * math.cos(angle)
        sy = y + sample_radius * math.sin(angle)

        if sy < y_min or sy > y_max:
            continue

        p = calculate_total_potential(sx, sy, vx, vy, obstacles, goal_x, goal_y)
        nearby_potentials.append(p)

    if not nearby_potentials:
        return False, current_potential, [], 0.0

    all_higher = all(p > current_potential - 0.1 for p in nearby_potentials)
    dist_to_goal = math.hypot(x - goal_x, y - goal_y)

    eps = 0.1
    ppx = calculate_total_potential(x + eps, y, vx, vy, obstacles, goal_x, goal_y)
    pmx = calculate_total_potential(x - eps, y, vx, vy, obstacles, goal_x, goal_y)
    ppy = calculate_total_potential(x, y + eps, vx, vy, obstacles, goal_x, goal_y)
    pmy = calculate_total_potential(x, y - eps, vx, vy, obstacles, goal_x, goal_y)

    grad_x = (ppx - pmx) / (2 * eps)
    grad_y = (ppy - pmy) / (2 * eps)
    gradient_mag = math.hypot(grad_x, grad_y)

    is_local_min = (gradient_mag < 0.5 and dist_to_goal > 5.0 and all_higher)
    return is_local_min, current_potential, nearby_potentials, gradient_mag


# =============================================================================
# COLLISION / DISTANCE
# =============================================================================

def get_closest_obstacle_distance(x, y, obstacles):
    min_dist = float("inf")
    for (xo, yo, *_rest) in obstacles:
        dist = math.hypot(xo - x, yo - y)
        if dist < min_dist:
            min_dist = dist
    return min_dist if obstacles else np.nan


def ego_collision_sumo():
    try:
        cols = traci.simulation.getCollisions()
    except Exception:
        return False, ""
    for c in cols:
        collider = getattr(c, "collider", "")
        victim = getattr(c, "victim", "")
        if collider == EGO_ID or victim == EGO_ID:
            return True, f"SUMO collision: collider={collider}, victim={victim}"
    return False, ""


def ego_collision_radius(ego_x, ego_y, obstacles):
    for (xo, yo, *_rest) in obstacles:
        if math.hypot(ego_x - xo, ego_y - yo) <= COLLISION_RADIUS:
            return True
    return False


# =============================================================================
# BUILD OBSTACLES / EGO CONTROL
# =============================================================================

def build_obstacle_list(x, y):
    obstacles = []
    ids = traci.vehicle.getIDList()
    for vid in ids:
        if vid == EGO_ID:
            continue

        xo, yo = traci.vehicle.getPosition(vid)

        if xo < x - OBS_RADIUS_BEHIND or xo > x + OBS_RADIUS_AHEAD:
            continue
        if abs(yo - y) > LATERAL_BAND:
            continue

        vxo = traci.vehicle.getSpeed(vid)
        try:
            vyo = traci.vehicle.getLateralSpeed(vid)
        except Exception:
            vyo = 0.0

        try:
            typ = traci.vehicle.getTypeID(vid)
        except Exception:
            typ = "car"

        if typ == "bike":
            w_eff, l_eff = 0.8, 2.0
        elif typ == "auto":
            w_eff, l_eff = 1.6, 3.0
        elif typ in ("bus", "truck"):
            w_eff, l_eff = 2.7, 12.0
        else:
            w_eff, l_eff = 2.1, 5.0

        obstacles.append((xo, yo, vxo, vyo, w_eff, l_eff))

    return obstacles


def setup_ego_after_spawn():
    traci.vehicle.setSpeedMode(EGO_ID, 0)
    traci.vehicle.setLaneChangeMode(EGO_ID, 0)
    try:
        traci.vehicle.setColor(EGO_ID, (255, 0, 255, 255))
    except Exception:
        pass


def ego_control_step(dt, hist):
    x, y = traci.vehicle.getPosition(EGO_ID)
    vx = traci.vehicle.getSpeed(EGO_ID)
    try:
        vy = traci.vehicle.getLateralSpeed(EGO_ID)
    except Exception:
        vy = 0.0

    y_min, y_max, y_left, y_right = road_bounds_y_safe_for_ego()
    goal_y = lane_centers_y()[GOAL_LANE_INDEX]

    if success_goal_reached(x, y):
        x_stop = clamp_x_to_road(EGO_GOAL_X)
        y_stop = clamp(y, y_min, y_max)
        lane_i_stop = nearest_lane_index(y_stop)
        moved = safe_move_ego(x_stop, y_stop, lane_i_hint=lane_i_stop)
        if not moved:
            return "MAPFAIL"
        traci.vehicle.setSpeed(EGO_ID, 0.0)
        return "DONE"

    obstacles = build_obstacle_list(x, y)

    if ego_collision_radius(x, y, obstacles):
        return "COLLISION"

    dUx, dUy = numerical_grad_U_total(
        (x, y, vx, vy),
        obstacles,
        EGO_GOAL_X,
        goal_y,
        y_left,
        y_right,
    )

    ax = clamp(-dUx, -EGO_A_MAX, EGO_A_MAX)
    ay = clamp(-dUy, -EGO_AY_MAX, EGO_AY_MAX)

    vx_apply = clamp(vx + ax * dt, 0.0, EGO_MAX_SPEED)
    vy_apply = clamp(vy + ay * dt, -EGO_VY_MAX, EGO_VY_MAX)

    x_new = clamp_x_to_road(x + vx * dt + 0.5 * ax * dt * dt)
    y_new = clamp(y + vy * dt + 0.5 * ay * dt * dt, y_min, y_max)
    lane_i = nearest_lane_index(y_new)

    moved = safe_move_ego(x_new, y_new, lane_i_hint=lane_i)
    if not moved:
        return "MAPFAIL"

    traci.vehicle.setSpeed(EGO_ID, vx_apply)

    t = traci.simulation.getTime()
    grad_norm = float(math.hypot(dUx, dUy))
    total_potential = calculate_total_potential(x, y, vx, vy, obstacles, EGO_GOAL_X, goal_y)
    is_min, _, _, grad_mag = check_local_minima(
        x, y, vx, vy, obstacles, EGO_GOAL_X, goal_y, total_potential
    )
    min_clear = get_closest_obstacle_distance(x, y, obstacles)

    hist.append({
        "time": float(t),
        "x": float(x),
        "y": float(y),
        "vx": float(vx),
        "vy": float(vy),
        "vx_apply": float(vx_apply),
        "vy_apply": float(vy_apply),
        "ax": float(ax),
        "ay": float(ay),
        "grad": float(grad_norm),
        "grad_localmin": float(grad_mag),
        "pot": float(total_potential),
        "min_clearance": float(min_clear) if not np.isnan(min_clear) else np.nan,
        "is_local_min": int(is_min),
        "progress": float((x - EGO_START_X) / max(1e-6, (EGO_GOAL_X - EGO_START_X))),
    })

    if t % DEBUG_INTERVAL < STEP_LENGTH:
        print(
            f"Time {t:.1f}s | lane={lane_i + 1} | v={vx_apply:.2f} | vy={vy_apply:.2f} "
            f"| ax={ax:.2f} | ay={ay:.2f} | |∇U|={grad_norm:.3f}"
        )

    return "OK"


# =============================================================================
# RUN ONE SIMULATION
# =============================================================================

def compute_local_minima_events(hist_df):
    if hist_df.empty:
        return 0, []

    flags = hist_df["is_local_min"].astype(int).values
    times = hist_df["time"].values
    xs = hist_df["x"].values
    ys = hist_df["y"].values

    events = []
    prev = 0
    for i, f in enumerate(flags):
        if f == 1 and prev == 0:
            events.append({
                "time": float(times[i]),
                "x": float(xs[i]),
                "y": float(ys[i]),
            })
        prev = f

    return len(events), events


def compute_stuck_flag(hist_df):
    if hist_df.empty:
        return False

    window_steps = max(1, int(STUCK_WINDOW_SEC / STEP_LENGTH))
    if len(hist_df) < window_steps:
        return False

    speeds = hist_df["vx"].values
    xs = hist_df["x"].values

    for i in range(window_steps, len(hist_df)):
        s = speeds[i - window_steps:i]
        xp = xs[i - window_steps:i]
        if np.mean(s) < STUCK_SPEED_THRESH and (xp[-1] - xp[0]) < STUCK_PROGRESS_THRESH:
            return True
    return False


def run_single_sim(seed, run_name):
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    hist = []
    traci_started = False
    ego_initialized = False
    ego_started = False
    done_reason = "timeout"

    log_file = run_dir / "sumo.log"
    err_log_file = run_dir / "sumo_error.log"

    sumo_bin = sumolib.checkBinary("sumo-gui")
    cmd = [
        sumo_bin, "-c", str(cfg_file),
        "--seed", str(seed),
        "--lateral-resolution", str(LATERAL_RESOLUTION),
        "--collision.action", "warn",
        "--time-to-teleport", "-1",
        "--no-step-log", "true",
        "--log", str(log_file),
        "--error-log", str(err_log_file),
    ]

    try:
        traci.start(cmd)
        traci_started = True

        for _ in range(PREVIEW_STEPS):
            traci.simulationStep()

            hit, msg = ego_collision_sumo()
            if hit:
                print(msg)
                done_reason = "collision"
                break

            if (not ego_initialized) and (EGO_ID in traci.vehicle.getIDList()):
                setup_ego_after_spawn()
                ego_initialized = True
                ego_started = False

            if ego_initialized and (EGO_ID in traci.vehicle.getIDList()):
                if not ego_started:
                    ego_started = True
                    print(f"\nRun {run_name} | Ego spawned at {traci.simulation.getTime():.1f}s")

                status = ego_control_step(STEP_LENGTH, hist)
                if status == "DONE":
                    done_reason = "success"
                    break
                if status == "COLLISION":
                    done_reason = "collision"
                    break
                if status == "MAPFAIL":
                    done_reason = "mapfail"
                    print(f"Run {run_name}: moveToXY mapping failed, ending run safely.")
                    break

        if done_reason == "timeout" and len(hist) > 0:
            hdf = pd.DataFrame(hist)
            if compute_stuck_flag(hdf):
                done_reason = "stuck"

    finally:
        if traci_started:
            try:
                traci.close()
            except Exception:
                pass

    hist_df = pd.DataFrame(hist)
    hist_csv = run_dir / "trajectory.csv"
    hist_df.to_csv(hist_csv, index=False)

    if hist_df.empty:
        metrics = {
            "run_name": run_name,
            "seed": seed,
            "status": "failed_before_ego_spawn",
            "success": 0,
            "collision": 0,
            "timeout": 1,
            "stuck": 0,
            "mapfail": 0,
            "travel_time": np.nan,
            "min_clearance": np.nan,
            "avg_progress": 0.0,
            "final_progress": 0.0,
            "avg_speed": np.nan,
            "avg_abs_ay": np.nan,
            "jerk_rms": np.nan,
            "local_minima_count": 0,
            "near_miss_count": 0,
            "path_length": np.nan,
            "final_x": np.nan,
            "final_y": np.nan,
            "final_lane": "NA",
            "reached_longitudinal_goal": 0,
            "trajectory_csv": str(hist_csv),
        }
    else:
        local_minima_count, local_minima_events = compute_local_minima_events(hist_df)

        if len(hist_df) >= 2:
            dx = np.diff(hist_df["x"].values)
            dy = np.diff(hist_df["y"].values)
            path_length = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
        else:
            path_length = 0.0

        ay = hist_df["ay"].values
        if len(ay) >= 2:
            jerk_y = np.diff(ay) / STEP_LENGTH
            jerk_rms = float(np.sqrt(np.mean(jerk_y ** 2)))
        else:
            jerk_rms = np.nan

        near_miss_count = int(np.sum(hist_df["min_clearance"].fillna(999) < NEAR_MISS_THRESH))

        metrics = {
            "run_name": run_name,
            "seed": seed,
            "status": done_reason,
            "success": int(done_reason == "success"),
            "collision": int(done_reason == "collision"),
            "timeout": int(done_reason == "timeout"),
            "stuck": int(done_reason == "stuck"),
            "mapfail": int(done_reason == "mapfail"),
            "travel_time": float(hist_df["time"].iloc[-1] - EGO_DEPART_TIME) if done_reason == "success" else np.nan,
            "min_clearance": float(np.nanmin(hist_df["min_clearance"].values)) if hist_df["min_clearance"].notna().any() else np.nan,
            "avg_progress": float(np.nanmean(hist_df["progress"].values)),
            "final_progress": float(hist_df["progress"].iloc[-1]),
            "avg_speed": float(np.mean(hist_df["vx"].values)),
            "avg_abs_ay": float(np.mean(np.abs(hist_df["ay"].values))),
            "jerk_rms": jerk_rms,
            "local_minima_count": int(local_minima_count),
            "near_miss_count": near_miss_count,
            "path_length": path_length,
            "final_x": float(hist_df["x"].iloc[-1]),
            "final_y": float(hist_df["y"].iloc[-1]),
            "final_lane": final_lane_name(float(hist_df["y"].iloc[-1])),
            "reached_longitudinal_goal": int(abs(float(hist_df["x"].iloc[-1]) - EGO_GOAL_X) <= GOAL_TOL_X),
            "trajectory_csv": str(hist_csv),
        }

        with open(run_dir / "local_minima_events.json", "w", encoding="utf-8") as f:
            json.dump(local_minima_events, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, hist_df


# =============================================================================
# PLOTTING
# =============================================================================

def add_trajectory_number_labels(selected_rows):
    labeled = selected_rows.copy().reset_index(drop=True)
    labeled["traj_label"] = [f"Trajectory {i + 1}" for i in range(len(labeled))]
    return labeled


def plot_selected_trajectories(selected_rows, title, save_path):
    plt.figure(figsize=(16, 8))

    yL, yR = road_bounds_y()

    plt.axhline(yL, linewidth=2)
    plt.axhline(yR, linewidth=2)
    for yc in lane_centers_y():
        plt.axhline(yc, linestyle="--", linewidth=1, alpha=0.5)

    plt.axvline(EGO_GOAL_X, linestyle=":", linewidth=2, alpha=0.8, label="Longitudinal goal")

    selected_rows = add_trajectory_number_labels(selected_rows)

    for _, row in selected_rows.iterrows():
        traj = pd.read_csv(row["trajectory_csv"])
        if traj.empty:
            continue

        label = row["traj_label"]
        plt.plot(traj["x"], traj["y"], linewidth=2.2, label=label)
        plt.scatter(traj["x"].iloc[0], traj["y"].iloc[0], s=60)
        plt.scatter(traj["x"].iloc[-1], traj["y"].iloc[-1], s=85, marker="x")

        end_x = float(traj["x"].iloc[-1])
        end_y = float(traj["y"].iloc[-1])
        plt.annotate(
            label,
            xy=(end_x, end_y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.25),
        )

    plt.title(title)
    plt.xlabel("Longitudinal Position x (m)")
    plt.ylabel("Lateral Position y (m)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_selected_local_minima_timeline(selected_rows, title, save_path):
    plt.figure(figsize=(16, 7))

    y_positions = []
    y_labels = []

    for idx, (_, row) in enumerate(selected_rows.iterrows()):
        traj = pd.read_csv(row["trajectory_csv"])
        if traj.empty:
            continue

        count, events = compute_local_minima_events(traj)
        ypos = idx + 1
        y_positions.append(ypos)
        y_labels.append(f"Trajectory {idx + 1} ({count})")

        if len(events) > 0:
            times = [e["time"] for e in events]
            plt.scatter(times, [ypos] * len(times), s=140, marker="X")
        else:
            plt.scatter([0], [ypos], s=60, marker="o")

    plt.title(title)
    plt.xlabel("Time of local minima encounter (s)")
    plt.ylabel("Run")
    plt.yticks(y_positions, y_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_selected_local_minima_bars(selected_rows, title, save_path):
    labels = []
    counts = []

    selected_rows = add_trajectory_number_labels(selected_rows)

    for _, row in selected_rows.iterrows():
        traj = pd.read_csv(row["trajectory_csv"])
        if traj.empty:
            c = 0
        else:
            c, _ = compute_local_minima_events(traj)
        labels.append(row["traj_label"])
        counts.append(c)

    plt.figure(figsize=(14, 6))
    plt.bar(labels, counts)
    plt.title(title)
    plt.ylabel("Number of local minima encountered")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_summary_metrics(results_df):
    plt.figure(figsize=(8, 5))
    status_counts = results_df["status"].value_counts()
    plt.bar(status_counts.index, status_counts.values)
    plt.title("Run Status Counts")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_status_counts.png", dpi=160, bbox_inches="tight")
    plt.close()

    succ = results_df[results_df["success"] == 1]
    if not succ.empty:
        plt.figure(figsize=(10, 5))
        plt.hist(succ["travel_time"].dropna(), bins=12)
        plt.title("Travel Time Distribution (Successful Runs)")
        plt.xlabel("Travel Time (s)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "summary_travel_time_success.png", dpi=160, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(results_df["min_clearance"].dropna(), bins=12)
    plt.title("Minimum Clearance Distribution")
    plt.xlabel("Minimum Clearance (m)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_min_clearance.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(results_df["local_minima_count"].dropna(), bins=12)
    plt.title("Local Minima Count Distribution")
    plt.xlabel("Local minima count")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_local_minima_count.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(results_df["jerk_rms"].dropna(), bins=12)
    plt.title("Jerk RMS Distribution")
    plt.xlabel("Jerk RMS")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_jerk_rms.png", dpi=160, bbox_inches="tight")
    plt.close()


def make_run_diagnostic_plot(row):
    traj = pd.read_csv(row["trajectory_csv"])
    if traj.empty:
        return

    run_name = row["run_name"]
    goal_y = lane_centers_y()[GOAL_LANE_INDEX]

    fig, axes = plt.subplots(4, 2, figsize=(18, 12))

    axes[0, 0].plot(traj["time"], traj["x"], linewidth=2)
    axes[0, 0].axhline(EGO_GOAL_X, linestyle="--", linewidth=2)
    axes[0, 0].set_title("Longitudinal Position")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(traj["time"], traj["y"], linewidth=2)
    axes[0, 1].axhline(goal_y, linestyle="--", linewidth=2)
    axes[0, 1].set_title("Lateral Position")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(traj["time"], traj["vx"], linewidth=2, label="vx")
    axes[1, 0].set_title("Longitudinal Speed")
    axes[1, 0].set_ylabel("vx (m/s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(traj["time"], traj["vy"], linewidth=2)
    axes[1, 1].axhline(0.0, linewidth=1)
    axes[1, 1].set_title("Lateral Speed")
    axes[1, 1].set_ylabel("vy (m/s)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(traj["time"], traj["ax"], linewidth=2)
    axes[2, 0].axhline(EGO_A_MAX, linestyle="--", linewidth=1.2)
    axes[2, 0].axhline(-EGO_A_MAX, linestyle="--", linewidth=1.2)
    axes[2, 0].set_title("Longitudinal Acceleration")
    axes[2, 0].set_ylabel("ax (m/s²)")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(traj["time"], traj["ay"], linewidth=2)
    axes[2, 1].axhline(EGO_AY_MAX, linestyle="--", linewidth=1.2)
    axes[2, 1].axhline(-EGO_AY_MAX, linestyle="--", linewidth=1.2)
    axes[2, 1].set_title("Lateral Acceleration")
    axes[2, 1].set_ylabel("ay (m/s²)")
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 0].plot(traj["time"], traj["pot"], linewidth=2)
    axes[3, 0].set_title("Total Potential")
    axes[3, 0].set_ylabel("Potential")
    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].plot(traj["time"], traj["grad"], linewidth=2, label="|∇U|")
    lm = traj[traj["is_local_min"] == 1]
    if not lm.empty:
        axes[3, 0].scatter(lm["time"], lm["pot"], s=90, marker="X", zorder=5, label="local min")
        axes[3, 1].scatter(lm["time"], lm["grad"], s=90, marker="X", zorder=5, label="local min")
        axes[3, 0].legend()
        axes[3, 1].legend()
    axes[3, 1].set_title("Potential Gradient Magnitude")
    axes[3, 1].set_ylabel("|∇U|")
    axes[3, 1].set_xlabel("Time (s)")
    axes[3, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Diagnostic Plot - {run_name} | status={row['status']} | "
        f"final lane={final_lane_name(row['final_y']) if pd.notna(row['final_y']) else 'NA'}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(SELECTED_DIR / f"{run_name}_diagnostic.png", dpi=160, bbox_inches="tight")
    plt.close()


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def select_runs_for_plotting(results_df):
    success_df = results_df[results_df["success"] == 1].copy()
    failure_df = results_df[results_df["success"] == 0].copy()

    success_df = success_df.sort_values(
        by=["travel_time", "min_clearance"],
        ascending=[True, False],
    )
    selected_success = success_df.head(3)

    failure_df["failure_priority"] = failure_df["status"].map({
        "collision": 0,
        "mapfail": 1,
        "stuck": 2,
        "timeout": 3,
        "failed_before_ego_spawn": 4,
    }).fillna(5)

    failure_df = failure_df.sort_values(
        by=["failure_priority", "local_minima_count", "final_progress"],
        ascending=[True, False, False],
    )
    selected_failure = failure_df.head(3)

    return selected_success, selected_failure


def create_summary_json(results_df):
    succ = results_df[results_df["success"] == 1]
    summary = {
        "full_params_used": {
            "LAMBDA_OBS": LAMBDA_OBS,
            "ALPHA": ALPHA,
            "TAU_X_BASE": TAU_X_BASE,
            "TAU_Y_BASE": TAU_Y_BASE,
            "SX": SX,
            "SY": SY,
            "DELTA_LANE": DELTA_LANE,
            "W_OBS": W_OBS,
            "W_LANE": W_LANE,
            "W_FRONT": W_FRONT,
            "W_REAR": W_REAR,
            "W_SIDE": W_SIDE,
            "DV_MAX_FRONT": DV_MAX_FRONT,
            "DV_MAX_REAR": DV_MAX_REAR,
            "DV_MAX_LAT": DV_MAX_LAT,
            "OBS_RADIUS_AHEAD": OBS_RADIUS_AHEAD,
            "OBS_RADIUS_BEHIND": OBS_RADIUS_BEHIND,
            "LATERAL_BAND": LATERAL_BAND,
            "GRAD_EPS": GRAD_EPS,
            "Y_MARGIN": Y_MARGIN,
        },
        "flows": FLOWS,
        "total_runs": int(len(results_df)),
        "success_rate": float(100.0 * results_df["success"].mean()) if len(results_df) else np.nan,
        "collision_rate": float(100.0 * results_df["collision"].mean()) if len(results_df) else np.nan,
        "timeout_rate": float(100.0 * results_df["timeout"].mean()) if len(results_df) else np.nan,
        "stuck_rate": float(100.0 * results_df["stuck"].mean()) if len(results_df) else np.nan,
        "mapfail_rate": float(100.0 * results_df["mapfail"].mean()) if len(results_df) else np.nan,
        "travel_time_success_mean": safe_mean(succ["travel_time"].dropna().tolist()),
        "travel_time_success_std": float(np.std(succ["travel_time"].dropna().values)) if not succ["travel_time"].dropna().empty else np.nan,
        "min_clearance_mean": safe_mean(results_df["min_clearance"].dropna().tolist()),
        "avg_progress_mean": safe_mean(results_df["avg_progress"].dropna().tolist()),
        "final_progress_mean": safe_mean(results_df["final_progress"].dropna().tolist()),
        "avg_speed_mean": safe_mean(results_df["avg_speed"].dropna().tolist()),
        "avg_abs_ay_mean": safe_mean(results_df["avg_abs_ay"].dropna().tolist()),
        "jerk_rms_mean": safe_mean(results_df["jerk_rms"].dropna().tolist()),
        "local_minima_count_mean": safe_mean(results_df["local_minima_count"].dropna().tolist()),
        "near_miss_count_mean": safe_mean(results_df["near_miss_count"].dropna().tolist()),
        "reached_longitudinal_goal_rate": float(100.0 * results_df["reached_longitudinal_goal"].mean()) if len(results_df) else np.nan,
    }

    with open(ANALYSIS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_batch_evaluation():
    all_metrics = []

    for i, seed in enumerate(SEEDS, start=1):
        run_name = f"run_{i:02d}_seed_{seed}"
        print("\n" + "=" * 90)
        print(f"STARTING {run_name}")
        print("=" * 90)

        metrics, _ = run_single_sim(seed=seed, run_name=run_name)
        all_metrics.append(metrics)

        print(
            f"Completed {run_name} | status={metrics['status']} | "
            f"final_progress={metrics['final_progress']:.3f} | "
            f"local_minima={metrics['local_minima_count']}"
        )

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(ANALYSIS_DIR / "all_run_metrics.csv", index=False)

    summary = create_summary_json(results_df)
    plot_summary_metrics(results_df)

    selected_success, selected_failure = select_runs_for_plotting(results_df)

    selected_success.to_csv(SELECTED_DIR / "selected_success_runs.csv", index=False)
    selected_failure.to_csv(SELECTED_DIR / "selected_failure_runs.csv", index=False)

    if not selected_success.empty:
        plot_selected_trajectories(
            selected_success,
            title="Top 3 Successful Trajectories",
            save_path=SELECTED_DIR / "top3_success_trajectories.png",
        )
        plot_selected_local_minima_timeline(
            selected_success,
            title="Local Minima Timeline - 3 Successful Runs",
            save_path=SELECTED_DIR / "top3_success_local_minima_timeline.png",
        )
        plot_selected_local_minima_bars(
            selected_success,
            title="Local Minima Count - 3 Successful Runs",
            save_path=SELECTED_DIR / "top3_success_local_minima_bar.png",
        )
        for _, row in selected_success.iterrows():
            make_run_diagnostic_plot(row)

    if not selected_failure.empty:
        plot_selected_trajectories(
            selected_failure,
            title="Top 3 Failure Trajectories",
            save_path=SELECTED_DIR / "top3_failure_trajectories.png",
        )
        plot_selected_local_minima_timeline(
            selected_failure,
            title="Local Minima Timeline - 3 Failure Runs",
            save_path=SELECTED_DIR / "top3_failure_local_minima_timeline.png",
        )
        plot_selected_local_minima_bars(
            selected_failure,
            title="Local Minima Count - 3 Failure Runs",
            save_path=SELECTED_DIR / "top3_failure_local_minima_bar.png",
        )
        for _, row in selected_failure.iterrows():
            make_run_diagnostic_plot(row)

    combined = pd.concat([selected_success, selected_failure], axis=0, ignore_index=True)
    if not combined.empty:
        plot_selected_local_minima_timeline(
            combined,
            title="Local Minima Timeline - 3 Success + 3 Failure Runs",
            save_path=SELECTED_DIR / "combined_6runs_local_minima_timeline.png",
        )
        plot_selected_local_minima_bars(
            combined,
            title="Local Minima Count - 3 Success + 3 Failure Runs",
            save_path=SELECTED_DIR / "combined_6runs_local_minima_bar.png",
        )

    return results_df, summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Writing nodes/edges...")
    write_nodes_edges()

    print("Building net.xml...")
    build_net()

    print("Writing routes/flows (BVs + ego)...")
    write_routes_flows_with_ego()

    print("Writing sumocfg...")
    write_sumocfg()

    print("\n" + "=" * 90)
    print(f"Files created in: {OUT_DIR}")
    print(f"Analysis output folder: {ANALYSIS_DIR}")
    print(f"Flows used: {FLOWS}")
    print(f"Runs = {N_RUNS}")
    print(f"Seeds = {SEEDS[0]} ... {SEEDS[-1]}")
    print("HEADLESS ONLY: sumo-gui is not used in this script.")
    print("=" * 90 + "\n")

    results_df, summary = run_batch_evaluation()

    print("\n" + "=" * 90)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 90)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved all outputs to: {ANALYSIS_DIR}")