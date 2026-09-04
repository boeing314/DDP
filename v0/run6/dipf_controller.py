"""
FULL PURE-DIPF CONTROLLER - YAML CONFIGURATION VERSION

This file:
  - READS all controller/simulation parameters from config.yaml
  - DOES NOT create or modify any SUMO XML files
  - Uses the Pure-DIPF potential functions directly from Pure_DIPF_codes(8).py

Required files in the same directory:
    config.yaml
    straight3lane.net.xml
    straight3lane.rou.xml
    straight3lane.sumocfg

The .net.xml, .rou.xml and .sumocfg files must already exist.
"""

import os
import sys
import math
import json
import csv
from pathlib import Path

import numpy as np
import yaml


# =============================================================================
# PATHS / YAML
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

with CONFIG_FILE.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

if not isinstance(CONFIG, dict):
    raise ValueError("config.yaml must contain a YAML mapping/object.")


def required(section, key):
    if section not in CONFIG:
        raise KeyError(f"Missing section '{section}' in config.yaml")
    if key not in CONFIG[section]:
        raise KeyError(f"Missing '{section}.{key}' in config.yaml")
    return CONFIG[section][key]


SIM = CONFIG["simulation"]
EGO = CONFIG["ego"]
DIPF = CONFIG["dipf"]
MOVE = CONFIG.get("movement", {})
STUCK = CONFIG.get("stuck_detection", {})
OBS_DIMS = CONFIG.get("obstacle_effective_dimensions", {})

NET_NAME = SIM.get("net_name", "straight3lane")

NET_FILE = BASE_DIR / f"{NET_NAME}.net.xml"
ROU_FILE = BASE_DIR / f"{NET_NAME}.rou.xml"
CFG_FILE = BASE_DIR / f"{NET_NAME}.sumocfg"

if not NET_FILE.exists():
    raise FileNotFoundError(f"SUMO network not found: {NET_FILE}")
if not ROU_FILE.exists():
    raise FileNotFoundError(f"SUMO route file not found: {ROU_FILE}")
if not CFG_FILE.exists():
    raise FileNotFoundError(f"SUMO config not found: {CFG_FILE}")


# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

ROAD_LENGTH_M = float(required("simulation", "road_length_m"))
LANES = int(required("simulation", "lanes"))
LANE_WIDTH_M = float(required("simulation", "lane_width_m"))
SPEED_MPS = float(required("simulation", "edge_speed_mps"))

SIM_BEGIN = float(required("simulation", "begin"))
SIM_END = float(required("simulation", "end"))
STEP_LENGTH = float(required("simulation", "step_length"))

LATERAL_RESOLUTION = float(SIM.get("lateral_resolution", 0.25))
DEBUG_INTERVAL = float(SIM.get("debug_interval", 2.0))

EDGE_ID = str(SIM.get("edge_id", "e0"))


# =============================================================================
# EGO SETTINGS
# =============================================================================

EGO_ID = str(required("ego", "id"))
EGO_TYPE_ID = str(EGO.get("type_id", "egoType"))

EGO_DEPART_TIME = float(required("ego", "depart_time"))
EGO_START_X = float(required("ego", "start_x"))
EGO_START_LANE = int(required("ego", "start_lane"))

GOAL_LANE_INDEX = int(required("ego", "goal_lane_index"))

# These are retained here because they are used by the DIPF controller.
# If your YAML contains them, they are read from YAML; no hard-coded
# calibrated values are used below.
EGO_LENGTH = float(required("ego", "length_m"))
EGO_WIDTH = float(required("ego", "width_m"))

EGO_MAX_SPEED = float(required("ego", "max_speed_mps"))
EGO_A_MAX = float(required("ego", "max_accel_mps2"))
EGO_AY_MAX = float(required("ego", "max_lateral_accel_mps2"))
EGO_VY_MAX = float(required("ego", "max_lateral_speed_mps"))

GOAL_X_OFFSET = float(required("ego", "goal_x_offset_from_end_m"))
EGO_GOAL_X = ROAD_LENGTH_M - GOAL_X_OFFSET

GOAL_TOL_X = float(EGO.get("goal_tolerance_x", 1.5))
GOAL_TOL_Y = float(EGO.get("goal_tolerance_y", 1.2))
SUCCESS_ANY_LANE = bool(EGO.get("success_any_lane", True))

COLLISION_RADIUS = float(EGO.get("collision_radius_m", 0.25))


# =============================================================================
# PURE-DIPF PARAMETERS
#
# These names correspond directly to the calibrated Pure-DIPF implementation
# in Pure_DIPF_codes(8).py.
# =============================================================================

LAMBDA_OBS = float(required("dipf", "lambda_obs"))
ALPHA = float(required("dipf", "alpha"))
TAU_X_BASE = float(required("dipf", "tau_x_base"))
TAU_Y_BASE = float(required("dipf", "tau_y_base"))

SX = float(required("dipf", "sx"))
SY = float(required("dipf", "sy"))
DELTA_LANE = float(required("dipf", "delta_lane"))

W_DEST = float(required("dipf", "w_dest"))
W_OBS = float(required("dipf", "w_obs"))
W_LANE = float(required("dipf", "w_lane"))

W_FRONT = float(required("dipf", "w_front"))
W_REAR = float(required("dipf", "w_rear"))
W_SIDE = float(required("dipf", "w_side"))

DV_MAX_FRONT = float(required("dipf", "dv_max_front"))
DV_MAX_REAR = float(required("dipf", "dv_max_rear"))
DV_MAX_LAT = float(required("dipf", "dv_max_lat"))

OBS_RADIUS_AHEAD = float(required("dipf", "obs_radius_ahead"))
OBS_RADIUS_BEHIND = float(required("dipf", "obs_radius_behind"))
LATERAL_BAND = float(required("dipf", "lateral_band"))

GRAD_EPS = float(required("dipf", "grad_eps"))
Y_MARGIN = float(required("dipf", "y_margin"))


# =============================================================================
# STUCK / MOVEMENT SETTINGS
# =============================================================================

STUCK_SPEED_THRESH = float(STUCK.get("speed_threshold_mps", 0.35))
STUCK_WINDOW_SEC = float(STUCK.get("window_sec", 8.0))
STUCK_PROGRESS_THRESH = float(STUCK.get("progress_threshold_m", 2.0))

MOVE_MATCH_THRESHOLD = float(MOVE.get("match_threshold", 1000.0))
MOVE_KEEP_ROUTE = int(MOVE.get("keep_route", 2))

X_MIN_SAFE = float(MOVE.get("x_min_safe", 0.5))
X_MAX_SAFE = float(
    MOVE.get("x_max_safe", ROAD_LENGTH_M - 0.5)
)


# =============================================================================
# SUMO IMPORT
# =============================================================================

def ensure_sumo_tools():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME is not set.\n"
            "Set it to your SUMO installation directory."
        )

    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    if str(tools) not in sys.path:
        sys.path.append(str(tools))


ensure_sumo_tools()

import sumolib
import traci
import traci.constants as tc


# =============================================================================
# BASIC HELPERS
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
    half = (LANES*LANE_WIDTH_M)/2.0
    return -half, +half

def road_bounds_y_safe_for_ego():
    y_left, y_right = road_bounds_y()
    half_ego = 0.5*EGO_WIDTH
    buf = 0.05
    y_min = y_left + half_ego + Y_MARGIN + buf
    y_max = y_right - half_ego - Y_MARGIN - buf
    return y_min, y_max, y_left, y_right

def nearest_lane_index(y):
    centers = lane_centers_y()
    best_i = 0
    best_d = float("inf")
    for i, yc in enumerate(centers):
        d = abs(y - yc)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

## chk
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
    return reached_x and abs(y - goal_y) <= GOAL_TOL_Y

def final_lane_name(y):
    return f"Lane {nearest_lane_index(y) + 1}"

# =============================================================================
# ROBUST EGO MOVEMENT
# =============================================================================

def safe_move_ego(x_cmd, y_cmd, lane_i_hint=None):
    x_cmd = clamp_x_to_road(x_cmd)

    y_min, y_max, _, _ = road_bounds_y_safe_for_ego()
    y_cmd = clamp(y_cmd, y_min, y_max)

    if lane_i_hint is None:
        lane_i_hint = nearest_lane_index(y_cmd)

    invalid_angle = tc.INVALID_DOUBLE_VALUE

    # Attempt 1: commanded lane and commanded y.
    try:
        traci.vehicle.moveToXY(
            EGO_ID,
            EDGE_ID,
            lane_i_hint,
            x_cmd,
            y_cmd,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD,
        )
        return True
    except Exception:
        pass

    # Attempt 2: nearest lane center.
    lane_i2, y2 = project_to_lane_center(y_cmd)

    try:
        traci.vehicle.moveToXY(
            EGO_ID,
            EDGE_ID,
            lane_i2,
            x_cmd,
            y2,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD,
        )
        return True
    except Exception:
        pass

    # Attempt 3: keep the current SUMO position/lane.
    try:
        x_cur, y_cur = traci.vehicle.getPosition(EGO_ID)

        x3 = clamp_x_to_road(x_cur)
        lane_i3, y3 = project_to_lane_center(y_cur)

        traci.vehicle.moveToXY(
            EGO_ID,
            EDGE_ID,
            lane_i3,
            x3,
            y3,
            angle=invalid_angle,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD,
        )
        return True
    except Exception:
        return False


# =============================================================================
# PURE-DIPF POTENTIAL FIELD
# =============================================================================

def tau(delta_pos, delta_v, base):
    """
    Exact velocity-dependent tau formulation from Pure_DIPF_codes(8).py.
    """
    z = delta_pos * delta_v
    scale = 0.5 * ((1.0 + ALPHA * delta_v) + (ALPHA * delta_v - 1.0) * math.tanh(z))
    return base * scale


def k_prime_exact(dx, dy, dvx, dvy, w_eff, l_eff):
    """
    Exact k' formulation from Pure_DIPF_codes(8).py.
    """
    tau_x = tau(dx, dvx, TAU_X_BASE)
    tau_y = tau(dy, dvy, TAU_Y_BASE)

    eps = 1e-3

    abs_dx = abs(dx) if abs(dx) > eps else eps
    abs_dy = abs(dy) if abs(dy) > eps else eps

    if abs_dy <= (l_eff * abs_dx) / w_eff:
        term1 = (1.0 - (w_eff / abs_dx) if abs_dx > w_eff else 0.1)
        term1 = max(0.1, term1)
        kp = term1 * math.sqrt((dx / tau_x) ** 2+ (dy / tau_y) ** 2)

    else:
        term2 = (1.0 - (l_eff / abs_dy) if abs_dy > l_eff else 0.1)
        term2 = max(0.1, term2)
        kp = term2 * math.sqrt((dx / tau_x) ** 2+ (dy / tau_y) ** 2)

    return kp + 0.05  ##chk


def U_dest(xh, yh, xd, yd):
    """
    Exact destination potential from Pure_DIPF_codes(8).py.
    """
    return math.sqrt(SX * (xh - xd) ** 2+ SY * (yh - yd) ** 2)


def U_lane(yh, y_left, y_right):
    """
    Exact lane-boundary potential from Pure_DIPF_codes(8).py.
    """
    eps = 1e-3

    return DELTA_LANE * ((1.0 /max(eps,yh - y_left,)) ** 2+(1.0 /max(eps,y_right - yh,)) ** 2)

def classify_region(dx, dy):
    if abs(dx) >= abs(dy):
        if dx >= 0.0:
            return "front"
        return "rear"
    if dy >= 0.0:
        return "left"
    return "right"


def compute_closing_speed_by_region(dx,dy,dvx,dvy,region,):
    if region == "front":
        return max(0.0, -dvx)
    if region == "rear":
        return max(0.0, dvx)
    if region == "left":
        return max(0.0, -dvy) ##chk
    if region == "right":
        return max(0.0, dvy)
    return 0.0


def f_closing_region_aware(dx, dy, dvx, dvy):
    """
    Exact region-aware closing function from Pure_DIPF_codes(8).py.
    """
    region = classify_region(dx, dy)
    closing_speed = compute_closing_speed_by_region(dx,dy,dvx,dvy,region,)

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


def U_obs_exact(xh,yh,vxh,vyh,xo,yo,vxo,vyo,w_eff=2.0,l_eff=5.0,):
    """
    Exact obstacle potential from Pure_DIPF_codes(8).py.
    """
    dx = xo - xh
    dy = yo - yh
    dvx = vxo - vxh
    dvy = vyo - vyh

    if dx > OBS_RADIUS_AHEAD or dx < -OBS_RADIUS_BEHIND:
        return 0.0

    if abs(dy) > LATERAL_BAND:
        return 0.0

    f_dv = f_closing_region_aware(dx,dy,dvx,dvy,)

    if f_dv <= 0.0:
        return 0.0 #chk

    w_combined = (EGO_WIDTH + w_eff) / 2.0
    l_combined = (EGO_LENGTH + l_eff) / 2.0

    kp = k_prime_exact(dx,dy,dvx,dvy,w_combined,l_combined,)

    return LAMBDA_OBS * f_dv / kp


def numerical_grad_U_total(state,obstacles,xd,yd,y_left,y_right,):
    x, y, vx, vy = state

    def U_total(xx, yy):
        U = W_DEST * U_dest(xx,yy,xd,yd,)
        U += W_LANE * U_lane(yy,y_left,y_right,)
        for (xo,yo,vxo,vyo,w_eff,l_eff,) in obstacles:
            U += W_OBS * U_obs_exact(xx,yy,vx,vy,xo,yo,vxo,vyo,w_eff=w_eff,l_eff=l_eff,)
        return U

    eps = GRAD_EPS

    dUx = (U_total(x + eps, y)- U_total(x - eps, y)) / (2.0 * eps)
    dUy = (U_total(x, y + eps)- U_total(x, y - eps)) / (2.0 * eps)
    return dUx, dUy


def calculate_total_potential(x,y,vx,vy,obstacles,goal_x,goal_y,):
    _, _, y_left, y_right = road_bounds_y_safe_for_ego()

    total = W_DEST * U_dest(x,y,goal_x,goal_y,)
    total += W_LANE * U_lane(y,y_left,y_right,)

    for (xo,yo,vxo,vyo,w_eff,l_eff,) in obstacles:

        total += W_OBS * U_obs_exact(x,y,vx,vy,xo,yo,vxo,vyo,w_eff=w_eff,l_eff=l_eff,)

    return total


def check_local_minima(x,y,vx,vy,obstacles,goal_x,goal_y,current_potential=None,):
    y_min, y_max, _, _ = road_bounds_y_safe_for_ego()

    if current_potential is None:
        current_potential = calculate_total_potential(x,y,vx,vy,obstacles,goal_x,goal_y,)

    sample_radius = 2.0
    sample_points = 8

    nearby_potentials = []

    for i in range(sample_points):
        angle = 2.0 * math.pi * i / sample_points

        sx = x + sample_radius * math.cos(angle)
        sy = y + sample_radius * math.sin(angle)

        if sy < y_min or sy > y_max:
            continue

        p = calculate_total_potential(sx,sy,vx,vy,obstacles,goal_x,goal_y,)
        nearby_potentials.append(p)

    if not nearby_potentials:
        return False, current_potential, [], 0.0

    all_higher = all(p > current_potential - 0.1 for p in nearby_potentials)

    dist_to_goal = math.hypot(x - goal_x,y - goal_y,)

    eps = 0.1

    ppx = calculate_total_potential(x + eps,y,vx,vy,obstacles,goal_x,goal_y,)

    pmx = calculate_total_potential(x - eps,y,vx,vy,obstacles,goal_x,goal_y,)

    ppy = calculate_total_potential(x,y + eps,vx,vy,obstacles,goal_x,goal_y,)

    pmy = calculate_total_potential(x,y - eps,vx,vy,obstacles,goal_x,goal_y,)

    grad_x = (ppx - pmx) / (2.0 * eps)
    grad_y = (ppy - pmy) / (2.0 * eps)

    gradient_mag = math.hypot(grad_x,grad_y,)

    is_local_min = (gradient_mag < 0.5 and dist_to_goal > 5.0 and all_higher)

    return (is_local_min,current_potential,nearby_potentials,gradient_mag,)

# =============================================================================
# COLLISION / DISTANCE
# =============================================================================

def ego_collision_sumo():
    try:
        cols = traci.simulation.getCollisions()
    except Exception:
        return False, ""

    for c in cols:
        collider = getattr(c, "collider", "")
        victim = getattr(c, "victim", "")

        if collider == EGO_ID or victim == EGO_ID:
            return (True,f"SUMO collision: collider={collider}, victim={victim}",)

    return False, ""


def ego_collision_radius(ego_x, ego_y,obstacles,):
    for (xo,yo,*_rest,) in obstacles:

        if math.hypot(ego_x - xo,ego_y - yo,) <= COLLISION_RADIUS:
            return True
    return False  ##chk

# =============================================================================
# OBSTACLE LIST
# =============================================================================

def get_obstacle_dimensions(type_id):
    """
    Read effective obstacle dimensions from YAML.
    """
    item = OBS_DIMS.get(type_id,OBS_DIMS.get("default", {}),)

    if not item:
        return 2.1, 5.0

    return (float(item.get("width_m", 2.1)),float(item.get("length_m", 5.0)),)


def build_obstacle_list(x, y):
    obstacles = []

    ids = traci.vehicle.getIDList()

    for vid in ids:
        if vid == EGO_ID:
            continue

        xo, yo = traci.vehicle.getPosition(vid)

        if (xo < x - OBS_RADIUS_BEHIND or xo > x + OBS_RADIUS_AHEAD):
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
            typ = "default"

        w_eff, l_eff = get_obstacle_dimensions(typ)

        obstacles.append((xo,yo,vxo,vyo,w_eff,l_eff,))

    return obstacles


# =============================================================================
# EGO SETUP
# =============================================================================

def setup_ego_after_spawn():
    # Pure-DIPF controls the motion directly.
    traci.vehicle.setSpeedMode(EGO_ID, 0)  ##removing the automatic speed control from SUMO
    traci.vehicle.setLaneChangeMode(EGO_ID, 0) ## removing the automatic lane change from SUMO

    try:
        traci.vehicle.setColor(EGO_ID,(255, 0, 255, 255),)
    except Exception:
        pass


# =============================================================================
# ONE DIPF CONTROL STEP
# =============================================================================

def ego_control_step(dt, hist):
    x, y = traci.vehicle.getPosition(EGO_ID)
    vx = traci.vehicle.getSpeed(EGO_ID)

    try:
        vy = traci.vehicle.getLateralSpeed(EGO_ID)
    except Exception:
        vy = 0.0

    y_min, y_max, y_left, y_right = (road_bounds_y_safe_for_ego())

    goal_y = lane_centers_y()[GOAL_LANE_INDEX]

    # -------------------------------------------------------------------------
    # Goal
    # -------------------------------------------------------------------------
    if success_goal_reached(x, y):
        x_stop = clamp_x_to_road(EGO_GOAL_X)
        y_stop = clamp(y, y_min, y_max)

        lane_i_stop = nearest_lane_index(y_stop)

        moved = safe_move_ego(x_stop,y_stop,lane_i_hint=lane_i_stop,)

        if not moved:
            return "MAPFAIL"

        traci.vehicle.setSpeed(EGO_ID,0.0,)

        return "DONE"

    # -------------------------------------------------------------------------
    # Obstacles
    # -------------------------------------------------------------------------
    obstacles = build_obstacle_list(x, y)

    if ego_collision_radius(x,y,obstacles,):
        return "COLLISION"

    # -------------------------------------------------------------------------
    # Pure DIPF gradient
    # -------------------------------------------------------------------------
    dUx, dUy = numerical_grad_U_total((x, y, vx, vy),obstacles,EGO_GOAL_X,goal_y,y_left,y_right,)

    # Pure potential-gradient acceleration.
    ax = clamp(-dUx,-EGO_A_MAX,EGO_A_MAX,)

    ay = clamp(-dUy,-EGO_AY_MAX,EGO_AY_MAX,)

    vx_apply = clamp(vx + ax * dt,0.0,EGO_MAX_SPEED,)

    vy_apply = clamp(vy + ay * dt,-EGO_VY_MAX,EGO_VY_MAX,)

    # Integrate the DIPF acceleration.
    x_new = clamp_x_to_road(x+ vx * dt+ 0.5 * ax * dt * dt)

    y_new = clamp(y+ vy * dt+ 0.5 * ay * dt * dt,y_min,y_max,)

    lane_i = nearest_lane_index(y_new)

    moved = safe_move_ego(x_new,y_new,lane_i_hint=lane_i,)

    if not moved:
        return "MAPFAIL"

    traci.vehicle.setSpeed(EGO_ID,vx_apply,)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    t = traci.simulation.getTime()

    grad_norm = float(math.hypot(dUx, dUy))

    total_potential = calculate_total_potential(x,y,vx,vy,obstacles,EGO_GOAL_X,goal_y,)

    is_min, _, _, grad_mag = (check_local_minima(x,y,vx,vy,obstacles,EGO_GOAL_X,goal_y,total_potential,))

    min_clear = float("inf")

    for (xo,yo,*_rest,) in obstacles:

        min_clear = min(min_clear,math.hypot(xo - x,yo - y,),)

    if not obstacles:
        min_clear = np.nan

    progress = ((x - EGO_START_X)/ max(1e-6,EGO_GOAL_X - EGO_START_X,))

    hist.append(
        {
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
            "min_clearance": (
                float(min_clear)
                if not np.isnan(min_clear)
                else np.nan
            ),
            "is_local_min": int(is_min),
            "progress": float(progress),
        }
    )

    if DEBUG_INTERVAL > 0.0:
        if (abs((t / DEBUG_INTERVAL)- round(t / DEBUG_INTERVAL))< (STEP_LENGTH / DEBUG_INTERVAL)):
            print(
                f"Time {t:.1f}s | "
                f"lane={lane_i + 1} | "
                f"x={x:.2f} | "
                f"y={y:.2f} | "
                f"v={vx_apply:.2f} | "
                f"vy={vy_apply:.2f} | "
                f"ax={ax:.2f} | "
                f"ay={ay:.2f} | "
                f"|gradU|={grad_norm:.3f} | "
                f"obs={len(obstacles)}"
            )

    return "OK"


# =============================================================================
# STUCK DETECTION
# =============================================================================

def compute_stuck_flag(hist):
    if len(hist) < 2:
        return False

    window_steps = max(1,int(STUCK_WINDOW_SEC / STEP_LENGTH),)

    if len(hist) < window_steps:
        return False

    speeds = np.array([row["vx"] for row in hist],dtype=float,)

    xs = np.array([row["x"] for row in hist],dtype=float,)

    for i in range(window_steps,len(hist),):
        s = speeds[i - window_steps:i]

        xp = xs[i - window_steps:i]

        if (np.mean(s) < STUCK_SPEED_THRESH and (xp[-1] - xp[0])< STUCK_PROGRESS_THRESH):
            return True

    return False


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(hist, done_reason):
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True,exist_ok=True,)

    trajectory_file = (results_dir / "trajectory.csv")

    fieldnames = [
        "time",
        "x",
        "y",
        "vx",
        "vy",
        "vx_apply",
        "vy_apply",
        "ax",
        "ay",
        "grad",
        "grad_localmin",
        "pot",
        "min_clearance",
        "is_local_min",
        "progress",
    ]

    with trajectory_file.open("w",newline="",encoding="utf-8",) as f:
        writer = csv.DictWriter(f,fieldnames=fieldnames,)

        writer.writeheader()

        for row in hist:
            writer.writerow(row)

    if hist:
        final = hist[-1]

        metrics = {
            "status": done_reason,
            "success": int(done_reason == "success"),
            "collision": int(done_reason == "collision"),
            "timeout": int(done_reason == "timeout"),
            "stuck": int(done_reason == "stuck"),
            "mapfail": int(done_reason == "mapfail"),
            "final_x": final["x"],
            "final_y": final["y"],
            "final_lane": final_lane_name(final["y"]),
            "final_progress": final["progress"],
            "avg_speed": float(np.mean([r["vx"] for r in hist])),
            "avg_abs_ay": float(np.mean([abs(r["ay"]) for r in hist])),
            "min_clearance": (
                float(
                    np.nanmin(
                        [
                            r["min_clearance"]
                            for r in hist
                            if not np.isnan(
                                r["min_clearance"]
                            )
                        ]
                    )
                )
                if any(
                    not np.isnan(r["min_clearance"])
                    for r in hist
                )
                else np.nan
            ),
        }

    else:
        metrics = {
            "status": done_reason,
            "success": 0,
            "collision": 0,
            "timeout": int(done_reason == "timeout"),
            "stuck": 0,
            "mapfail": 0,
            "final_x": np.nan,
            "final_y": np.nan,
            "final_lane": "NA",
            "final_progress": 0.0,
            "avg_speed": np.nan,
            "avg_abs_ay": np.nan,
            "min_clearance": np.nan,
        }

    with (results_dir / "metrics.json").open("w",encoding="utf-8",) as f:
        json.dump(metrics,f,indent=2,allow_nan=True,)

    return metrics


# =============================================================================
# RUN
# =============================================================================

def run_simulation():
    hist = []
    ego_initialized = False
    done_reason = "timeout"

    max_steps = max(1,int(math.ceil((SIM_END - SIM_BEGIN)/ STEP_LENGTH)),)

    sumo_binary = sumolib.checkBinary("sumo-gui")

    cmd = [
        sumo_binary,
        "-c",
        str(CFG_FILE),
        "--lateral-resolution",
        str(LATERAL_RESOLUTION),
        "--collision.action",
        "warn",
        "--time-to-teleport",
        "-1",
        "--no-step-log",
        "true",
    ]

    print("=" * 90)
    print("FULL PURE-DIPF CONTROLLER")
    print("=" * 90)
    print(f"Config : {CFG_FILE}")
    print(f"Network: {NET_FILE}")
    print(f"Goal   : x={EGO_GOAL_X:.1f} m, lane={GOAL_LANE_INDEX + 1}")
    print(f"Step   : {STEP_LENGTH:.3f} s")
    print(f"End    : {SIM_END:.1f} s")
    print("=" * 90)

    try:
        traci.start(cmd)

        for _ in range(max_steps):
            traci.simulationStep()

            t = traci.simulation.getTime()

            # Check actual SUMO collision.
            hit, msg = ego_collision_sumo()

            if hit:
                print(msg)
                done_reason = "collision"
                break

            vehicle_ids = traci.vehicle.getIDList()

            # Detect ego after departure.
            if (
                not ego_initialized
                and EGO_ID in vehicle_ids
            ):
                setup_ego_after_spawn()
                ego_initialized = True

                print(
                    f"Ego spawned at {t:.1f}s"
                )

            # Run DIPF controller.
            if (
                ego_initialized
                and EGO_ID in traci.vehicle.getIDList()
            ):
                try:
                    status = ego_control_step(STEP_LENGTH, hist,)
                except Exception as exc:
                    print("\nERROR inside ego_control_step:")
                    print( f"{type(exc).__name__}: {exc}")
                    raise

                # IMPORTANT:
                # Print the status so an early termination can be diagnosed.
                if status != "OK":
                    print(f"[STATUS] t={t:.1f}s -> {status}")

                if status == "DONE":
                    done_reason = "success"
                    print("DIPF goal reached.")
                    break

                if status == "COLLISION":
                    done_reason = "collision"
                    break

                if status == "MAPFAIL":
                    done_reason = "mapfail"
                    print("moveToXY mapping failed.")
                    break

            # SUMO can end before SIM_END.
            if t >= SIM_END:
                break

        # Only classify as stuck after enough trajectory data exists.
        if (done_reason == "timeout" and len(hist) > 0 and compute_stuck_flag(hist)):
            done_reason = "stuck"

    except Exception:
        # Never silently hide controller/TraCI errors.
        print("\nSimulation terminated because of an exception.")
        raise

    finally:
        try:
            traci.close()
        except Exception:
            pass

    metrics = save_results(hist,done_reason,)

    print("\n" + "=" * 90)
    print(f"SIMULATION COMPLETE | status={done_reason}")
    print(f"Trajectory: {BASE_DIR / 'results' / 'trajectory.csv'}")
    print(f"Metrics   : {BASE_DIR / 'results' / 'metrics.json'}")
    print("=" * 90)

    return metrics


if __name__ == "__main__":
    run_simulation()
