"""
FULL PURE-DIPF CONTROLLER - OPTIMIZATION VERSION (v3)

This directory's sole purpose is running optimize.py: a search over
(alpha, tau_y_base, sx, delta_lane, lambda_obs) that minimizes the average
episode cost C (see compute_episode_cost()) across a fixed batch of seeds.
tau_x_base is held fixed at its config.yaml value. Everything else is
unchanged from v2's controller.

This file:
  - READS all controller/simulation parameters from config.yaml
  - DOES NOT create or modify any SUMO XML files
  - Uses the Pure-DIPF potential functions directly from Pure_DIPF_codes(8).py
  - Accepts DIPF_ALPHA / DIPF_TAU_Y_BASE / DIPF_SX / DIPF_DELTA_LANE /
    DIPF_LAMBDA_OBS env var overrides for the params optimize.py tunes
    (plus DIPF_TAU_X_BASE, unused by optimize.py but still available for
    manual runs), and DIPF_OUTPUT_DIR to redirect results output for
    parallel trials.

Changes in this version vs. the original:
  1. Obstacle potential U_obs no longer classifies the obstacle's position as
     front/rear/side. It uses a single, direction-agnostic closing-speed
     term Delta_v (rate of closure between host and obstacle) exactly
     matching
        U_o = lambda * f(Delta_v) / k'(Dx,Dy,Dvx,Dvy),  f(Delta_v) = Delta_v / Delta_v_max
     with k' and tau(Dx,Dvx), tau(Dy,Dvy) as given. Note: tau's tanh()
     argument is the velocity term alone (tanh(Dv)), not delta_pos*delta_v
     as the old code had it -- this has been corrected to match the
     formula exactly.
  2. There is no more silent position clamping at the road boundary. If the
     acceleration-limited kinematic step would put any part of the ego
     vehicle's body outside the road, that step is treated as a COLLISION
     and the episode ends -- it is never corrected/truncated. This is what
     guarantees the acceleration limits (EGO_A_MAX / EGO_AY_MAX) are never
     violated by a boundary clamp.
  3. safe_move_ego() (which retried with fallback lane/position on failure)
     has been removed. There is now a single moveToXY() attempt per step.
     If it fails because of a collision, the step is reported as COLLISION.
     If it fails for any other reason, a RuntimeError is raised -- the
     simulation does not silently continue on a bad move.

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

# Controls SUMO's own RNG for anything randomized in the route/traffic
# definition (e.g. depart_lane: random, depart_speed: random, insertion
# timing jitter). Set simulation.seed in config.yaml to a fixed integer to
# get byte-for-byte identical traffic (and therefore identical DIPF
# trajectories) on every run; leave it unset/null for a different random
# traffic pattern each run.
#
# DIPF_SEED (env var) overrides config.yaml's simulation.seed when set --
# this is what run_seeds.py uses to launch one subprocess per seed without
# editing the YAML file for every run.
if "DIPF_SEED" in os.environ:
    SEED = int(os.environ["DIPF_SEED"])
else:
    SEED = SIM.get("seed", None)


# =============================================================================
# EGO SETTINGS
# =============================================================================

EGO_ID = str(required("ego", "id"))
EGO_TYPE_ID = str(EGO.get("type_id", "egoType"))

EGO_DEPART_TIME = float(required("ego", "depart_time"))
EGO_START_X = float(required("ego", "start_x"))
EGO_START_LANE = int(required("ego", "start_lane"))

GOAL_LANE_INDEX = int(required("ego", "goal_lane_index"))

EGO_LENGTH = float(required("ego", "length_m"))
EGO_WIDTH = float(required("ego", "width_m"))

EGO_MAX_SPEED = float(required("ego", "max_speed_mps"))
EGO_A_MAX = float(required("ego", "max_accel_mps2"))
EGO_AY_MAX = float(required("ego", "max_lateral_accel_mps2"))
EGO_VY_MAX = float(required("ego", "max_lateral_speed_mps"))

# Jerk limits (m/s^3): cap how fast the COMMANDED acceleration itself is
# allowed to change tick-to-tick, on top of the existing |ax|<=EGO_A_MAX /
# |ay|<=EGO_AY_MAX magnitude clamps. Without this, a single spurious
# gradient spike (e.g. from the piecewise k_prime() branch switch sampled
# right at its kink) can swing the commanded acceleration from one extreme
# to the other in one 0.1s tick -- a jerk no real vehicle could produce,
# and one that can destabilize nearby SUMO-native vehicles' own
# car-following math badly enough to crash the SUMO process outright.
EGO_JERK_MAX = float(EGO.get("max_jerk_mps3", 10.0))
EGO_JERK_Y_MAX = float(EGO.get("max_lateral_jerk_mps3", 10.0))

GOAL_X_OFFSET = float(required("ego", "goal_x_offset_from_end_m"))
EGO_GOAL_X = ROAD_LENGTH_M - GOAL_X_OFFSET

GOAL_TOL_X = float(EGO.get("goal_tolerance_x", 1.5))
GOAL_TOL_Y = float(EGO.get("goal_tolerance_y", 1.2))
SUCCESS_ANY_LANE = bool(EGO.get("success_any_lane", True))



# =============================================================================
# PURE-DIPF PARAMETERS
#
# These names correspond directly to the calibrated Pure-DIPF implementation
# in Pure_DIPF_codes(8).py.
# =============================================================================

def env_override(env_name, config_value):
    """
    Optimization-driver hook (v3): if env_name is set, it overrides
    config.yaml's value for this run. This is what optimize.py uses to try
    a candidate parameter vector via a subprocess env, without editing
    config.yaml (which stays the checked-in "current best" reference).
    """
    if env_name in os.environ:
        return float(os.environ[env_name])
    return config_value


LAMBDA_OBS = env_override("DIPF_LAMBDA_OBS", float(required("dipf", "lambda_obs")))
ALPHA = env_override("DIPF_ALPHA", float(required("dipf", "alpha")))
TAU_X_BASE = env_override("DIPF_TAU_X_BASE", float(required("dipf", "tau_x_base")))
TAU_Y_BASE = env_override("DIPF_TAU_Y_BASE", float(required("dipf", "tau_y_base")))

SX = env_override("DIPF_SX", float(required("dipf", "sx")))
SY = float(required("dipf", "sy"))
DELTA_LANE = env_override("DIPF_DELTA_LANE", float(required("dipf", "delta_lane")))

W_DEST = float(required("dipf", "w_dest"))
W_OBS = float(required("dipf", "w_obs"))
W_LANE = float(required("dipf", "w_lane"))

# Single closing-speed normalizer used by the unified (non-classified)
# obstacle potential below. Add a `dipf.dv_max` key to config.yaml; until
# then this falls back to `dv_max_front` if present, else 10.0 m/s.
DV_MAX = float(DIPF["dv_max"]) if "dv_max" in DIPF else float(DIPF.get("dv_max_front", 10.0))

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


# =============================================================================
# SUMO IMPORT
# =============================================================================

def ensure_sumo_tools():
    # Prefer the pip-installed "eclipse-sumo" package (a release build --
    # it's what let this project avoid a rare SUMO-engine assertion crash,
    # "MSCFModel::maximumSafeStopSpeedEuler", that the system-packaged sumo
    # can hit; see project notes) over whatever SUMO_HOME the shell
    # environment happens to have set, so this is the default without
    # requiring any manual environment setup. Falls back to the shell's
    # SUMO_HOME if the pip package isn't installed in this environment.
    try:
        import sumo as _sumo_pkg
        pip_sumo_home = getattr(_sumo_pkg, "SUMO_HOME", None)
        if pip_sumo_home and Path(pip_sumo_home).exists():
            os.environ["SUMO_HOME"] = pip_sumo_home
    except ImportError:
        pass

    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME is not set, and the 'eclipse-sumo' pip package is not "
            "installed.\nEither run `pip install eclipse-sumo` (recommended "
            "-- see README) or set SUMO_HOME to your SUMO installation "
            "directory."
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
    """True road edges (no ego-width/margin shrink)."""
    half = (LANES * LANE_WIDTH_M) / 2.0
    return -half, +half

def road_bounds_y_safe_for_ego():
    """
    Shrunk bounds used only to shape the soft U_lane repulsive potential
    (keeps ego away from the edges pre-emptively). This is NOT used to clamp
    the applied position any more -- see ego_within_road().
    """
    y_left, y_right = road_bounds_y()
    half_ego = 0.5 * EGO_WIDTH
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


def ego_within_road(x, y):
    """
    Hard boundary check on the ego's actual footprint (length x width),
    against the TRUE road edges. If any part of the vehicle body would sit
    outside the road, this returns False and the caller must treat the step
    as a collision -- it must NOT clamp x/y back onto the road, since doing
    so is what silently violates the acceleration limits.
    """
    y_left, y_right = road_bounds_y()
    half_w = EGO_WIDTH / 2.0
    half_l = EGO_LENGTH / 2.0

    if (y - half_w) < y_left or (y + half_w) > y_right:
        return False
    if (x - half_l) < 0.0 or (x + half_l) > ROAD_LENGTH_M:
        return False
    return True


def road_edge_dist(y):
    """
    Lateral clearance from the ego's body (not its centerline) to the
    nearest TRUE road edge -- the same edges/footprint convention as
    ego_within_road(). 0.0 once the body has reached or crossed an edge
    (ego_within_road() would already flag that step as a collision, so
    this should not go negative in logged data).
    """
    y_left, y_right = road_bounds_y()
    half_w = EGO_WIDTH / 2.0

    left_clear = (y - half_w) - y_left
    right_clear = y_right - (y + half_w)

    return max(0.0, min(left_clear, right_clear))


def success_goal_reached(x, y):
    reached_x = abs(x - EGO_GOAL_X) <= GOAL_TOL_X
    if SUCCESS_ANY_LANE:
        return reached_x
    goal_y = lane_centers_y()[GOAL_LANE_INDEX]
    return reached_x and abs(y - goal_y) <= GOAL_TOL_Y

def final_lane_name(y):
    return f"Lane {nearest_lane_index(y) + 1}"


# =============================================================================
# EGO MOVEMENT (single attempt -- no fallback / no silent clamping)
# =============================================================================

def move_ego(x_cmd, y_cmd, lane_i_hint):
    """
    Single moveToXY attempt. Returns "OK" on success, "COLLISION" if the
    failure is a collision, and otherwise raises RuntimeError -- the caller
    is expected to let that propagate rather than silently retry/continue.
    """
    invalid_angle = tc.INVALID_DOUBLE_VALUE

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
        return "OK"
    except traci.exceptions.TraCIException as exc:
        if "collision" in str(exc).lower():
            return "COLLISION"
        raise RuntimeError(
            f"ego_control_step: moveToXY failed for ego at "
            f"x={x_cmd:.3f}, y={y_cmd:.3f}, lane={lane_i_hint}: {exc}"
        ) from exc


# =============================================================================
# PURE-DIPF POTENTIAL FIELD
# =============================================================================

def tau_component(delta_v, base):
    """
    tau(D., Dv) = base . [(1 + alpha*Dv) + (alpha*Dv - 1) * tanh(Dv)] / 2

    Per the supplied formula, the tanh argument is the velocity term Dv
    itself (not delta_pos * delta_v).
    """
    scale = 0.5 * ((1.0 + ALPHA * delta_v) + (ALPHA * delta_v - 1.0) * math.tanh(delta_v))
    return base * scale


def k_prime(dx, dy, dvx, dvy, w_eff, l_eff):
    """
    k'(Dx,Dy,Dvx,Dvy) =
        { (1 - w/Dx) . sqrt( (Dx/tau(Dx,Dvx))^2 + (Dy/tau(Dy,Dvy))^2 ) }  if |Dy| <= l|Dx|/w
        { (1 - l/Dy) . sqrt( ... same ... ) }                             otherwise
    """
    tau_x = tau_component(dvx, TAU_X_BASE)
    tau_y = tau_component(dvy, TAU_Y_BASE)

    eps = 1e-3
    tau_x = tau_x if abs(tau_x) > eps else eps
    tau_y = tau_y if abs(tau_y) > eps else eps

    abs_dx = abs(dx) if abs(dx) > eps else eps
    abs_dy = abs(dy) if abs(dy) > eps else eps

    if abs_dy <= (l_eff * abs_dx) / w_eff:
        term = (1.0 - (w_eff / abs_dx)) if abs_dx > w_eff else 0.1
        term = max(0.1, term)
    else:
        term = (1.0 - (l_eff / abs_dy)) if abs_dy > l_eff else 0.1
        term = max(0.1, term)

    kp = term * math.sqrt((dx / tau_x) ** 2 + (dy / tau_y) ** 2)

    return kp + 0.05  # small additive floor so k' can never hit exactly 0


def U_dest(xh, yh, xd, yd):
    """Exact destination potential from Pure_DIPF_codes(8).py."""
    return math.sqrt(SX * (xh - xd) ** 2 + SY * (yh - yd) ** 2)


def U_lane(yh, y_left, y_right):
    """Exact lane-boundary potential from Pure_DIPF_codes(8).py."""
    eps = 1e-3
    return DELTA_LANE * (
        (1.0 / max(eps, yh - y_left)) ** 2
        + (1.0 / max(eps, y_right - yh)) ** 2
    )


def relative_closing_speed(dx, dy, dvx, dvy):
    """
    Direction-agnostic closing speed Delta_v -- replaces the old
    front/rear/side classification entirely. It is the rate at which the
    host-obstacle separation distance is shrinking:

        Dv = -d(|r|)/dt = -(Dx*Dvx + Dy*Dvy) / |r|,   clamped to >= 0

    Positive when host and obstacle are approaching each other (regardless
    of whether the obstacle is ahead, behind, or to the side); zero (no
    repulsion contribution) when the gap is constant or growing.
    """
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    closing = -(dx * dvx + dy * dvy) / dist
    return max(0.0, closing)


def U_obs_exact(xh, yh, vxh, vyh, xo, yo, vxo, vyo, w_eff=2.0, l_eff=5.0):
    """
    U_o = lambda * f(Dv) / k'(Dx,Dy,Dvx,Dvy),    f(Dv) = Dv / Dv_max

    No region classification (front/rear/side) is used anywhere in this
    function -- only Dx, Dy, Dvx, Dvy feed the potential, exactly as
    specified in the formula.
    """
    dx = xo - xh
    dy = yo - yh
    dvx = vxo - vxh
    dvy = vyo - vyh

    if dx > OBS_RADIUS_AHEAD or dx < -OBS_RADIUS_BEHIND:
        return 0.0
    if abs(dy) > LATERAL_BAND:
        return 0.0

    delta_v = relative_closing_speed(dx, dy, dvx, dvy)
    f_dv = clamp(delta_v / max(1e-6, DV_MAX), 0.0, 1.0)

    if f_dv <= 0.0:
        return 0.0

    w_combined = (EGO_WIDTH + w_eff) / 2.0
    l_combined = (EGO_LENGTH + l_eff) / 2.0

    kp = k_prime(dx, dy, dvx, dvy, w_combined, l_combined)

    return LAMBDA_OBS * f_dv / kp


def calculate_total_potential(x, y, vx, vy, obstacles, goal_x, goal_y):
    _, _, y_left, y_right = road_bounds_y_safe_for_ego()

    total = W_DEST * U_dest(x, y, goal_x, goal_y)
    total += W_LANE * U_lane(y, y_left, y_right)

    for (xo, yo, vxo, vyo, w_eff, l_eff, *_rest) in obstacles:
        total += W_OBS * U_obs_exact(x, y, vx, vy, xo, yo, vxo, vyo, w_eff=w_eff, l_eff=l_eff)

    return total


def numerical_grad_U_total(state, obstacles, xd, yd):
    x, y, vx, vy = state

    def U_total(xx, yy):
        return calculate_total_potential(xx, yy, vx, vy, obstacles, xd, yd)

    eps = GRAD_EPS

    dUx = (U_total(x + eps, y) - U_total(x - eps, y)) / (2.0 * eps)
    dUy = (U_total(x, y + eps) - U_total(x, y - eps)) / (2.0 * eps)
    return dUx, dUy


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
            return (True, f"SUMO collision: collider={collider}, victim={victim}")

    return False, ""


def closest_dist(dx, dy, sum_length, sum_width):
    """
    Bounding-box clearance between two rectangular vehicle bodies.

    dx, dy       -- center-to-center separation (obstacle - ego) in x/y
    sum_length   -- ego_length + obstacle_length (compared against dx)
    sum_width    -- ego_width + obstacle_width (compared against dy)

    Standard axis-aligned-bounding-box distance: each axis gap is clamped
    to >= 0 (a negative gap means the bodies are already aligned/overlapping
    on that axis, contributing nothing to the separation), then combined
    with hypot. This is 0.0 only when the bodies overlap on BOTH axes
    (a true collision); if they're aligned on one axis only (e.g. two
    vehicles in the same lane, dy ~ 0) the result correctly reduces to the
    remaining axis's gap instead of collapsing to 0.
    """
    gap_x = max(0.0, abs(dx) - sum_length / 2.0)
    gap_y = max(0.0, abs(dy) - sum_width / 2.0)

    return math.hypot(gap_x, gap_y)


def nearest_closest_dist(ego_x, ego_y, obstacles):
    """Minimum closest_dist() over all tracked obstacles (NaN if none)."""
    best = float("inf")
    for (xo, yo, vxo, vyo, w_eff, l_eff, w_actual, l_actual) in obstacles:
        dx = xo - ego_x
        dy = yo - ego_y
        best = min(best, closest_dist(dx, dy, EGO_LENGTH + l_actual, EGO_WIDTH + w_actual))

    return best if obstacles else np.nan


def append_collision_row(hist, t, x, y, vx=np.nan, vy=np.nan, vx_apply=np.nan,
                          vy_apply=np.nan, ax=np.nan, ay=np.nan, grad_norm=np.nan,
                          pot=np.nan):
    """
    Log the collision itself as one final row in hist, instead of the
    episode's CSV output simply stopping at the last successful step.
    closest_dist is forced to 0.0 -- by definition, a reported collision
    means the ego's body has reached/overlapped another vehicle (or, for a
    road-departure collision, the road edge), so there is no meaningful
    nonzero separation left to report. Fields that were never computed on
    the failing branch (e.g. acceleration, gradient) are logged as NaN
    rather than guessed.
    """
    edge_dist = road_edge_dist(y)
    progress = (x - EGO_START_X) / max(1e-6, EGO_GOAL_X - EGO_START_X)

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
            "pot": float(pot),
            "min_clearance": np.nan,
            "closest_dist": 0.0,
            "road_edge_dist": float(edge_dist),
            "progress": float(progress),
            "status": "COLLISION",
        }
    )


# =============================================================================
# OBSTACLE LIST
# =============================================================================

_WARNED_MISSING_OBS_DIMS = set()


def get_obstacle_dimensions(type_id):
    """Read effective obstacle dimensions from YAML."""
    if type_id not in OBS_DIMS and type_id not in _WARNED_MISSING_OBS_DIMS:
        _WARNED_MISSING_OBS_DIMS.add(type_id)
        print(
            f"[WARNING] No 'obstacle_effective_dimensions' entry for vehicle "
            f"type '{type_id}' in config.yaml -- falling back to 'default' "
            f"dimensions. Add a '{type_id}' entry under "
            f"obstacle_effective_dimensions to give it its own padding."
        )

    item = OBS_DIMS.get(type_id, OBS_DIMS.get("default", {}))

    if not item:
        print(
            "[WARNING] No 'default' entry under 'obstacle_effective_dimensions' "
            "in config.yaml either -- using hardcoded fallback (width_m=2.1, "
            "length_m=5.0)."
        )
        return 2.1, 5.0

    return (float(item.get("width_m", 2.1)), float(item.get("length_m", 5.0)))


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

        # Real physical dimensions (as SUMO actually models the vehicle),
        # used for the predictive body-overlap collision check -- distinct
        # from w_eff/l_eff above, which are the padded "effective" sizes
        # used only to shape the DIPF obstacle potential.
        try:
            l_actual = traci.vehicle.getLength(vid)
            w_actual = traci.vehicle.getWidth(vid)
        except Exception:
            l_actual, w_actual = l_eff, w_eff

        obstacles.append((xo, yo, vxo, vyo, w_eff, l_eff, w_actual, l_actual))

    return obstacles


# =============================================================================
# EGO SETUP
# =============================================================================

def setup_ego_after_spawn():
    # Pure-DIPF controls the motion directly.
    traci.vehicle.setSpeedMode(EGO_ID, 0)       # remove SUMO's automatic speed control
    traci.vehicle.setLaneChangeMode(EGO_ID, 0)  # remove SUMO's automatic lane change

    try:
        traci.vehicle.setColor(EGO_ID, (255, 0, 255, 255))
    except Exception:
        pass

    # Fresh episode -- the jerk limiter in ego_control_step should not carry
    # over acceleration state from any previous run in this interpreter.
    ego_control_step.prev_ax = 0.0
    ego_control_step.prev_ay = 0.0


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

    goal_y = lane_centers_y()[GOAL_LANE_INDEX]

    # -------------------------------------------------------------------------
    # Goal
    # -------------------------------------------------------------------------
    if success_goal_reached(x, y):
        lane_i_stop = nearest_lane_index(y)

        result = move_ego(EGO_GOAL_X, y, lane_i_stop)
        if result == "COLLISION":
            append_collision_row(hist, traci.simulation.getTime(), EGO_GOAL_X, y, vx=vx, vy=vy)
            return "COLLISION"

        traci.vehicle.setSpeed(EGO_ID, 0.0)
        return "DONE"

    # -------------------------------------------------------------------------
    # Vehicle-vehicle collisions are determined solely by SUMO's own built-in
    # detector (traci.simulation.getCollisions(), checked in run_simulation's
    # main loop as ego_collision_sumo()) -- no manual proximity/rectangle
    # check is done here any more.
    # -------------------------------------------------------------------------
    obstacles = build_obstacle_list(x, y)

    # -------------------------------------------------------------------------
    # Pure DIPF gradient (unified, non-region-classified obstacle potential)
    # -------------------------------------------------------------------------
    dUx, dUy = numerical_grad_U_total((x, y, vx, vy), obstacles, EGO_GOAL_X, goal_y)

    # Pure potential-gradient acceleration, magnitude-clamped to the
    # vehicle's physical accel limits.
    ax_raw = clamp(-dUx, -EGO_A_MAX, EGO_A_MAX)
    ay_raw = clamp(-dUy, -EGO_AY_MAX, EGO_AY_MAX)

    # Jerk limit: the commanded acceleration may only change so much from
    # the previous step's value. This is what actually prevents a one-tick
    # gradient spike (e.g. a numerical artifact from the k_prime() branch
    # kink) from swinging ax/ay between +-max in a single 0.1s tick -- a
    # jerk no real vehicle produces, and one that can destabilize nearby
    # SUMO-native vehicles' own car-following math.
    prev_ax = ego_control_step.prev_ax
    prev_ay = ego_control_step.prev_ay

    max_dax = EGO_JERK_MAX * dt
    max_day = EGO_JERK_Y_MAX * dt

    ax = clamp(ax_raw, prev_ax - max_dax, prev_ax + max_dax)
    ay = clamp(ay_raw, prev_ay - max_day, prev_ay + max_day)

    # Re-clamp to the physical accel limits (the jerk step above could not
    # have pushed us outside them if prev_ax/prev_ay were already within
    # bounds, but this keeps the invariant explicit and robust).
    ax = clamp(ax, -EGO_A_MAX, EGO_A_MAX)
    ay = clamp(ay, -EGO_AY_MAX, EGO_AY_MAX)

    ego_control_step.prev_ax = ax
    ego_control_step.prev_ay = ay

    vx_apply = clamp(vx + ax * dt, 0.0, EGO_MAX_SPEED)
    vy_apply = clamp(vy + ay * dt, -EGO_VY_MAX, EGO_VY_MAX)

    # Integrate the DIPF acceleration WITHOUT clamping to the road. If this
    # step would put any part of the vehicle off the road, that is a
    # collision, not something to silently correct.
    x_new = x + vx * dt + 0.5 * ax * dt * dt
    y_new = y + vy * dt + 0.5 * ay * dt * dt

    grad_norm = float(math.hypot(dUx, dUy))

    if not ego_within_road(x_new, y_new):
        append_collision_row(
            hist, traci.simulation.getTime(), x_new, y_new, vx=vx, vy=vy,
            vx_apply=vx_apply, vy_apply=vy_apply, ax=ax, ay=ay, grad_norm=grad_norm,
        )
        return "COLLISION"

    lane_i = nearest_lane_index(y_new)

    result = move_ego(x_new, y_new, lane_i)
    if result == "COLLISION":
        append_collision_row(
            hist, traci.simulation.getTime(), x_new, y_new, vx=vx, vy=vy,
            vx_apply=vx_apply, vy_apply=vy_apply, ax=ax, ay=ay, grad_norm=grad_norm,
        )
        return "COLLISION"

    traci.vehicle.setSpeed(EGO_ID, vx_apply)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    t = traci.simulation.getTime()

    total_potential = calculate_total_potential(x, y, vx, vy, obstacles, EGO_GOAL_X, goal_y)

    min_clear = float("inf")
    for (xo, yo, *_rest) in obstacles:
        min_clear = min(min_clear, math.hypot(xo - x, yo - y))

    if not obstacles:
        min_clear = np.nan

    closest = nearest_closest_dist(x, y, obstacles)
    edge_dist = road_edge_dist(y)

    progress = (x - EGO_START_X) / max(1e-6, EGO_GOAL_X - EGO_START_X)

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
            "pot": float(total_potential),
            "min_clearance": float(min_clear) if not np.isnan(min_clear) else np.nan,
            "closest_dist": float(closest) if not np.isnan(closest) else np.nan,
            "road_edge_dist": float(edge_dist),
            "progress": float(progress),
            "status": "OK",
        }
    )

    if DEBUG_INTERVAL > 0.0:
        if (abs((t / DEBUG_INTERVAL) - round(t / DEBUG_INTERVAL)) < (STEP_LENGTH / DEBUG_INTERVAL)):
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


# Default jerk-limiter state (setup_ego_after_spawn() resets these at the
# start of each episode; this covers the case of ego_control_step being
# called before that, defensively).
ego_control_step.prev_ax = 0.0
ego_control_step.prev_ay = 0.0


# =============================================================================
# STUCK DETECTION
# =============================================================================

def compute_stuck_flag(hist):
    if len(hist) < 2:
        return False

    window_steps = max(1, int(STUCK_WINDOW_SEC / STEP_LENGTH))

    if len(hist) < window_steps:
        return False

    speeds = np.array([row["vx"] for row in hist], dtype=float)
    xs = np.array([row["x"] for row in hist], dtype=float)

    for i in range(window_steps, len(hist)):
        s = speeds[i - window_steps:i]
        xp = xs[i - window_steps:i]

        if (np.mean(s) < STUCK_SPEED_THRESH and (xp[-1] - xp[0]) < STUCK_PROGRESS_THRESH):
            return True

    return False


def compute_episode_cost(hist):
    """
    Scalar cost summarizing how good this episode was for the current
    dipf.* parameter set -- lower is better. Combines a safety-margin term,
    a progress term, and a time-efficiency term:

        d            = min(closest_dist, road_edge_dist, 0.5)   -- capped at 0.5 m
        C_collision  = e^(1 - d/0.5) / (e - 1)
        C_dist       = 1 - progress
        C_time       = min(1, (final_time - start_time)*EGO_MAX_SPEED / (progress*ROAD_LENGTH_M*3))
        C            = 25*C_collision + 5*C_dist + C_time

    closest_dist/road_edge_dist are taken over rows with status=="OK" only,
    excluding any final append_collision_row() marker row -- that row's
    closest_dist is a hardcoded 0.0 by construction (see
    append_collision_row), not a real measurement, and would otherwise force
    every collision episode's minimum to exactly 0 regardless of how close
    the ego actually got during real driving.
    """
    if not hist:
        return {"C_collision": np.nan, "C_dist": np.nan, "C_time": np.nan, "C": np.nan}

    ok_rows = [r for r in hist if r.get("status") == "OK"]

    closest_vals = [r["closest_dist"] for r in ok_rows if not np.isnan(r["closest_dist"])]
    edge_vals = [r["road_edge_dist"] for r in ok_rows if not np.isnan(r["road_edge_dist"])]

    min_closest = min(closest_vals) if closest_vals else float("inf")
    min_edge = min(edge_vals) if edge_vals else float("inf")

    d = min(min_closest, min_edge, 0.1)
    c_collision = math.exp(1.0 - d / 0.1) / (math.e - 1.0)

    final = hist[-1]
    progress = final["progress"]
    c_dist = 1.0 - progress

    start_time = hist[0]["time"]
    final_time = final["time"]
    safe_progress = progress if abs(progress) > 1e-9 else 1e-9
    c_time = min(1.0, (final_time - start_time) * EGO_MAX_SPEED / (safe_progress * ROAD_LENGTH_M * 3.0))

    c_total = 25.0 * c_collision + 5.0 * c_dist + c_time

    return {
        "C_collision": float(c_collision),
        "C_dist": float(c_dist),
        "C_time": float(c_time),
        "C": float(c_total),
    }


def update_average_cost(results_root):
    """
    Scans every results_*/metrics.json already on disk and writes their
    average cost to results/average_cost.json. Re-run (cheaply) at the end
    of every save_results() call, so it stays current as run_seeds.py works
    through a sweep -- no separate aggregation step needed.
    """
    costs = []
    seeds_used = []

    for metrics_file in sorted(results_root.glob("results_*/metrics.json")):
        try:
            with metrics_file.open("r", encoding="utf-8") as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        c = m.get("C")
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue

        costs.append(c)
        seeds_used.append(metrics_file.parent.name)

    summary = {
        "num_runs": len(costs),
        "average_C": float(np.mean(costs)) if costs else np.nan,
        "min_C": float(np.min(costs)) if costs else np.nan,
        "max_C": float(np.max(costs)) if costs else np.nan,
        "runs_included": seeds_used,
    }

    with (results_root / "average_cost.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    return summary


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(hist, done_reason):
    # DIPF_OUTPUT_DIR (v3): lets an external driver (optimize.py) redirect
    # this run's output to a scratch directory instead of the shared
    # results/ folder -- needed so concurrent trials/seeds evaluating
    # different parameter vectors never collide writing to the same
    # results_<seed>/ path.
    results_root = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results")))
    results_root.mkdir(parents=True, exist_ok=True)

    run_name = f"results_{int(SEED)}" if SEED is not None else "results_random"
    results_dir = results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

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
        "pot",
        "min_clearance",
        "closest_dist",
        "road_edge_dist",
        "progress",
        "status",
    ]

    with trajectory_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hist:
            writer.writerow(row)

    cost = compute_episode_cost(hist)

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
            "avg_speed": float(np.nanmean([r["vx"] for r in hist])),
            "avg_abs_ay": float(np.nanmean([abs(r["ay"]) for r in hist])),
            "min_clearance": (
                float(np.nanmin([r["min_clearance"] for r in hist if not np.isnan(r["min_clearance"])]))
                if any(not np.isnan(r["min_clearance"]) for r in hist)
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

    metrics.update(cost)

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)

    # Skip when DIPF_OUTPUT_DIR is set: that means we're running under
    # optimize.py's per-seed isolated scratch directory, which by
    # construction only ever contains this one seed's results_<seed>/ --
    # "averaging" over it would just restate this single run's own C, not
    # a real average across seeds. Only meaningful in the normal shared
    # results/ folder (e.g. run_seeds.py), where many seeds coexist.
    if "DIPF_OUTPUT_DIR" not in os.environ:
        update_average_cost(results_root)

    return metrics


# =============================================================================
# RUN
# =============================================================================

def run_simulation():
    hist = []
    ego_initialized = False
    done_reason = "timeout"

    max_steps = max(1, int(math.ceil((SIM_END - SIM_BEGIN) / STEP_LENGTH)))

    sumo_binary = sumolib.checkBinary("sumo")

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

    if SEED is not None:
        cmd += ["--seed", str(int(SEED))]
    else:
        cmd += ["--random"]

    print("=" * 90)
    print("FULL PURE-DIPF CONTROLLER (v3 -- optimization variant)")
    print("=" * 90)
    print(f"Config : {CFG_FILE}")
    print(f"Network: {NET_FILE}")
    print(f"Goal   : x={EGO_GOAL_X:.1f} m, lane={GOAL_LANE_INDEX + 1}")
    print(f"Step   : {STEP_LENGTH:.3f} s")
    print(f"End    : {SIM_END:.1f} s")
    print(f"Seed   : {SEED if SEED is not None else '(none -- random traffic each run)'}")
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
                if EGO_ID in traci.vehicle.getIDList():
                    ex, ey = traci.vehicle.getPosition(EGO_ID)
                    evx = traci.vehicle.getSpeed(EGO_ID)
                    try:
                        evy = traci.vehicle.getLateralSpeed(EGO_ID)
                    except Exception:
                        evy = 0.0
                    append_collision_row(hist, t, ex, ey, vx=evx, vy=evy)
                done_reason = "collision"
                break

            vehicle_ids = traci.vehicle.getIDList()

            # Detect ego after departure.
            if not ego_initialized and EGO_ID in vehicle_ids:
                setup_ego_after_spawn()
                ego_initialized = True
                print(f"Ego spawned at {t:.1f}s")

            # Run DIPF controller.
            if ego_initialized and EGO_ID in traci.vehicle.getIDList():
                try:
                    status = ego_control_step(STEP_LENGTH, hist)
                except Exception as exc:
                    print("\nERROR inside ego_control_step:")
                    print(f"{type(exc).__name__}: {exc}")
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

            # SUMO can end before SIM_END.
            if t >= SIM_END:
                break

        # Only classify as stuck after enough trajectory data exists.
        if (done_reason == "timeout" and len(hist) > 0 and compute_stuck_flag(hist)):
            done_reason = "stuck"

    except Exception:
        # Never silently hide controller/TraCI errors.
        print("\nSimulation terminated because of an exception.")
        if hist:
            debug_file = BASE_DIR / "crash_debug_hist.csv"
            with debug_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                writer.writeheader()
                for row in hist:
                    writer.writerow(row)
            print(f"[DEBUG] dumped {len(hist)} pre-crash rows to {debug_file}")
        raise

    finally:
        try:
            traci.close()
        except Exception:
            pass

    metrics = save_results(hist, done_reason)

    run_name = f"results_{int(SEED)}" if SEED is not None else "results_random"
    output_root = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results")))
    run_dir = output_root / run_name

    print("\n" + "=" * 90)
    print(f"SIMULATION COMPLETE | status={done_reason}")
    print(f"Trajectory: {run_dir / 'trajectory.csv'}")
    print(f"Metrics   : {run_dir / 'metrics.json'}")
    print("=" * 90)

    return metrics


if __name__ == "__main__":
    run_simulation()