"""
FULL PURE-DIPF CONTROLLER (v13: optional adaptive parameters -- a learned gain-scheduling policy, see policy.py)

One SUMO episode driven by the Pure-DIPF controller, with every parameter
read from config.yaml. v6 is identical to v5 except for two hooks used by
optimize.py (the v4 differential evolution driver):

    DIPF_PARAMS      JSON object of dipf.* values overriding config.yaml
                     for this run, e.g. '{"lambda_obs": 12.5, "sx": 0.01}'
    DIPF_OUTPUT_DIR  write results/ there instead of ./results

config.yaml is never edited by a search.

v7 adds state capture (driven by capture_states.py):

    --capture-times 34.5,51.2,67.9,84.6   simulation times to capture at
    --capture-dir   DIR                   where to write the captures

At each capture time (taken at that step, before the controller acts) it
writes state_t<time>.xml (full SUMO state incl. RNG, via
traci.simulation.saveState) and ego_t<time>.json (the controller-held ego
state that SUMO's state file does not contain: vx, vy, last commanded
ax/ay, heading).

v7 episode mode (driven by optimize_episodes.py):

    --episode DIR      start from DIR/state.xml (a captured SUMO state) and
                       DIR/ego.json (the controller-held ego state), instead
                       of the normal t=0 start
    --distance M       success = ego centre advanced M metres from where it
                       starts in the state (default 150; capped at the road
                       goal x = road_length - goal_x_offset)
    --max-time S       give up (timeout) after S seconds (default 60)

v9 additions:
  - encoding.py: laneless, type-aware 80-value description of the surrounding
    traffic; --log-encoding writes it for every step to encoding.csv

From the command line:

    python dipf_controller.py                # seed from config.yaml
    python dipf_controller.py --seed 7       # override the seed
    python dipf_controller.py --seed 7 --gui # watch it in sumo-gui

Changes vs. v4:
  1. Obstacle potential numerator is used exactly as written:
         U_o = lambda * (|Dv| / Dv_max) / k'(Dx,Dy,Dvx,Dvy)
     with |Dv| = hypot(Dvx, Dvy), the magnitude of the relative velocity.
     There is no closing-speed projection, no clamp of the numerator to
     [0, 1], and no zeroing when the gap is constant or growing -- receding
     obstacles now repel too. (U_o is still 0 when both vehicles have
     exactly the same velocity, because |Dv| = 0 in the formula itself.)
  2. Collision risk is measured in TIME: every step logs the time-to-
     collision (TTC) to the nearest obstacle, i.e. the time until the two
     vehicle bodies would first overlap if both kept their current
     velocities. C_collision in the episode cost is computed from the
     minimum TTC instead of a distance.
  3. The ego is given its configured max velocity: generate_sumo.py syncs
     the egoType maxSpeed to ego.max_speed_mps and sets a speedFactor so
     the lane speed limit does not cap it, and the ego departs at
     ego.start_speed_mps.
  4. The ego's velocity (vx, vy) is kept as controller state instead of
     being read back from SUMO every step. getLateralSpeed() returns 0
     after moveToXY(), which previously erased the lateral velocity each
     step and limited sideways motion to 0.5*ay*dt^2 per step.
  5. All positions are vehicle BODY CENTRES. SUMO's getPosition()/moveToXY()
     use the centre of the front bumper, so they are converted (using each
     vehicle's length and heading) at the TraCI boundary. The ego's heading
     passed to moveToXY is the direction of its velocity. ego.start_x is
     the centre too; generate_sumo.py converts it to SUMO's departPos.
  6. U_lane is measured from the ego's BODY edge: its walls sit where the
     body (lateral half-extent incl. heading, see ego_lateral_half_extent())
     plus dipf.y_margin touches the road edge, not where the centre does.
  7. movement.* config keys are now actually read (v4 looked up names that
     did not exist in config.yaml and silently used defaults).

Everything else (tau, k', U_dest, U_lane, road-departure =
collision, single moveToXY attempt) is unchanged from v4.

Required files in the same directory (run generate_sumo.py first):
    config.yaml
    straight3lane.net.xml
    straight3lane.rou.xml
    straight3lane.sumocfg
"""

import os
import sys
import argparse
import contextlib
import subprocess
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

# Optimizer hook (v6): optimize.py passes each candidate's dipf.* values as
# JSON so that parallel runs never touch config.yaml itself.
if "DIPF_PARAMS" in os.environ:
    _overrides = json.loads(os.environ["DIPF_PARAMS"])
    _unknown = set(_overrides) - set(CONFIG.get("dipf", {}))
    if _unknown:
        raise KeyError(f"DIPF_PARAMS sets unknown dipf.* keys: {sorted(_unknown)}")
    CONFIG["dipf"].update(_overrides)


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

LATERAL_RESOLUTION = float(MOVE.get("lateral_resolution_m", 0.25))
DEBUG_INTERVAL = float(MOVE.get("debug_interval_sec", 2.0))

EDGE_ID = str(SIM.get("edge_id", "e0"))

# Controls SUMO's own RNG for anything randomized in the route/traffic
# definition (depart_lane: random, depart_speed: random, ...). A fixed
# integer gives identical traffic every run; null gives random traffic.
# Override per run with:  python dipf_controller.py --seed N
SEED = SIM.get("seed", None)


# =============================================================================
# EGO SETTINGS
# =============================================================================

EGO_ID = str(required("ego", "id"))
EGO_TYPE_ID = str(EGO.get("type_id", "egoType"))

EGO_DEPART_TIME = float(required("ego", "depart_time"))
EGO_START_X = float(required("ego", "start_x"))

# How the ego enters (ego.spawn_mode):
#   route    -- its own <vehicle> in the route file, inserted at depart_time
#   convert  -- traffic runs for convert_time_s, then a vehicle of type
#               convert_type is turned into the ego (gets the ego's vType/size,
#               SUMO control switched off): the one whose centre is nearest
#               convert_x_m (convert_pick: nearest) or the most recently inserted
#               one (convert_pick: latest). Avoids the ego waiting in SUMO's
#               insertion queue when the entrance is saturated.
SPAWN_MODE = str(EGO.get("spawn_mode", "route"))
CONVERT_TIME = float(EGO.get("convert_time_s", 30.0))
CONVERT_TYPE = EGO.get("convert_type", "car")      # vType id, or null = any vehicle type
CONVERT_PICK = str(EGO.get("convert_pick", "nearest"))   # nearest: centre nearest convert_x_m | latest: last inserted
CONVERT_X = float(EGO.get("convert_x_m", 100.0))
EGO_ID_CONFIG = EGO_ID
# int lane index, or "random": SUMO picks the lane at insertion from its own RNG
# (differs between seeds, identical for the same seed)
EGO_START_LANE = required("ego", "start_lane")

GOAL_LANE_INDEX = int(required("ego", "goal_lane_index"))

EGO_LENGTH = float(required("ego", "length_m"))
EGO_WIDTH = float(required("ego", "width_m"))

EGO_MAX_SPEED = float(required("ego", "max_speed_mps"))
EGO_A_MAX = float(required("ego", "max_accel_mps2"))    # forward (throttle) limit
EGO_D_MAX = float(required("ego", "max_decel_mps2"))    # braking limit (positive number)
EGO_AY_MAX = float(required("ego", "max_lateral_accel_mps2"))
EGO_VY_MAX = float(required("ego", "max_lateral_speed_mps"))

# Switch for the limits above. false = the commanded acceleration is the raw
# potential gradient -dU (no magnitude clamp). Speed limits (max_speed_mps,
# max_lateral_speed_mps, no reversing) still apply either way. (v9: there is no
# jerk limit -- the acceleration may change by any amount between ticks.)
LIMIT_ACCEL = bool(EGO.get("limit_acceleration", True))

# Car-like motion: the sideways speed may be at most the forward speed times
# tan(max_heading_deg), i.e. |vy| <= vx * tan(theta_max). Since the ego's
# heading follows its velocity, this caps the heading at theta_max and stops
# a (nearly) stationary ego from sliding sideways.
LIMIT_HEADING = bool(EGO.get("limit_heading", True))
EGO_MAX_HEADING_DEG = float(EGO.get("max_heading_deg", 15.0))
TAN_MAX_HEADING = math.tan(math.radians(EGO_MAX_HEADING_DEG))

GOAL_X_OFFSET = float(required("ego", "goal_x_offset_from_end_m"))
EGO_GOAL_X = ROAD_LENGTH_M - GOAL_X_OFFSET

GOAL_TOL_X = float(EGO.get("goal_tolerance_x", 1.5))
GOAL_TOL_Y = float(EGO.get("goal_tolerance_y", 1.2))
SUCCESS_ANY_LANE = bool(EGO.get("success_any_lane", True))



# =============================================================================
# PURE-DIPF PARAMETERS (read straight from config.yaml -- no overrides)
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

# Normalizer for the |Dv| / Dv_max numerator of the obstacle potential.
DV_MAX = float(required("dipf", "dv_max"))

# true : U_o = lambda * (|Dv| / Dv_max) / k'
# false: U_o = lambda / k'   (velocity term removed from the numerator; Dv
#        still enters through tau inside k')
USE_DV_NUMERATOR = bool(DIPF.get("use_dv_numerator", True))

# k' form. false: as written in the report -- (1 - w/|Dx|) on the
# longitudinal branch, (1 - l/|Dy|) on the lateral one, branch test
# |Dy| <= l|Dx|/w. true: w and l swapped everywhere, so each factor reaches
# 0 exactly when the bodies touch (|Dx| = l nose-to-tail, |Dy| = w
# side-to-side) and the branches meet at the box corner (l, w).
SWAP_WL = bool(DIPF.get("swap_wl_in_kprime", False))

# Laneless traffic encoding (encoding.py), written per step with --log-encoding.
ENCODING_RANGE = float(CONFIG.get("encoding", {}).get("range_m", 150.0))
LOG_ENCODING = False      # --log-encoding
ENCODING_ROWS = []

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

MOVE_MATCH_THRESHOLD = float(MOVE.get("move_match_threshold", 1000.0))
MOVE_KEEP_ROUTE = int(MOVE.get("move_keep_route", 2))


# Time-based collision measure (see time_to_collision()).
COST = CONFIG.get("cost", {})
TTC_THRESHOLD = float(COST.get("ttc_threshold_s", 3.0))
W_COST_COLLISION = float(COST.get("w_collision", 25.0))
W_COST_DIST = float(COST.get("w_dist", 5.0))
W_COST_TIME = float(COST.get("w_time", 1.0))


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


SUMO_GUI = False  # set by --gui in __main__
CAPTURE_TIMES = []    # set by --capture-times in __main__ (sorted, seconds)
CAPTURE_DIR = None    # set by --capture-dir in __main__
CAPTURE_STOP = False  # v11: end the run right after the last capture (scenario creation)
# v13: adaptive parameters. ADAPTIVE = a policy dict (policy.make_policy) -> every control step the
# parameters in ADAPTIVE["params"] are recomputed from the traffic features (policy.py); None = fixed.
ADAPTIVE = None
ADAPT_LOG = False     # record ADAPT_TRACE rows: [time, *features, *parameter values]
ADAPT_TRACE = []

# Episode mode (set by --episode / --distance / --max-time in __main__).
EPISODE_DIR = None        # folder with state.xml + ego.json
EPISODE_EGO = None        # parsed ego.json
EPISODE_MAX_TIME = 60.0   # s after the state's time
EPISODE_SUMO_SEED = None  # episode mode: --seed for the reloaded SUMO (None = SUMO default); different values
                          # give different random futures from the same saved state
SEGMENT_LENGTH_M = None   # distance to cover; None = whole road (normal mode)
RUN_NAME = None           # results sub-folder name; None = results_<seed>

# In-process mode (run_episode(), used by optimize_episodes.py): the Python
# process (imports, config) is reused across evaluations and every
# evaluation starts a FRESH SUMO with a fast connect (10 ms retry interval
# instead of traci.start's fixed 1 s). A fresh SUMO per evaluation is
# required: reusing one SUMO via traci.simulation.loadState() does not give
# identical results (state not fully reset between episodes).
FAST_START = False
QUIET = False             # silence SUMO's own output (warnings) too
MAX_SIM_TIME = None       # normal mode: stop this many seconds after the ego's departure (None = simulation.end)


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

def ego_lateral_half_extent(vx, vy):
    """
    Half of the ego body's footprint across the road (y direction), for a
    body of EGO_LENGTH x EGO_WIDTH whose heading follows its velocity:

        theta = atan(|vy| / vx),   half = (L/2)|sin theta| + (W/2)|cos theta|

    Equals W/2 when driving straight; grows when the ego is angled during a
    lateral manoeuvre (e.g. 12 deg -> 1.32 m instead of 0.87 m).
    """
    theta = math.atan2(abs(vy), max(vx, 0.0)) if math.hypot(vx, vy) > 1e-6 else 0.0
    return 0.5 * EGO_LENGTH * abs(math.sin(theta)) + 0.5 * EGO_WIDTH * abs(math.cos(theta))


def road_bounds_y_safe_for_ego(half_extent):
    """
    Lateral limits for the ego's CENTRE such that its body (lateral
    half-extent `half_extent`) stays dipf.y_margin inside the true road
    edges. U_lane is singular at these limits, i.e. at the point where the
    body edge (plus margin) touches the road edge -- not where the centre
    does. Still not used to clamp the applied position -- see
    ego_within_road().
    """
    y_left, y_right = road_bounds_y()
    y_min = y_left + half_extent + Y_MARGIN
    y_max = y_right - half_extent - Y_MARGIN
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
    if EPISODE_DIR is not None:
        # Episode mode: reaching or passing the end of the segment counts
        # (a +-tolerance window can be jumped over in one step).
        return x >= EGO_GOAL_X - GOAL_TOL_X
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

# SUMO reports and accepts vehicle positions at the CENTRE OF THE FRONT
# BUMPER (getPosition, moveToXY), with headings in degrees measured clockwise
# from north (0 = +y, 90 = +x). Everything inside this controller -- the
# potentials, TTC, clearances, road/goal checks -- works with the CENTRE of
# the vehicle body, so positions are converted at the TraCI boundary only.

def heading_unit(angle_deg):
    """Unit heading vector (hx, hy) for a SUMO angle."""
    a = math.radians(angle_deg)
    return math.sin(a), math.cos(a)


def front_to_center(x_front, y_front, length, angle_deg):
    hx, hy = heading_unit(angle_deg)
    return x_front - 0.5 * length * hx, y_front - 0.5 * length * hy


def center_to_front(x_center, y_center, length, angle_deg):
    hx, hy = heading_unit(angle_deg)
    return x_center + 0.5 * length * hx, y_center + 0.5 * length * hy


def vehicle_center(vid, length):
    """Body-centre position of a SUMO vehicle."""
    x_front, y_front = traci.vehicle.getPosition(vid)
    return front_to_center(x_front, y_front, length, traci.vehicle.getAngle(vid))


def velocity_angle(vx, vy, fallback_deg):
    """SUMO heading (deg) of the velocity vector; fallback when ~stationary."""
    if math.hypot(vx, vy) < 1e-6:
        return fallback_deg
    return math.degrees(math.atan2(vx, vy))


def move_ego(x_center, y_center, lane_i_hint, angle_deg):
    """
    Single moveToXY attempt placing the ego's body CENTRE at
    (x_center, y_center) with heading angle_deg. Returns "OK" on success,
    "COLLISION" if the failure is a collision, and otherwise raises
    RuntimeError -- the caller is expected to let that propagate rather than
    silently retry/continue.
    """
    x_front, y_front = center_to_front(x_center, y_center, EGO_LENGTH, angle_deg)

    try:
        traci.vehicle.moveToXY(
            EGO_ID,
            EDGE_ID,
            lane_i_hint,
            x_front,
            y_front,
            angle=angle_deg,
            keepRoute=MOVE_KEEP_ROUTE,
            matchThreshold=MOVE_MATCH_THRESHOLD,
        )
        return "OK"
    except traci.exceptions.TraCIException as exc:
        if "collision" in str(exc).lower():
            return "COLLISION"
        raise RuntimeError(
            f"ego_control_step: moveToXY failed for ego at centre "
            f"x={x_center:.3f}, y={y_center:.3f}, lane={lane_i_hint}: {exc}"
        ) from exc


# =============================================================================
# PURE-DIPF POTENTIAL FIELD
# =============================================================================

def tau_component(delta_pos, delta_v, base):
    """
    tau(D., Dv, base) = base/2 . [ (2 + alpha*|Dv|) + alpha*|Dv| * tanh(-D. * Dv) ]

    D. and Dv are obstacle minus ego, so D.*Dv < 0 while the gap is closing:
    tanh -> +1 gives tau = base*(1 + alpha*|Dv|) (larger tau, stronger field),
    tanh -> -1 (gap opening) gives tau = base.
    """
    adv = ALPHA * abs(delta_v)
    scale = 0.5 * ((2.0 + adv) + adv * math.tanh(-delta_pos * delta_v))
    return base * scale


def k_prime(dx, dy, dvx, dvy, w_eff, l_eff):
    """
    k'(Dx,Dy,Dvx,Dvy) =
        { (1 - w/Dx) . sqrt( (Dx/tau(Dx,Dvx,tau2))^2 + (Dy/tau(Dy,Dvy,tau1))^2 ) }  if |Dy| <= l|Dx|/w
        { (1 - l/Dy) . sqrt( (Dx/tau(Dx,Dvx,tau1))^2 + (Dy/tau(Dy,Dvy,tau2))^2 ) }  otherwise

    base_tau: different tau per region, with tau1 = tau_x_base and tau2 = tau_y_base.

    With dipf.swap_wl_in_kprime: true, w and l trade places throughout:
        { (1 - l/Dx) . sqrt(...) }  if |Dy| <= w|Dx|/l
        { (1 - w/Dy) . sqrt(...) }  otherwise
    """
    if SWAP_WL:
        w_eff, l_eff = l_eff, w_eff

    eps = 1e-3
    abs_dx = abs(dx) if abs(dx) > eps else eps
    abs_dy = abs(dy) if abs(dy) > eps else eps

    if abs_dy <= (l_eff * abs_dx) / w_eff:
        term = (1.0 - (w_eff / abs_dx)) if abs_dx > w_eff else 0.1
        term = max(0.0001, term)
        tau_x = tau_component(dx, dvx, TAU_Y_BASE)   # tau2
        tau_y = tau_component(dy, dvy, TAU_X_BASE)   # tau1
    else:
        term = (1.0 - (l_eff / abs_dy)) if abs_dy > l_eff else 0.1
        term = max(0.0001, term)
        tau_x = tau_component(dx, dvx, TAU_X_BASE)   # tau1
        tau_y = tau_component(dy, dvy, TAU_Y_BASE)   # tau2

    tau_x = tau_x if abs(tau_x) > eps else eps
    tau_y = tau_y if abs(tau_y) > eps else eps

    kp = term * math.sqrt((dx / tau_x) ** 2 + (dy / tau_y) ** 2)

    return kp  # small additive floor so k' can never hit exactly 0


def U_dest(xh, yh, xd, yd):
    """Exact destination potential from Pure_DIPF_codes(8).py."""
    return math.sqrt(SX * (xh - xd) ** 2 + SY * (yh - yd) ** 2)


def _edge_wall(d):
    """
    1/d^2 wall term. Below d0 it continues as a steep straight line with the
    same slope instead of flattening out, so a body that has already
    crossed the limit (possible with a discrete 0.1 s step) is still pushed
    back rather than feeling zero force.
    """
    d0 = 1e-2
    if d >= d0:
        return 1.0 / (d * d)
    return 1.0 / (d0 * d0) + (2.0 / d0 ** 3) * (d0 - d)


def U_lane(yh, y_min, y_max):
    """
    Lane-boundary potential (Pure_DIPF_codes(8).py form), with distances
    measured from the ego's BODY edge (+ margin) to the road edge:

        U_lane = delta_lane * [ 1/(yh - y_min)^2 + 1/(y_max - yh)^2 ]

    where y_min / y_max come from road_bounds_y_safe_for_ego() and already
    include the ego's lateral half-extent and dipf.y_margin.
    """
    return DELTA_LANE * (_edge_wall(yh - y_min) + _edge_wall(y_max - yh))


def U_obs_exact(xh, yh, vxh, vyh, xo, yo, vxo, vyo, w_eff=2.0, l_eff=5.0):
    """
    U_o = lambda * (|Dv| / Dv_max) / k'(Dx,Dy,Dvx,Dvy),    |Dv| = hypot(Dvx, Dvy)

    or, with dipf.use_dv_numerator: false,

    U_o = lambda / k'(Dx,Dy,Dvx,Dvy)

    The numerator is used exactly as written: no projection onto the line
    of sight, no clamp to [0, 1], and no zeroing for a constant or growing
    gap. The only cut-off is the sensing window (obs_radius_ahead/behind,
    lateral_band), the same window build_obstacle_list() uses.
    """
    dx = xo - xh
    dy = yo - yh
    dvx = vxo - vxh
    dvy = vyo - vyh

    if dx > OBS_RADIUS_AHEAD or dx < -OBS_RADIUS_BEHIND:
        return 0.0
    if abs(dy) > LATERAL_BAND:
        return 0.0

    f_dv = math.hypot(dvx, dvy) / max(1e-6, DV_MAX) if USE_DV_NUMERATOR else 1.0

    w_combined = (EGO_WIDTH + w_eff) / 2.0
    l_combined = (EGO_LENGTH + l_eff) / 2.0

    kp = k_prime(dx, dy, dvx, dvy, w_combined, l_combined)

    return LAMBDA_OBS * f_dv / kp


def calculate_total_potential(x, y, vx, vy, obstacles, goal_x, goal_y):
    y_min, y_max, _, _ = road_bounds_y_safe_for_ego(ego_lateral_half_extent(vx, vy))

    total = W_DEST * U_dest(x, y, goal_x, goal_y)
    total += W_LANE * U_lane(y, y_min, y_max)

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


def report_potential_at_collision(x, y, vx, vy, reason, other_id=None):
    """
    Print the DIPF potential and its gradient at the ego state where a
    collision was detected, split into destination / lane / obstacle terms
    (and the colliding vehicle's own obstacle term when it is known).
    Gradients are the same central differences the controller uses
    (step dipf.grad_eps); the force on the ego is -grad.

    Also prints the ego's acceleration: the value the controller actually
    applied on its last step (ego_control_step.prev_ax/prev_ay, i.e. after
    the acceleration limits, if enabled) next to -grad U at this
    state (what the potential asks for, before any limits).
    """
    eps = GRAD_EPS
    goal_y = lane_centers_y()[GOAL_LANE_INDEX]
    obstacles = build_obstacle_list(x, y)
    y_min, y_max, _, _ = road_bounds_y_safe_for_ego(ego_lateral_half_extent(vx, vy))

    def grad(f):
        return ((f(x + eps, y) - f(x - eps, y)) / (2.0 * eps),
                (f(x, y + eps) - f(x, y - eps)) / (2.0 * eps))

    def f_dest(xx, yy):
        return W_DEST * U_dest(xx, yy, EGO_GOAL_X, goal_y)

    def f_lane(xx, yy):
        return W_LANE * U_lane(yy, y_min, y_max)

    def f_obs(xx, yy):
        return sum(W_OBS * U_obs_exact(xx, yy, vx, vy, o[0], o[1], o[2], o[3], w_eff=o[4], l_eff=o[5])
                   for o in obstacles)

    def f_total(xx, yy):
        return calculate_total_potential(xx, yy, vx, vy, obstacles, EGO_GOAL_X, goal_y)

    rows = [("destination", f_dest), ("lane", f_lane), (f"obstacles (n={len(obstacles)})", f_obs)]

    # The colliding vehicle's own term (it may sit outside the sensing
    # window, in which case U_obs_exact returns 0 for it).
    if other_id is not None and other_id in traci.vehicle.getIDList():
        typ = traci.vehicle.getTypeID(other_id)
        w_eff, l_eff = get_obstacle_dimensions(typ)
        xo, yo = vehicle_center(other_id, traci.vehicle.getLength(other_id))
        vxo = traci.vehicle.getSpeed(other_id)
        try:
            vyo = traci.vehicle.getLateralSpeed(other_id)
        except Exception:
            vyo = 0.0

        def f_other(xx, yy):
            return W_OBS * U_obs_exact(xx, yy, vx, vy, xo, yo, vxo, vyo, w_eff=w_eff, l_eff=l_eff)

        rows.append((f"{other_id} [{typ}] rel=({xo - x:+.2f}, {yo - y:+.2f})", f_other))

    U_tot = f_total(x, y)
    gx, gy = grad(f_total)

    print(f"[COLLISION POTENTIAL] t={traci.simulation.getTime():.1f}s  reason: {reason}")
    print(f"  ego centre=({x:.2f}, {y:.2f})  v=({vx:.2f}, {vy:.2f})")
    print(f"  accel applied (last step) : ax={ego_control_step.prev_ax:+.4f}  ay={ego_control_step.prev_ay:+.4f} m/s^2"
          f"   [acceleration limits: {'on' if LIMIT_ACCEL else 'OFF'}]")
    print(f"  accel asked (-grad U here): ax={-gx:+.4f}  ay={-gy:+.4f} m/s^2")
    print(f"  {'term':42s} {'U':>12s} {'dU/dx':>12s} {'dU/dy':>12s}")
    print(f"  {'TOTAL':42s} {U_tot:12.4f} {gx:12.4f} {gy:12.4f}   |grad U|={math.hypot(gx, gy):.4f}")
    for name, f in rows:
        gxi, gyi = grad(f)
        print(f"    {name:40s} {f(x, y):12.4f} {gxi:12.4f} {gyi:12.4f}")


# =============================================================================
# COLLISION / DISTANCE
# =============================================================================

def ego_collision_sumo():
    try:
        cols = traci.simulation.getCollisions()
    except Exception:
        return False, "", None

    for c in cols:
        collider = getattr(c, "collider", "")
        victim = getattr(c, "victim", "")

        if collider == EGO_ID or victim == EGO_ID:
            other = victim if collider == EGO_ID else collider
            return (True, f"SUMO collision: collider={collider}, victim={victim}", other)

    return False, "", None


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


def _axis_overlap_interval(d, dv, half_sum):
    """
    Time interval [t_in, t_out] during which |d + dv*t| < half_sum, i.e. the
    two bodies overlap on this axis under constant relative velocity.
    Returns None if they never overlap on this axis.
    """
    if abs(dv) < 1e-9:
        return (-math.inf, math.inf) if abs(d) < half_sum else None
    t1 = (-half_sum - d) / dv
    t2 = (half_sum - d) / dv
    return (min(t1, t2), max(t1, t2))


def time_to_collision(dx, dy, dvx, dvy, sum_length, sum_width):
    """
    Time until two axis-aligned vehicle boxes first overlap, assuming both
    keep their current velocities.

    dx, dy    -- obstacle - ego centre offset
    dvx, dvy  -- obstacle - ego velocity
    sum_*     -- ego + obstacle length / width (actual SUMO sizes)

    A collision needs overlap on BOTH axes at the same time, so the
    TTC is the start of the intersection of the two per-axis overlap
    intervals. Returns 0.0 if the boxes already overlap and inf if they
    never will (e.g. same velocity, or moving apart).
    """
    ix = _axis_overlap_interval(dx, dvx, sum_length / 2.0)
    iy = _axis_overlap_interval(dy, dvy, sum_width / 2.0)
    if ix is None or iy is None:
        return math.inf

    t_in = max(ix[0], iy[0])
    t_out = min(ix[1], iy[1])

    if t_in > t_out or t_out < 0.0:
        return math.inf
    return max(0.0, t_in)


def min_time_to_collision(ego_x, ego_y, ego_vx, ego_vy, obstacles):
    """Minimum TTC over all tracked obstacles (inf if none / none closing)."""
    best = math.inf
    for (xo, yo, vxo, vyo, w_eff, l_eff, w_actual, l_actual) in obstacles:
        ttc = time_to_collision(
            xo - ego_x, yo - ego_y, vxo - ego_vx, vyo - ego_vy,
            EGO_LENGTH + l_actual, EGO_WIDTH + w_actual,
        )
        best = min(best, ttc)
    return best


def append_collision_row(hist, t, x, y, vx=np.nan, vy=np.nan, vx_apply=np.nan,
                          vy_apply=np.nan, ax=np.nan, ay=np.nan, grad_norm=np.nan,
                          pot=np.nan):
    """
    Log the collision itself as one final row in hist, instead of the
    episode's CSV output simply stopping at the last successful step.
    closest_dist and ttc are forced to 0.0 -- by definition, a reported collision
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
            "ttc": 0.0,
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


# Per-vehicle constants (type, effective and actual size), looked up once per
# vehicle id instead of every step -- they never change for traffic vehicles.
_VEHICLE_STATIC = {}


def _vehicle_static(vid):
    info = _VEHICLE_STATIC.get(vid)
    if info is None:
        try:
            typ = traci.vehicle.getTypeID(vid)
        except Exception:
            typ = "default"

        w_eff, l_eff = get_obstacle_dimensions(typ)

        # Real physical dimensions (as SUMO actually models the vehicle),
        # used for the TTC / clearance checks and for converting SUMO's
        # front-bumper position to the body centre -- distinct from
        # w_eff/l_eff, which are the padded "effective" sizes used only to
        # shape the DIPF obstacle potential.
        try:
            l_actual = traci.vehicle.getLength(vid)
            w_actual = traci.vehicle.getWidth(vid)
        except Exception:
            l_actual, w_actual = l_eff, w_eff

        info = (w_eff, l_eff, w_actual, l_actual)
        _VEHICLE_STATIC[vid] = info
    return info


_CTX_VARS = (tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_SPEED_LAT)


def context_radius():
    """
    Radius (around the ego's FRONT bumper, where SUMO measures it) that is
    guaranteed to contain every vehicle the sensing window can accept: window
    reach + ego half-length (front -> centre) + the longest vehicle's
    half-length (its centre -> front) + 1 m slack.
    """
    longest = max([float(v["length_m"]) for v in CONFIG.get("vehicle_types", {}).values()] + [EGO_LENGTH])
    slack = 0.5 * EGO_LENGTH + 0.5 * longest + 1.0
    reach = max(OBS_RADIUS_AHEAD, OBS_RADIUS_BEHIND, ENCODING_RANGE if (LOG_ENCODING or ADAPTIVE) else 0.0)
    if ADAPTIVE and "obs_radius_ahead" in ADAPTIVE["params"]:
        reach = max(reach, float(ADAPTIVE["limits"]["obs_radius_ahead"][1]))     # largest radius it may set
    return math.hypot(reach + slack, LATERAL_BAND + slack)


def subscribe_obstacle_context():
    """
    One context subscription on the ego: after every simulation step SUMO
    sends position/angle/speed/lateral speed of all vehicles within
    context_radius() in a single message, instead of one TraCI round trip
    per value per vehicle. Called once per episode (setup_ego_after_spawn).
    """
    traci.vehicle.subscribeContext(EGO_ID, tc.CMD_GET_VEHICLE_VARIABLE, context_radius(), list(_CTX_VARS))


def build_obstacle_list(x, y):
    """
    Obstacles near the ego's body centre (x, y), all in body-centre coordinates.

    Values come from the ego's context subscription (see
    subscribe_obstacle_context); vehicles outside its radius cannot be in
    the sensing window. Vehicles are visited in getIDList() order, exactly
    as when every vehicle was queried one by one, so the list -- and every
    sum over it -- is identical.
    """
    obstacles = []

    ctx = traci.vehicle.getContextSubscriptionResults(EGO_ID) or {}
    ids = traci.vehicle.getIDList()

    for vid in ids:
        if vid == EGO_ID:
            continue

        r = ctx.get(vid)
        if r is None:
            continue

        w_eff, l_eff, w_actual, l_actual = _vehicle_static(vid)

        xf, yf = r[tc.VAR_POSITION]
        xo, yo = front_to_center(xf, yf, l_actual, r[tc.VAR_ANGLE])

        if (xo < x - OBS_RADIUS_BEHIND or xo > x + OBS_RADIUS_AHEAD):
            continue

        if abs(yo - y) > LATERAL_BAND:
            continue

        vxo = r[tc.VAR_SPEED]
        vyo = r[tc.VAR_SPEED_LAT]

        obstacles.append((xo, yo, vxo, vyo, w_eff, l_eff, w_actual, l_actual))

    return obstacles


def nearby_vehicles(x, y):
    """
    Every vehicle within ENCODING_RANGE (longitudinally) of the ego centre, from
    the context subscription: (vid, x_centre, y_centre, vx, vy, length, width),
    actual SUMO sizes -- used by the encoding.
    """
    out = []
    ctx = traci.vehicle.getContextSubscriptionResults(EGO_ID) or {}
    for vid, r in ctx.items():
        if vid == EGO_ID:
            continue
        w_eff, l_eff, w_actual, l_actual = _vehicle_static(vid)
        xf, yf = r[tc.VAR_POSITION]
        xo, yo = front_to_center(xf, yf, l_actual, r[tc.VAR_ANGLE])
        if abs(xo - x) > ENCODING_RANGE:
            continue
        out.append((vid, xo, yo, r[tc.VAR_SPEED], r[tc.VAR_SPEED_LAT], l_actual, w_actual))
    out.sort(key=lambda v: v[0])            # deterministic order
    return out


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

    # Fresh episode -- reset the last commanded acceleration (kept for the
    # collision report and state captures).
    ego_control_step.prev_ax = 0.0
    ego_control_step.prev_ay = 0.0

    # The ego's velocity is controller state, not something read back from
    # SUMO each step: SUMO's getLateralSpeed() only reports sideways motion
    # made by its own lane-change model (disabled above), so after a
    # moveToXY() it always returns 0 and the lateral velocity would be lost
    # every step. Seed it from SUMO once, at departure.
    ego_control_step.vx = traci.vehicle.getSpeed(EGO_ID)
    ego_control_step.vy = 0.0
    ego_control_step.angle = traci.vehicle.getAngle(EGO_ID)
    ego_control_step.adapt_z = None             # v13: smoothed exponents of the adaptive policy
    ego_control_step.adapt_base = None          # v13: base parameter values of this run

    subscribe_obstacle_context()


def adapt_parameters(x, y, vx, vy, dt):
    """
    v13: set this step's adapted DIPF parameters from the traffic around the ego
    (policy.py): p = base * exp(z), z smoothed towards clip(W f + b) with time
    constant ADAPTIVE["smooth_s"]. The first step starts at the target (no lag at
    the takeover). The base is the parameter value the run started with.
    """
    from encoding import encode
    import policy as pol
    g = globals()
    names = ADAPTIVE["params"]
    if ego_control_step.adapt_base is None:
        ego_control_step.adapt_base = {k: g[_DIPF_GLOBALS[k][0]] for k in names}
    nearby = nearby_vehicles(x, y)
    enc = encode((x, y, vx, vy), [v[1:] for v in nearby], road_bounds_y(), EGO_LENGTH, EGO_WIDTH)
    f = pol.features(enc)
    z_target = pol.target_exponents(ADAPTIVE, f)
    z = ego_control_step.adapt_z
    if z is None:
        z = z_target
    else:
        z = z + min(1.0, dt / max(ADAPTIVE["smooth_s"], 1e-9)) * (z_target - z)
    ego_control_step.adapt_z = z
    values = pol.apply(ADAPTIVE, ego_control_step.adapt_base, z)
    for k, v in values.items():
        g[_DIPF_GLOBALS[k][0]] = v
    if ADAPT_LOG:
        ADAPT_TRACE.append([traci.simulation.getTime()] + list(f) + [values[k] for k in names])


# =============================================================================
# ONE DIPF CONTROL STEP
# =============================================================================

def ego_control_step(dt, hist):
    # Position comes from SUMO (moveToXY placed it there last step); velocity
    # is the controller's own integrated state (see setup_ego_after_spawn).
    x, y = vehicle_center(EGO_ID, EGO_LENGTH)
    vx = ego_control_step.vx
    vy = ego_control_step.vy

    goal_y = lane_centers_y()[GOAL_LANE_INDEX]

    # -------------------------------------------------------------------------
    # Goal
    # -------------------------------------------------------------------------
    if success_goal_reached(x, y):
        lane_i_stop = nearest_lane_index(y)

        result = move_ego(EGO_GOAL_X, y, lane_i_stop, ego_control_step.angle)
        if result == "COLLISION":
            report_potential_at_collision(EGO_GOAL_X, y, vx, vy, "moveToXY collision at goal")
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
    # v13: adaptive parameters for this step (before the obstacle window, which uses obs_radius_ahead)
    if ADAPTIVE is not None:
        adapt_parameters(x, y, vx, vy, dt)

    obstacles = build_obstacle_list(x, y)

    # -------------------------------------------------------------------------
    # Pure DIPF gradient (unified, non-region-classified obstacle potential)
    # -------------------------------------------------------------------------
    dUx, dUy = numerical_grad_U_total((x, y, vx, vy), obstacles, EGO_GOAL_X, goal_y)

    # Pure potential-gradient acceleration, magnitude-clamped to the
    # vehicle's physical accel limits (unless ego.limit_acceleration: false).
    if LIMIT_ACCEL:
        ax_lo, ax_hi = -EGO_D_MAX, EGO_A_MAX
        ay_lo, ay_hi = -EGO_AY_MAX, EGO_AY_MAX
    else:
        ax_lo = ay_lo = -math.inf
        ax_hi = ay_hi = math.inf

    ax_raw = clamp(-dUx, ax_lo, ax_hi)
    ay_raw = clamp(-dUy, ay_lo, ay_hi)

    # No jerk limit (v9): the commanded acceleration is the clamped gradient.
    ax, ay = ax_raw, ay_raw

    if LOG_ENCODING:
        from encoding import encode
        nearby = nearby_vehicles(x, y)
        ENCODING_ROWS.append([traci.simulation.getTime()] + encode(
            (x, y, vx, vy), [v[1:] for v in nearby], road_bounds_y(), EGO_LENGTH, EGO_WIDTH))

    ego_control_step.prev_ax = ax
    ego_control_step.prev_ay = ay

    vx_apply = clamp(vx + ax * dt, 0.0, EGO_MAX_SPEED)
    vy_apply = clamp(vy + ay * dt, -EGO_VY_MAX, EGO_VY_MAX)

    # Sideways speed in proportion to forward speed (ego.limit_heading).
    if LIMIT_HEADING:
        vy_cap = vx_apply * TAN_MAX_HEADING
        vy_apply = clamp(vy_apply, -vy_cap, vy_cap)

    # Integrate the DIPF acceleration WITHOUT clamping to the road. If this
    # step would put any part of the vehicle off the road, that is a
    # collision, not something to silently correct.
    # Trapezoidal step on the (clamped) velocity: identical to
    # x + v*dt + 0.5*a*dt^2 when no speed clamp is active, and consistent
    # with the stored velocity when one is.
    x_new = x + 0.5 * (vx + vx_apply) * dt
    y_new = y + 0.5 * (vy + vy_apply) * dt

    grad_norm = float(math.hypot(dUx, dUy))

    if not ego_within_road(x_new, y_new):
        report_potential_at_collision(x_new, y_new, vx_apply, vy_apply, "road departure (ego body off the road)")
        append_collision_row(
            hist, traci.simulation.getTime(), x_new, y_new, vx=vx, vy=vy,
            vx_apply=vx_apply, vy_apply=vy_apply, ax=ax, ay=ay, grad_norm=grad_norm,
        )
        return "COLLISION"

    lane_i = nearest_lane_index(y_new)

    angle_new = velocity_angle(vx_apply, vy_apply, ego_control_step.angle)

    result = move_ego(x_new, y_new, lane_i, angle_new)
    if result == "COLLISION":
        report_potential_at_collision(x_new, y_new, vx_apply, vy_apply, "moveToXY collision")
        append_collision_row(
            hist, traci.simulation.getTime(), x_new, y_new, vx=vx, vy=vy,
            vx_apply=vx_apply, vy_apply=vy_apply, ax=ax, ay=ay, grad_norm=grad_norm,
        )
        return "COLLISION"

    traci.vehicle.setSpeed(EGO_ID, vx_apply)

    # Carry the velocity to the next step (only after a successful move).
    ego_control_step.vx = vx_apply
    ego_control_step.vy = vy_apply
    ego_control_step.angle = angle_new

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
    ttc = min_time_to_collision(x, y, vx, vy, obstacles)
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
            "ttc": float(ttc),
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


# Default controller state (setup_ego_after_spawn() resets these at the start
# of each episode; this covers the case of ego_control_step being called
# before that, defensively).
ego_control_step.prev_ax = 0.0
ego_control_step.prev_ay = 0.0
ego_control_step.vx = 0.0
ego_control_step.vy = 0.0
ego_control_step.angle = 90.0


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
    Scalar episode cost, lower is better. The collision term is TIME based:

        T            = cost.ttc_threshold_s
        t            = min(min TTC over the episode, T)
        C_collision  = (e^(1 - t/T) - 1) / (e - 1)      -- 0 if TTC never < T, 1 at TTC = 0
        C_dist       = 1 - progress
        C_time       = min(1, (final_time - start_time)*EGO_MAX_SPEED / (progress*ROAD_LENGTH_M*3))
        C            = w_collision*C_collision + w_dist*C_dist + w_time*C_time

    The minimum TTC is taken over every row, including a final collision
    row (whose ttc is 0.0), so an actual collision always gives
    C_collision = 1.
    """
    if not hist:
        return {"min_ttc": np.nan, "C_collision": np.nan, "C_dist": np.nan, "C_time": np.nan, "C": np.nan}

    ttc_vals = [r["ttc"] for r in hist if not np.isnan(r["ttc"])]
    min_ttc = min(ttc_vals) if ttc_vals else math.inf

    t = min(min_ttc, TTC_THRESHOLD)
    c_collision = (math.exp(1.0 - t / TTC_THRESHOLD) - 1.0) / (math.e - 1.0)

    final = hist[-1]
    progress = final["progress"]
    c_dist = 1.0 - progress

    start_time = hist[0]["time"]
    final_time = final["time"]
    safe_progress = progress if abs(progress) > 1e-9 else 1e-9
    length = SEGMENT_LENGTH_M if SEGMENT_LENGTH_M is not None else ROAD_LENGTH_M
    c_time = min(1.0, (final_time - start_time) * EGO_MAX_SPEED / (safe_progress * length * 3.0))

    c_total = W_COST_COLLISION * c_collision + W_COST_DIST * c_dist + W_COST_TIME * c_time

    return {
        "min_ttc": float(min_ttc),
        "C_collision": float(c_collision),
        "C_dist": float(c_dist),
        "C_time": float(c_time),
        "C": float(c_total),
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(hist, done_reason):
    results_root = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results")))
    results_root.mkdir(parents=True, exist_ok=True)

    run_name = results_run_name()
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
        "ttc",
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

    if LOG_ENCODING and ENCODING_ROWS:
        from encoding import FEATURES as ENC_FEATURES
        with (results_dir / "encoding.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time"] + ENC_FEATURES)
            w.writerows(ENCODING_ROWS)

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)


    return metrics


# =============================================================================
# RUN
# =============================================================================

def results_run_name():
    if RUN_NAME is not None:
        return RUN_NAME
    return f"results_{int(SEED)}" if SEED is not None else "results_random"


def restore_ego_from_episode():
    """Continue the controller exactly where the capture left it."""
    ego_control_step.vx = float(EPISODE_EGO["vx"])
    ego_control_step.vy = float(EPISODE_EGO["vy"])
    ego_control_step.prev_ax = float(EPISODE_EGO["prev_ax"])
    ego_control_step.prev_ay = float(EPISODE_EGO["prev_ay"])
    ego_control_step.angle = float(EPISODE_EGO["heading_deg"])


def capture_state(t_requested, t):
    """
    Save the full SUMO state (all vehicles, routes, flows, RNG) plus the
    controller-held ego state at simulation time t. Called before
    ego_control_step at that step, so the ego JSON holds exactly the state
    the controller would continue from.
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"t{t:.1f}"
    state_file = CAPTURE_DIR / f"state_{stem}.xml"
    traci.simulation.saveState(str(state_file))

    x_front, y_front = traci.vehicle.getPosition(EGO_ID)
    x_c, y_c = vehicle_center(EGO_ID, EGO_LENGTH)
    ego = {
        "time": round(t, 6),
        "time_requested": round(t_requested, 6),
        "seed": SEED,
        "ego_id": EGO_ID,
        "state_file": state_file.name,
        "centre_x": x_c,
        "centre_y": y_c,
        "front_x": x_front,
        "front_y": y_front,
        "lane_index": traci.vehicle.getLaneIndex(EGO_ID),
        "lane_position": traci.vehicle.getLanePosition(EGO_ID),
        "lateral_lane_position": traci.vehicle.getLateralLanePosition(EGO_ID),
        "sumo_speed": traci.vehicle.getSpeed(EGO_ID),
        "sumo_angle": traci.vehicle.getAngle(EGO_ID),
        # controller-held state (not in the SUMO state file)
        "vx": ego_control_step.vx,
        "vy": ego_control_step.vy,
        "prev_ax": ego_control_step.prev_ax,
        "prev_ay": ego_control_step.prev_ay,
        "heading_deg": ego_control_step.angle,
        "length_m": EGO_LENGTH,
        "width_m": EGO_WIDTH,
        "n_vehicles": len(traci.vehicle.getIDList()),
        "dipf": {k: v for k, v in DIPF.items()},
    }
    if LOG_ENCODING:
        # v11: the traffic encoding at the capture moment = the model input for this scenario
        from encoding import encode
        nearby = nearby_vehicles(x_c, y_c)
        ego["encoding"] = encode((x_c, y_c, ego_control_step.vx, ego_control_step.vy),
                                 [v[1:] for v in nearby], road_bounds_y(), EGO_LENGTH, EGO_WIDTH)
    with (CAPTURE_DIR / f"ego_{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(ego, f, indent=2)
    print(f"[CAPTURE] t={t:.1f}s (requested {t_requested:.1f}s) -> {state_file}")


# ---- fast, race-free SUMO start for in-process runs (run_episode) -----------
#
# sumolib's getFreeSocketPort() only *finds* a free port (bind to port 0,
# close); with many workers starting SUMO several times a second, two workers
# occasionally get the same port, one of them connects to the OTHER worker's
# SUMO (evaluating its parameters on the wrong episode) and later waits forever
# for its own SUMO, which sits in accept(). To make that impossible, every
# process claims an exclusive slot (flock on a lock file) that owns a block of
# PORTS_PER_SLOT ports below the OS's ephemeral range (which starts at 32768),
# so no other process -- ours or the OS's "any free port" -- can be handed them.
PORT_BASE = 10000
PORTS_PER_SLOT = 256
N_PORT_SLOTS = 88                  # 10000 + 88*256 = 32528 < 32768
SUMO_SOCKET_TIMEOUT = 60.0         # s; a TraCI call that takes longer raises instead of hanging
_PORT_SLOT = None                  # (pid, slot, fd)
_PORT_COUNTER = 0
_FAST_PROC = None


def _port_slot():
    global _PORT_SLOT
    if _PORT_SLOT is not None and _PORT_SLOT[0] == os.getpid():
        return _PORT_SLOT[1]
    import fcntl
    import tempfile
    lock_dir = Path(tempfile.gettempdir()) / "dipf_sumo_port_slots"
    lock_dir.mkdir(exist_ok=True)
    for slot in range(N_PORT_SLOTS):
        fd = os.open(str(lock_dir / f"slot_{slot}.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)     # held for this process's lifetime
        except OSError:
            os.close(fd)
            continue
        _PORT_SLOT = (os.getpid(), slot, fd)
        return slot
    raise RuntimeError(f"all {N_PORT_SLOTS} SUMO port slots are in use")


def _next_port():
    """Next port of this process's own block that can be bound right now."""
    global _PORT_COUNTER
    import socket
    base = PORT_BASE + _port_slot() * PORTS_PER_SLOT
    for _ in range(PORTS_PER_SLOT):
        port = base + (_PORT_COUNTER % PORTS_PER_SLOT)
        _PORT_COUNTER += 1
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError(f"no bindable port in {base}..{base + PORTS_PER_SLOT - 1}")


def _kill_with_parent():
    """
    preexec_fn (Linux), runs in the child just before SUMO is exec'd:
      - SIGKILL this SUMO if the Python process that started it dies (PR_SET_PDEATHSIG);
      - v13: disable address-space layout randomisation for it (= setarch -R). SUMO
        orders some internal work by memory address, so with ASLR the SAME run (state,
        seed, commands) draws its random numbers in a different order from launch to
        launch -- other vehicles' speed noise moves between vehicles and an identical
        segment run can end in success one time and in a collision the next. Without
        ASLR every run is exactly repeatable, so all candidates really see the same
        futures.
    """
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)                                     # PR_SET_PDEATHSIG
        ADDR_NO_RANDOMIZE = 0x0040000
        libc.personality(libc.personality(0xFFFFFFFF) | ADDR_NO_RANDOMIZE)
    except Exception:
        pass


def _drop_connection():
    """Forget any 'default' TraCI connection without ever blocking on it."""
    try:
        conn = traci.getConnection("default")
    except Exception:
        return
    try:
        conn.close(wait=False)
    except Exception:
        pass
    for key in ("default", ""):
        if traci.connection._connections.get(key) is conn:
            del traci.connection._connections[key]
    sock = getattr(conn, "_socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
    proc = getattr(conn, "_process", None)
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait()


def start_sumo_fast(cmd):
    """
    Same as traci.start(cmd) (fresh SUMO process, connection under the default
    label) but: polls the port every 10 ms instead of sleeping 1 s, uses this
    process's own port block (no cross-process port races), puts a timeout on
    every TraCI call, and makes SUMO die with its parent process.
    """
    global _FAST_PROC
    _drop_connection()
    port = _next_port()
    # start_new_session: Ctrl+C in the terminal goes to the whole process group;
    # SUMO must not get it (it would die mid-run and the run would "crash").
    # It still dies with its parent through _kill_with_parent (PR_SET_PDEATHSIG).
    # v14: identical command line AND environment for identical runs, whichever folder they are started
    # from -- SUMO's internal ordering (and so its random draws) depends on its memory layout, which
    # depends on argv/env string lengths. So: run in the network's folder with a relative network path
    # (episode mode), and drop the variables that hold the caller's directory.
    env = {k: v for k, v in os.environ.items() if k not in ("PWD", "OLDPWD", "_")}
    proc = subprocess.Popen(cmd + ["--remote-port", str(port)],
                            stdout=subprocess.DEVNULL if QUIET else None,
                            preexec_fn=_kill_with_parent, start_new_session=True,
                            cwd=str(NET_FILE.parent), env=env)
    try:
        traci.main.connect(port, numRetries=2000, proc=proc, waitBetweenRetries=0.01, label="default")
        traci.switch("default")
        traci.getConnection("default")._socket.settimeout(SUMO_SOCKET_TIMEOUT)
    except BaseException:
        proc.kill()
        proc.wait()
        _drop_connection()
        raise
    _FAST_PROC = proc


def close_sumo_fast():
    """Close the connection and make sure the SUMO process is gone (never blocks for long)."""
    global _FAST_PROC
    try:
        traci.close(wait=False)
    except Exception:
        _drop_connection()
    proc, _FAST_PROC = _FAST_PROC, None
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def convert_to_ego(vid):
    """Turn traffic vehicle `vid` into the ego (convert spawn mode)."""
    global EGO_ID, EGO_START_X
    old_type = traci.vehicle.getTypeID(vid)
    traci.vehicle.setType(vid, EGO_TYPE_ID)     # ego size / limits; front bumper stays where it is
    EGO_ID = vid
    EGO_START_X = vehicle_center(vid, EGO_LENGTH)[0]
    print(f"Converted {vid} ({old_type}) into the ego at t={traci.simulation.getTime():.1f}s, "
          f"x={EGO_START_X:.1f} m")


def run_simulation():
    global EGO_ID
    EGO_ID = EGO_ID_CONFIG                     # fresh run (in-process runs reuse this module)
    if EPISODE_DIR is not None and EPISODE_EGO is not None:
        EGO_ID = EPISODE_EGO.get("ego_id", EGO_ID_CONFIG)    # a converted ego keeps its traffic id
    latest_candidate = None
    hist = []
    ENCODING_ROWS.clear()
    ego_initialized = False
    done_reason = "timeout"

    max_steps = max(1, int(math.ceil((SIM_END - SIM_BEGIN) / STEP_LENGTH)))
    if EPISODE_DIR is not None:
        max_steps = int(math.ceil(EPISODE_MAX_TIME / STEP_LENGTH)) + 1
    elif MAX_SIM_TIME is not None:
        start = CONVERT_TIME if SPAWN_MODE == "convert" else EGO_DEPART_TIME
        max_steps = int(math.ceil((start + MAX_SIM_TIME - SIM_BEGIN) / STEP_LENGTH))

    sumo_binary = sumolib.checkBinary("sumo-gui" if SUMO_GUI else "sumo")

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

    if EPISODE_DIR is not None:
        # Start from the captured state instead of the .sumocfg (whose route
        # file would add a second copy of the flows). Same processing
        # options as the generated .sumocfg.
        cmd = [
            sumo_binary,
            "-n", NET_FILE.name,               # v14: relative, SUMO runs in NET_FILE's folder (see start_sumo_fast)
            "--load-state", str(EPISODE_DIR / "state.xml"),
            "--step-length", str(STEP_LENGTH),
            "--lateral-resolution", str(LATERAL_RESOLUTION),
            "--collision.action", "warn",
            "--collision.mingap-factor", "0",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
        ]
        if EPISODE_SUMO_SEED is not None:
            cmd += ["--seed", str(int(EPISODE_SUMO_SEED))]

    if CAPTURE_TIMES:
        # RNG state in the snapshot -> traffic after it replays identically;
        # 6 decimals instead of SUMO's default 2 for positions/speeds.
        cmd += ["--save-state.rng", "true", "--save-state.precision", "6"]
    pending_captures = list(CAPTURE_TIMES)

    print("=" * 90)
    print("FULL PURE-DIPF CONTROLLER (v13)")
    print("=" * 90)
    print(f"Config : {CFG_FILE}")
    print(f"Network: {NET_FILE}")
    print(f"Goal   : x={EGO_GOAL_X:.1f} m, lane={GOAL_LANE_INDEX + 1}")
    print(f"Step   : {STEP_LENGTH:.3f} s")
    print(f"End    : {SIM_END:.1f} s")
    print(f"U_obs  : {'lambda*(|dv|/dv_max)/k' if USE_DV_NUMERATOR else 'lambda/k'}'")
    print(f"k'     : {'w/l swapped' if SWAP_WL else 'as in report'}")
    print(f"Limits : acceleration={'on' if LIMIT_ACCEL else 'OFF'} (+{EGO_A_MAX:g} / -{EGO_D_MAX:g} m/s^2), "
          f"heading={'<= %.1f deg (|vy| <= vx*tan)' % EGO_MAX_HEADING_DEG if LIMIT_HEADING else 'OFF'}")
    print(f"Seed   : {SEED if SEED is not None else '(none -- random traffic each run)'}")
    if EPISODE_DIR is not None:
        print(f"Episode: {EPISODE_DIR}  (t0={EPISODE_EGO['time']:.1f}s, start x={EGO_START_X:.2f}, "
              f"target x={EGO_GOAL_X:.2f} = +{SEGMENT_LENGTH_M:.1f} m, max {EPISODE_MAX_TIME:.0f} s)")
    if CAPTURE_TIMES:
        print(f"Capture: {', '.join(f'{c:.1f}' for c in CAPTURE_TIMES)} s -> {CAPTURE_DIR}")
    print("=" * 90)

    if QUIET:
        cmd += ["--no-warnings", "true"]

    try:
        if FAST_START:
            start_sumo_fast(cmd)
        else:
            traci.start(cmd, stdout=subprocess.DEVNULL if QUIET else None)

        # Episode mode: the capture was taken BEFORE the controller acted at
        # that step, so act first and only then advance SUMO.
        skip_first_step = EPISODE_DIR is not None

        for _ in range(max_steps):
            stepped = not skip_first_step
            if skip_first_step:
                skip_first_step = False
            else:
                traci.simulationStep()

            t = traci.simulation.getTime()

            # Check actual SUMO collision (only after a step: right after
            # loading a state nothing has moved, and a reused SUMO would still
            # report the previous episode's last collision).
            hit, msg, other_id = ego_collision_sumo() if stepped else (False, "", None)

            if hit:
                print(msg)
                if EGO_ID in traci.vehicle.getIDList():
                    ex, ey = vehicle_center(EGO_ID, EGO_LENGTH)
                    # Controller-held velocity (SUMO's lateral speed is 0
                    # for a moveToXY-driven ego).
                    evx = ego_control_step.vx
                    evy = ego_control_step.vy
                    report_potential_at_collision(ex, ey, evx, evy, msg, other_id=other_id)
                    append_collision_row(hist, t, ex, ey, vx=evx, vy=evy)
                done_reason = "collision"
                break

            vehicle_ids = traci.vehicle.getIDList()

            # Convert mode: remember the latest inserted vehicle of the chosen
            # type; at convert_time_s turn it into the ego.
            if SPAWN_MODE == "convert" and EPISODE_DIR is None and not ego_initialized:
                for vid in traci.simulation.getDepartedIDList():
                    if CONVERT_TYPE is None or traci.vehicle.getTypeID(vid) == CONVERT_TYPE:
                        latest_candidate = vid
                if t >= CONVERT_TIME - 1e-9:
                    if CONVERT_PICK == "latest":
                        pick = latest_candidate if latest_candidate in vehicle_ids else None
                    else:
                        cands = [v for v in vehicle_ids
                                 if CONVERT_TYPE is None or traci.vehicle.getTypeID(v) == CONVERT_TYPE]
                        dist = {v: abs(vehicle_center(v, traci.vehicle.getLength(v))[0] - CONVERT_X) for v in cands}
                        pick = min(cands, key=lambda v: (dist[v], v)) if cands else None
                    if pick is not None:
                        convert_to_ego(pick)

            # Detect ego after departure.
            if not ego_initialized and EGO_ID in vehicle_ids:
                setup_ego_after_spawn()
                if EPISODE_DIR is not None:
                    restore_ego_from_episode()
                ego_initialized = True
                print(f"Ego spawned at {t:.1f}s")

            # State capture (v7): at/after each requested time, before the
            # controller acts this step. Half-step tolerance absorbs float
            # rounding of the step grid.
            if pending_captures and ego_initialized and EGO_ID in traci.vehicle.getIDList():
                while pending_captures and t >= pending_captures[0] - 0.5 * STEP_LENGTH:
                    capture_state(pending_captures.pop(0), t)
                if CAPTURE_STOP and not pending_captures:
                    done_reason = "captured"
                    break

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
        if FAST_START:
            close_sumo_fast()
        else:
            try:
                traci.close()
            except Exception:
                pass

    metrics = save_results(hist, done_reason)

    run_name = results_run_name()
    output_root = Path(os.environ.get("DIPF_OUTPUT_DIR", str(BASE_DIR / "results")))
    run_dir = output_root / run_name

    print("\n" + "=" * 90)
    print(f"SIMULATION COMPLETE | status={done_reason}")
    print(f"Trajectory: {run_dir / 'trajectory.csv'}")
    print(f"Metrics   : {run_dir / 'metrics.json'}")
    print("=" * 90)

    return metrics


# dipf.* key -> module global it feeds (see PURE-DIPF PARAMETERS above).
_DIPF_GLOBALS = {
    "lambda_obs": ("LAMBDA_OBS", float), "alpha": ("ALPHA", float),
    "tau_x_base": ("TAU_X_BASE", float), "tau_y_base": ("TAU_Y_BASE", float),
    "sx": ("SX", float), "sy": ("SY", float), "delta_lane": ("DELTA_LANE", float),
    "w_dest": ("W_DEST", float), "w_obs": ("W_OBS", float), "w_lane": ("W_LANE", float),
    "dv_max": ("DV_MAX", float), "use_dv_numerator": ("USE_DV_NUMERATOR", bool),
    "swap_wl_in_kprime": ("SWAP_WL", bool),
    "obs_radius_ahead": ("OBS_RADIUS_AHEAD", float), "obs_radius_behind": ("OBS_RADIUS_BEHIND", float),
    "lateral_band": ("LATERAL_BAND", float), "grad_eps": ("GRAD_EPS", float), "y_margin": ("Y_MARGIN", float),
}
_DIPF_DEFAULTS = {k: DIPF[k] for k in _DIPF_GLOBALS if k in DIPF}


def set_dipf_params(params):
    """
    Set dipf.* values for the next in-process run: config.yaml values (as
    loaded) overridden by `params` -- the same result as DIPF_PARAMS for a
    fresh process.
    """
    unknown = set(params) - set(_DIPF_GLOBALS)
    if unknown:
        raise KeyError(f"unknown dipf.* keys: {sorted(unknown)}")
    values = {**_DIPF_DEFAULTS, **params}
    g = globals()
    for key, (name, cast) in _DIPF_GLOBALS.items():
        if key in values:
            g[name] = cast(values[key])
            DIPF[key] = values[key]


def run_episode(episode_dir, params, distance=150.0, max_time=60.0, output_dir=None, sumo_seed=None, policy=None):
    """
    In-process episode evaluation (used by optimize_episodes.py). Same run as

        DIPF_PARAMS=<params> python dipf_controller.py --episode DIR --distance D --max-time T

    without starting a new Python process (imports and config are reused),
    with a fast SUMO start, and without printing anything. Each call still
    gets a fresh SUMO, so results are identical to the command above.
    Returns the metrics dict that save_results() also writes.
    """
    global EPISODE_DIR, EPISODE_EGO, EPISODE_MAX_TIME, SEED, EGO_START_X, EGO_GOAL_X
    global SEGMENT_LENGTH_M, RUN_NAME, FAST_START, QUIET, DEBUG_INTERVAL, EPISODE_SUMO_SEED, ADAPTIVE

    set_dipf_params(params)
    ADAPTIVE = policy                    # v13: adaptive policy (None = fixed parameters)
    ADAPT_TRACE.clear()
    EPISODE_SUMO_SEED = sumo_seed
    EPISODE_DIR = Path(episode_dir).resolve()
    EPISODE_EGO = json.loads((EPISODE_DIR / "ego.json").read_text(encoding="utf-8"))
    EPISODE_MAX_TIME = float(max_time)
    SEED = EPISODE_EGO.get("seed")
    EGO_START_X = float(EPISODE_EGO["centre_x"])
    EGO_GOAL_X = min(EGO_START_X + float(distance), ROAD_LENGTH_M - GOAL_X_OFFSET)
    SEGMENT_LENGTH_M = EGO_GOAL_X - EGO_START_X
    RUN_NAME = f"results_{EPISODE_DIR.name}"
    FAST_START, QUIET, DEBUG_INTERVAL = True, True, 0.0
    if output_dir is not None:
        os.environ["DIPF_OUTPUT_DIR"] = str(output_dir)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return run_simulation()


_NORMAL_MODE_DEFAULTS = {"EGO_START_X": EGO_START_X, "EGO_GOAL_X": EGO_GOAL_X}


def run_seed(seed, params, output_dir=None, max_time=300.0):
    """
    In-process FULL-ROAD run (normal mode: ego departs from the route file,
    goal at road_length - goal_x_offset), the equivalent of

        DIPF_PARAMS=<params> python dipf_controller.py --seed SEED

    with a fresh SUMO (fast, race-free start), nothing printed, and the run
    stopped `max_time` seconds after the ego's departure (None = simulation.end).
    Returns the metrics dict that save_results() also writes.
    """
    global EPISODE_DIR, EPISODE_EGO, SEGMENT_LENGTH_M, RUN_NAME, SEED, EGO_START_X, EGO_GOAL_X
    global MAX_SIM_TIME, FAST_START, QUIET, DEBUG_INTERVAL, ADAPTIVE

    set_dipf_params(params)
    ADAPTIVE = None
    EPISODE_DIR, EPISODE_EGO, SEGMENT_LENGTH_M, RUN_NAME = None, None, None, None
    SEED = int(seed)
    EGO_START_X = _NORMAL_MODE_DEFAULTS["EGO_START_X"]     # convert mode: replaced at conversion
    EGO_GOAL_X = _NORMAL_MODE_DEFAULTS["EGO_GOAL_X"]
    MAX_SIM_TIME = None if max_time is None else float(max_time)
    FAST_START, QUIET, DEBUG_INTERVAL = True, True, 0.0
    if output_dir is not None:
        os.environ["DIPF_OUTPUT_DIR"] = str(output_dir)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return run_simulation()


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Pure-DIPF episode in SUMO.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="SUMO traffic seed (default: simulation.seed from config.yaml)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="run with sumo-gui instead of headless sumo",
    )
    parser.add_argument(
        "--capture-times", type=str, default="",
        help="comma-separated simulation times (s) at which to save the full state",
    )
    parser.add_argument(
        "--capture-dir", type=str, default=None,
        help="directory for captured states (default: ./captures/seed_<seed>)",
    )
    parser.add_argument(
        "--log-encoding", action="store_true",
        help="write the laneless traffic encoding (encoding.py) for every step to encoding.csv",
    )
    parser.add_argument(
        "--episode", type=str, default=None,
        help="episode folder (state.xml + ego.json) to start from",
    )
    parser.add_argument(
        "--distance", type=float, default=150.0,
        help="episode mode: metres the ego must cover from its start (default 150)",
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0,
        help="episode mode: timeout in seconds after the state's time (default 60)",
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="v13: JSON file with an adaptive policy (policy.json from step 2, key 'policy')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        SEED = args.seed
    SUMO_GUI = args.gui
    LOG_ENCODING = args.log_encoding
    if args.capture_times:
        CAPTURE_TIMES = sorted(float(c) for c in args.capture_times.split(",") if c.strip())
        CAPTURE_DIR = Path(args.capture_dir) if args.capture_dir else BASE_DIR / "captures" / f"seed_{SEED}"
    if args.policy:
        _pj = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        ADAPTIVE = _pj.get("policy", _pj)
        print(f"Adaptive policy from {args.policy}: adapts {ADAPTIVE['params']}")
    if args.episode:
        EPISODE_DIR = Path(args.episode).resolve()
        EPISODE_EGO = json.loads((EPISODE_DIR / "ego.json").read_text(encoding="utf-8"))
        EPISODE_MAX_TIME = float(args.max_time)
        SEED = EPISODE_EGO.get("seed")
        EGO_START_X = float(EPISODE_EGO["centre_x"])
        EGO_GOAL_X = min(EGO_START_X + float(args.distance), ROAD_LENGTH_M - GOAL_X_OFFSET)
        SEGMENT_LENGTH_M = EGO_GOAL_X - EGO_START_X
        RUN_NAME = f"results_{EPISODE_DIR.name}"
    run_simulation()
