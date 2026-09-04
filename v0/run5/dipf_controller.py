import os
import sys
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS / BASIC SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

NET_NAME = "straight3lane"

NET_FILE = BASE_DIR / f"{NET_NAME}.net.xml"
ROU_FILE = BASE_DIR / f"{NET_NAME}.rou.xml"
CFG_FILE = BASE_DIR / f"{NET_NAME}.sumocfg"

ROAD_LENGTH_M = 1000.0

LANES = 3
LANE_WIDTH_M = 3.5
SPEED_MPS = 20.0

SIM_BEGIN = 0.0
SIM_END = 3600.0
STEP_LENGTH = 0.1

LATERAL_RESOLUTION = 0.25
DEBUG_INTERVAL = 2.0


# =============================================================================
# EGO SETTINGS
# =============================================================================

EGO_ID = "ego"

EGO_DEPART_TIME = 20.0

EGO_START_X = 1.0
EGO_START_LANE = 2

GOAL_LANE_INDEX = 0

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


# =============================================================================
# STUCK DETECTION
# =============================================================================

STUCK_SPEED_THRESH = 0.35
STUCK_WINDOW_SEC = 8.0
STUCK_PROGRESS_THRESH = 2.0


# =============================================================================
# ROBUST moveToXY SETTINGS
# =============================================================================

MOVE_MATCH_THRESHOLD = 1000.0
MOVE_KEEP_ROUTE = 2

X_MIN_SAFE = 0.5
X_MAX_SAFE = ROAD_LENGTH_M - 0.5


# =============================================================================
# SUMO IMPORT
# =============================================================================

def ensure_sumo_tools():

    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME is not set.\n"
            "Set it to your SUMO installation folder."
        )

    tools = Path(
        os.environ["SUMO_HOME"]
    ) / "tools"

    if str(tools) not in sys.path:
        sys.path.append(str(tools))


ensure_sumo_tools()

import sumolib
import traci
import traci.constants as tc


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def clamp(x, lo, hi):

    return (
        lo
        if x < lo
        else hi
        if x > hi
        else x
    )


def lane_centers_y():

    centers = []

    mid = (LANES - 1) / 2.0

    for i in range(LANES):
        centers.append(
            (i - mid) * LANE_WIDTH_M
        )

    return centers


def road_bounds_y():

    half = (
        LANES * LANE_WIDTH_M
    ) / 2.0

    return -half, +half


def road_bounds_y_safe_for_ego():

    yL, yR = road_bounds_y()

    half_ego = 0.5 * EGO_W

    buf = 0.05

    y_min = (
        yL
        + half_ego
        + Y_MARGIN
        + buf
    )

    y_max = (
        yR
        - half_ego
        - Y_MARGIN
        - buf
    )

    return y_min, y_max, yL, yR


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


def clamp_x_to_road(x):

    return clamp(
        x,
        X_MIN_SAFE,
        X_MAX_SAFE,
    )


def project_to_lane_center(y):

    lane_i = nearest_lane_index(y)

    centers = lane_centers_y()

    return lane_i, centers[lane_i]


def success_goal_reached(x, y):

    reached_x = (
        abs(x - EGO_GOAL_X)
        <= GOAL_TOL_X
    )

    if SUCCESS_ANY_LANE:
        return reached_x

    goal_y = (
        lane_centers_y()
        [GOAL_LANE_INDEX]
    )

    return (
        reached_x
        and
        abs(y - goal_y)
        <= GOAL_TOL_Y
    )


def final_lane_name(y):

    return (
        f"Lane "
        f"{nearest_lane_index(y) + 1}"
    )


# =============================================================================
# ROBUST EGO MOVEMENT
# =============================================================================

def safe_move_ego(
    x_cmd,
    y_cmd,
    lane_i_hint=None,
):

    x_cmd = clamp_x_to_road(
        x_cmd
    )

    y_min, y_max, _, _ = (
        road_bounds_y_safe_for_ego()
    )

    y_cmd = clamp(
        y_cmd,
        y_min,
        y_max,
    )

    if lane_i_hint is None:
        lane_i_hint = nearest_lane_index(
            y_cmd
        )

    invalid_angle = (
        tc.INVALID_DOUBLE_VALUE
    )

    # First attempt:
    # use the DIPF-commanded lane and y.
    try:

        traci.vehicle.moveToXY(
            EGO_ID,
            "e0",
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

    # Second attempt:
    # project to nearest lane center.
    lane_i2, y2 = (
        project_to_lane_center(y_cmd)
    )

    try:

        traci.vehicle.moveToXY(
            EGO_ID,
            "e0",
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

    # Final fallback:
    # retain current lane and position.
    try:

        x_cur, y_cur = (
            traci.vehicle.getPosition(
                EGO_ID
            )
        )

        x3 = clamp_x_to_road(
            x_cur
        )

        lane_i3, y3 = (
            project_to_lane_center(
                y_cur
            )
        )

        traci.vehicle.moveToXY(
            EGO_ID,
            "e0",
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
# DIPF: TIME-SCALE FUNCTION
# =============================================================================

def tau(
    delta_pos,
    delta_v,
    base,
):

    z = (
        delta_pos
        * delta_v
    )

    scale = 0.5 * (
        (1.0 + ALPHA * delta_v)
        +
        (ALPHA * delta_v - 1.0)
        * math.tanh(z)
    )

    return base * scale


# =============================================================================
# DIPF: EXACT k' TERM
# =============================================================================

def k_prime_exact(
    dx,
    dy,
    dvx,
    dvy,
    w_eff,
    l_eff,
):

    tau_x = tau(
        dx,
        dvx,
        TAU_X_BASE,
    )

    tau_y = tau(
        dy,
        dvy,
        TAU_Y_BASE,
    )

    eps = 1e-3

    abs_dx = (
        abs(dx)
        if abs(dx) > eps
        else eps
    )

    abs_dy = (
        abs(dy)
        if abs(dy) > eps
        else eps
    )

    if abs_dy <= (
        l_eff * abs_dx
    ) / w_eff:

        term1 = (
            1.0 - w_eff / abs_dx
            if abs_dx > w_eff
            else 0.1
        )

        term1 = max(
            0.1,
            term1,
        )

        kp = (
            term1
            *
            math.sqrt(
                (dx / tau_x) ** 2
                +
                (dy / tau_y) ** 2
            )
        )

    else:

        term2 = (
            1.0 - l_eff / abs_dy
            if abs_dy > l_eff
            else 0.1
        )

        term2 = max(
            0.1,
            term2,
        )

        kp = (
            term2
            *
            math.sqrt(
                (dx / tau_x) ** 2
                +
                (dy / tau_y) ** 2
            )
        )

    return kp + 0.05


# =============================================================================
# DIPF: DESTINATION POTENTIAL
# =============================================================================

def U_dest(
    xh,
    yh,
    xd,
    yd,
):

    return math.sqrt(
        SX * (xh - xd) ** 2
        +
        SY * (yh - yd) ** 2
    )


# =============================================================================
# DIPF: ROAD / LANE POTENTIAL
# =============================================================================

def U_lane(
    yh,
    y_left,
    y_right,
):

    eps = 1e-3

    return DELTA_LANE * (
        (
            1.0
            /
            max(
                eps,
                yh - y_left,
            )
        ) ** 2
        +
        (
            1.0
            /
            max(
                eps,
                y_right - yh,
            )
        ) ** 2
    )


# =============================================================================
# DIPF: REGION CLASSIFICATION
# =============================================================================

def classify_region(
    dx,
    dy,
):

    if abs(dx) >= abs(dy):

        if dx >= 0.0:
            return "front"

        return "rear"

    if dy >= 0.0:
        return "left"

    return "right"


# =============================================================================
# DIPF: CLOSING SPEED
# =============================================================================

def compute_closing_speed_by_region(
    dx,
    dy,
    dvx,
    dvy,
    region,
):

    if region == "front":
        return max(
            0.0,
            -dvx,
        )

    if region == "rear":
        return max(
            0.0,
            dvx,
        )

    if region == "left":
        return max(
            0.0,
            -dvy,
        )

    if region == "right":
        return max(
            0.0,
            dvy,
        )

    return 0.0


# =============================================================================
# DIPF: REGION-AWARE CLOSING FUNCTION
# =============================================================================

def f_closing_region_aware(
    dx,
    dy,
    dvx,
    dvy,
):

    region = classify_region(
        dx,
        dy,
    )

    closing_speed = (
        compute_closing_speed_by_region(
            dx,
            dy,
            dvx,
            dvy,
            region,
        )
    )

    if region == "front":

        f_base = (
            closing_speed
            /
            max(
                1e-6,
                DV_MAX_FRONT,
            )
        )

        weight = W_FRONT

    elif region == "rear":

        f_base = (
            closing_speed
            /
            max(
                1e-6,
                DV_MAX_REAR,
            )
        )

        weight = W_REAR

    else:

        f_base = (
            closing_speed
            /
            max(
                1e-6,
                DV_MAX_LAT,
            )
        )

        weight = W_SIDE

    f_base = clamp(
        f_base,
        0.0,
        1.0,
    )

    return weight * f_base


# =============================================================================
# DIPF: EXACT DYNAMIC OBSTACLE POTENTIAL
# =============================================================================

def U_obs_exact(
    xh,
    yh,
    vxh,
    vyh,
    xo,
    yo,
    vxo,
    vyo,
    w_eff=2.0,
    l_eff=5.0,
):

    dx = xo - xh
    dy = yo - yh

    dvx = vxo - vxh
    dvy = vyo - vyh

    # Longitudinal influence window.
    if (
        dx > OBS_RADIUS_AHEAD
        or
        dx < -OBS_RADIUS_BEHIND
    ):
        return 0.0

    # Lateral influence window.
    if abs(dy) > LATERAL_BAND:
        return 0.0

    f_dv = (
        f_closing_region_aware(
            dx,
            dy,
            dvx,
            dvy,
        )
    )

    # Only closing obstacles contribute.
    if f_dv <= 0.0:
        return 0.0

    w_combined = (
        EGO_W + w_eff
    ) / 2.0

    l_combined = (
        EGO_L + l_eff
    ) / 2.0

    kp = k_prime_exact(
        dx,
        dy,
        dvx,
        dvy,
        w_combined,
        l_combined,
    )

    return (
        LAMBDA_OBS
        * f_dv
        / kp
    )


# =============================================================================
# TOTAL DIPF POTENTIAL
# =============================================================================

def calculate_total_potential(
    x,
    y,
    vx,
    vy,
    obstacles,
    goal_x,
    goal_y,
):

    _, _, y_left, y_right = (
        road_bounds_y_safe_for_ego()
    )

    total = (
        W_DEST
        *
        U_dest(
            x,
            y,
            goal_x,
            goal_y,
        )
    )

    total += (
        W_LANE
        *
        U_lane(
            y,
            y_left,
            y_right,
        )
    )

    for (
        xo,
        yo,
        vxo,
        vyo,
        w_eff,
        l_eff,
    ) in obstacles:

        total += (
            W_OBS
            *
            U_obs_exact(
                x,
                y,
                vx,
                vy,
                xo,
                yo,
                vxo,
                vyo,
                w_eff=w_eff,
                l_eff=l_eff,
            )
        )

    return total


# =============================================================================
# NUMERICAL DIPF GRADIENT
# =============================================================================

def numerical_grad_U_total(
    state,
    obstacles,
    xd,
    yd,
    y_left,
    y_right,
):

    x, y, vx, vy = state

    def U_total(
        xx,
        yy,
    ):

        U = (
            W_DEST
            *
            U_dest(
                xx,
                yy,
                xd,
                yd,
            )
        )

        U += (
            W_LANE
            *
            U_lane(
                yy,
                y_left,
                y_right,
            )
        )

        for (
            xo,
            yo,
            vxo,
            vyo,
            w_eff,
            l_eff,
        ) in obstacles:

            U += (
                W_OBS
                *
                U_obs_exact(
                    xx,
                    yy,
                    vx,
                    vy,
                    xo,
                    yo,
                    vxo,
                    vyo,
                    w_eff=w_eff,
                    l_eff=l_eff,
                )
            )

        return U

    eps = GRAD_EPS

    dUx = (
        U_total(
            x + eps,
            y,
        )
        -
        U_total(
            x - eps,
            y,
        )
    ) / (2.0 * eps)

    dUy = (
        U_total(
            x,
            y + eps,
        )
        -
        U_total(
            x,
            y - eps,
        )
    ) / (2.0 * eps)

    return dUx, dUy


# =============================================================================
# LOCAL MINIMUM DETECTION
# =============================================================================

def check_local_minima(
    x,
    y,
    vx,
    vy,
    obstacles,
    goal_x,
    goal_y,
    current_potential=None,
):

    y_min, y_max, _, _ = (
        road_bounds_y_safe_for_ego()
    )

    if current_potential is None:

        current_potential = (
            calculate_total_potential(
                x,
                y,
                vx,
                vy,
                obstacles,
                goal_x,
                goal_y,
            )
        )

    sample_radius = 2.0
    sample_points = 8

    nearby_potentials = []

    for i in range(sample_points):

        angle = (
            2.0
            * math.pi
            * i
            / sample_points
        )

        sx = (
            x
            + sample_radius
            * math.cos(angle)
        )

        sy = (
            y
            + sample_radius
            * math.sin(angle)
        )

        if (
            sy < y_min
            or
            sy > y_max
        ):
            continue

        p = (
            calculate_total_potential(
                sx,
                sy,
                vx,
                vy,
                obstacles,
                goal_x,
                goal_y,
            )
        )

        nearby_potentials.append(p)

    if not nearby_potentials:

        return (
            False,
            current_potential,
            [],
            0.0,
        )

    all_higher = all(
        p > current_potential - 0.1
        for p in nearby_potentials
    )

    dist_to_goal = math.hypot(
        x - goal_x,
        y - goal_y,
    )

    eps = 0.1

    ppx = calculate_total_potential(
        x + eps,
        y,
        vx,
        vy,
        obstacles,
        goal_x,
        goal_y,
    )

    pmx = calculate_total_potential(
        x - eps,
        y,
        vx,
        vy,
        obstacles,
        goal_x,
        goal_y,
    )

    ppy = calculate_total_potential(
        x,
        y + eps,
        vx,
        vy,
        obstacles,
        goal_x,
        goal_y,
    )

    pmy = calculate_total_potential(
        x,
        y - eps,
        vx,
        vy,
        obstacles,
        goal_x,
        goal_y,
    )

    grad_x = (
        ppx - pmx
    ) / (2.0 * eps)

    grad_y = (
        ppy - pmy
    ) / (2.0 * eps)

    gradient_mag = math.hypot(
        grad_x,
        grad_y,
    )

    is_local_min = (
        gradient_mag < 0.5
        and
        dist_to_goal > 5.0
        and
        all_higher
    )

    return (
        is_local_min,
        current_potential,
        nearby_potentials,
        gradient_mag,
    )


# =============================================================================
# OBSTACLE LIST
# =============================================================================

def build_obstacle_list(
    x,
    y,
):

    obstacles = []

    ids = (
        traci.vehicle.getIDList()
    )

    for vid in ids:

        if vid == EGO_ID:
            continue

        xo, yo = (
            traci.vehicle.getPosition(
                vid
            )
        )

        if (
            xo < x - OBS_RADIUS_BEHIND
            or
            xo > x + OBS_RADIUS_AHEAD
        ):
            continue

        if abs(yo - y) > LATERAL_BAND:
            continue

        vxo = (
            traci.vehicle.getSpeed(
                vid
            )
        )

        try:

            vyo = (
                traci.vehicle
                .getLateralSpeed(vid)
            )

        except Exception:

            vyo = 0.0

        try:

            typ = (
                traci.vehicle
                .getTypeID(vid)
            )

        except Exception:

            typ = "car"

        # Effective dimensions from uploaded DIPF code.
        if typ == "bike":

            w_eff = 0.8
            l_eff = 2.0

        elif typ == "auto":

            w_eff = 1.6
            l_eff = 3.0

        elif typ in (
            "bus",
            "truck",
        ):

            w_eff = 2.7
            l_eff = 12.0

        else:

            w_eff = 2.1
            l_eff = 5.0

        obstacles.append(
            (
                xo,
                yo,
                vxo,
                vyo,
                w_eff,
                l_eff,
            )
        )

    return obstacles


# =============================================================================
# DISTANCE / COLLISION
# =============================================================================

def get_closest_obstacle_distance(
    x,
    y,
    obstacles,
):

    if not obstacles:
        return np.nan

    min_dist = float("inf")

    for (
        xo,
        yo,
        *_rest,
    ) in obstacles:

        dist = math.hypot(
            xo - x,
            yo - y,
        )

        if dist < min_dist:
            min_dist = dist

    return min_dist


def ego_collision_sumo():

    try:

        cols = (
            traci.simulation
            .getCollisions()
        )

    except Exception:

        return False, ""

    for c in cols:

        collider = getattr(
            c,
            "collider",
            "",
        )

        victim = getattr(
            c,
            "victim",
            "",
        )

        if (
            collider == EGO_ID
            or
            victim == EGO_ID
        ):

            return (
                True,
                "SUMO collision: "
                f"collider={collider}, "
                f"victim={victim}",
            )

    return False, ""


def ego_collision_radius(
    ego_x,
    ego_y,
    obstacles,
):

    for (
        xo,
        yo,
        *_rest,
    ) in obstacles:

        if (
            math.hypot(
                ego_x - xo,
                ego_y - yo,
            )
            <= COLLISION_RADIUS
        ):

            return True

    return False


# =============================================================================
# EGO SETUP
# =============================================================================

def setup_ego_after_spawn():

    # DIPF owns longitudinal acceleration.
    traci.vehicle.setSpeedMode(
        EGO_ID,
        0,
    )

    # DIPF owns lateral motion.
    traci.vehicle.setLaneChangeMode(
        EGO_ID,
        0,
    )

    try:

        traci.vehicle.setColor(
            EGO_ID,
            (255, 0, 255, 255),
        )

    except Exception:
        pass


# =============================================================================
# FULL DIPF CONTROL STEP
# =============================================================================

def ego_control_step(
    dt,
    history,
):

    x, y = (
        traci.vehicle.getPosition(
            EGO_ID
        )
    )

    vx = (
        traci.vehicle.getSpeed(
            EGO_ID
        )
    )

    try:

        vy = (
            traci.vehicle
            .getLateralSpeed(
                EGO_ID
            )
        )

    except Exception:

        vy = 0.0

    (
        y_min,
        y_max,
        y_left,
        y_right,
    ) = road_bounds_y_safe_for_ego()

    goal_y = (
        lane_centers_y()
        [GOAL_LANE_INDEX]
    )

    # ---------------------------------------------------------
    # GOAL
    # ---------------------------------------------------------

    if success_goal_reached(
        x,
        y,
    ):

        x_stop = clamp_x_to_road(
            EGO_GOAL_X
        )

        y_stop = clamp(
            y,
            y_min,
            y_max,
        )

        lane_i_stop = (
            nearest_lane_index(
                y_stop
            )
        )

        moved = safe_move_ego(
            x_stop,
            y_stop,
            lane_i_hint=lane_i_stop,
        )

        if not moved:
            return "MAPFAIL"

        traci.vehicle.setSpeed(
            EGO_ID,
            0.0,
        )

        return "DONE"

    # ---------------------------------------------------------
    # OBSTACLES
    # ---------------------------------------------------------

    obstacles = (
        build_obstacle_list(
            x,
            y,
        )
    )

    if ego_collision_radius(
        x,
        y,
        obstacles,
    ):

        return "COLLISION"

    # ---------------------------------------------------------
    # TOTAL POTENTIAL GRADIENT
    # ---------------------------------------------------------

    dUx, dUy = (
        numerical_grad_U_total(
            (x, y, vx, vy),
            obstacles,
            EGO_GOAL_X,
            goal_y,
            y_left,
            y_right,
        )
    )

    # ---------------------------------------------------------
    # DIPF -> ACCELERATION
    # ---------------------------------------------------------

    ax = clamp(
        -dUx,
        -EGO_A_MAX,
        EGO_A_MAX,
    )

    ay = clamp(
        -dUy,
        -EGO_AY_MAX,
        EGO_AY_MAX,
    )

    # ---------------------------------------------------------
    # VELOCITY UPDATE
    # ---------------------------------------------------------

    vx_apply = clamp(
        vx + ax * dt,
        0.0,
        EGO_MAX_SPEED,
    )

    vy_apply = clamp(
        vy + ay * dt,
        -EGO_VY_MAX,
        EGO_VY_MAX,
    )

    # ---------------------------------------------------------
    # POSITION UPDATE
    # ---------------------------------------------------------

    x_new = clamp_x_to_road(
        x
        + vx * dt
        + 0.5 * ax * dt * dt
    )

    y_new = clamp(
        y
        + vy * dt
        + 0.5 * ay * dt * dt,
        y_min,
        y_max,
    )

    lane_i = (
        nearest_lane_index(
            y_new
        )
    )

    # ---------------------------------------------------------
    # APPLY DIPF MOTION
    # ---------------------------------------------------------

    moved = safe_move_ego(
        x_new,
        y_new,
        lane_i_hint=lane_i,
    )

    if not moved:
        return "MAPFAIL"

    traci.vehicle.setSpeed(
        EGO_ID,
        vx_apply,
    )

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    t = (
        traci.simulation
        .getTime()
    )

    grad_norm = math.hypot(
        dUx,
        dUy,
    )

    total_potential = (
        calculate_total_potential(
            x,
            y,
            vx,
            vy,
            obstacles,
            EGO_GOAL_X,
            goal_y,
        )
    )

    (
        is_min,
        _,
        _,
        grad_mag,
    ) = check_local_minima(
        x,
        y,
        vx,
        vy,
        obstacles,
        EGO_GOAL_X,
        goal_y,
        total_potential,
    )

    min_clear = (
        get_closest_obstacle_distance(
            x,
            y,
            obstacles,
        )
    )

    progress = (
        (x - EGO_START_X)
        /
        max(
            1e-6,
            EGO_GOAL_X - EGO_START_X,
        )
    )

    history.append(
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
            "is_local_min": int(
                is_min
            ),
            "progress": float(
                progress
            ),
        }
    )

    if (
        t % DEBUG_INTERVAL
        < STEP_LENGTH
    ):

        print(
            f"Time {t:.1f}s | "
            f"lane={lane_i + 1} | "
            f"v={vx_apply:.2f} | "
            f"vy={vy_apply:.2f} | "
            f"ax={ax:.2f} | "
            f"ay={ay:.2f} | "
            f"|∇U|={grad_norm:.3f}"
        )

    return "OK"


# =============================================================================
# STUCK DETECTION
# =============================================================================

def compute_stuck_flag(
    hist_df,
):

    if hist_df.empty:
        return False

    window_steps = max(
        1,
        int(
            STUCK_WINDOW_SEC
            /
            STEP_LENGTH
        ),
    )

    if (
        len(hist_df)
        < window_steps
    ):
        return False

    speeds = (
        hist_df["vx"]
        .values
    )

    xs = (
        hist_df["x"]
        .values
    )

    for i in range(
        window_steps,
        len(hist_df),
    ):

        s = speeds[
            i - window_steps:i
        ]

        xp = xs[
            i - window_steps:i
        ]

        if (
            np.mean(s)
            < STUCK_SPEED_THRESH
            and
            (
                xp[-1]
                -
                xp[0]
            )
            < STUCK_PROGRESS_THRESH
        ):

            return True

    return False


# =============================================================================
# RUN SIMULATION
# =============================================================================

def run_simulation():

    if not CFG_FILE.exists():

        raise FileNotFoundError(
            f"Missing SUMO config: {CFG_FILE}"
        )

    if not NET_FILE.exists():

        raise FileNotFoundError(
            f"Missing SUMO network: {NET_FILE}\n"
            "Build it separately using netconvert."
        )

    history = []

    sumo_bin = (
        sumolib.checkBinary(
            "sumo-gui"
        )
    )

    cmd = [
        sumo_bin,
        "-c",
        str(CFG_FILE),
        "--seed",
        "42",
    ]

    print("=" * 90)
    print("FULL PURE-DIPF CONTROLLER")
    print("=" * 90)
    print(
        f"Config : {CFG_FILE}"
    )
    print(
        f"Network: {NET_FILE}"
    )
    print(
        f"Goal   : x={EGO_GOAL_X:.1f} m, "
        f"lane={GOAL_LANE_INDEX + 1}"
    )
    print("=" * 90)

    traci.start(cmd)

    ego_initialized = False
    done_reason = "timeout"

    try:

        max_steps = int(
            (
                SIM_END
                -
                SIM_BEGIN
            )
            /
            STEP_LENGTH
        )

        for _ in range(
            max_steps
        ):

            traci.simulationStep()

            # Check actual SUMO collisions.
            hit, msg = (
                ego_collision_sumo()
            )

            if hit:

                print(msg)
                done_reason = "collision"
                break

            # Detect ego after its departure time.
            if (
                not ego_initialized
                and
                EGO_ID
                in traci.vehicle.getIDList()
            ):

                setup_ego_after_spawn()

                ego_initialized = True

                print(
                    f"Ego spawned at "
                    f"{traci.simulation.getTime():.1f}s"
                )

            # Run DIPF controller.
            if (
                ego_initialized
                and
                EGO_ID
                in traci.vehicle.getIDList()
            ):

                status = (
                    ego_control_step(
                        STEP_LENGTH,
                        history,
                    )
                )

                if status == "DONE":

                    done_reason = "success"

                    print(
                        "DIPF goal reached."
                    )

                    break

                if status == "COLLISION":

                    done_reason = "collision"
                    break

                if status == "MAPFAIL":

                    done_reason = "mapfail"

                    print(
                        "moveToXY mapping failed."
                    )

                    break

        if (
            done_reason == "timeout"
            and
            len(history) > 0
        ):

            if compute_stuck_flag(
                pd.DataFrame(
                    history
                )
            ):

                done_reason = "stuck"

    finally:

        try:
            traci.close()

        except Exception:
            pass

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    results_dir = (
        BASE_DIR / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hist_df = pd.DataFrame(
        history
    )

    trajectory_file = (
        results_dir
        /
        "trajectory.csv"
    )

    hist_df.to_csv(
        trajectory_file,
        index=False,
    )

    metrics = {
        "status": done_reason,
        "success": int(
            done_reason == "success"
        ),
        "collision": int(
            done_reason == "collision"
        ),
        "timeout": int(
            done_reason == "timeout"
        ),
        "stuck": int(
            done_reason == "stuck"
        ),
        "mapfail": int(
            done_reason == "mapfail"
        ),
    }

    if not hist_df.empty:

        metrics.update(
            {
                "final_x": float(
                    hist_df["x"].iloc[-1]
                ),
                "final_y": float(
                    hist_df["y"].iloc[-1]
                ),
                "final_lane": (
                    final_lane_name(
                        float(
                            hist_df[
                                "y"
                            ].iloc[-1]
                        )
                    )
                ),
                "final_progress": float(
                    hist_df[
                        "progress"
                    ].iloc[-1]
                ),
                "avg_speed": float(
                    hist_df[
                        "vx"
                    ].mean()
                ),
                "avg_abs_ay": float(
                    hist_df[
                        "ay"
                    ].abs().mean()
                ),
                "min_clearance": float(
                    hist_df[
                        "min_clearance"
                    ].min()
                ),
                "local_minima_count": int(
                    hist_df[
                        "is_local_min"
                    ].sum()
                ),
            }
        )

    with open(
        results_dir
        /
        "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("SIMULATION COMPLETE")
    print("=" * 90)
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )
    print(
        f"\nTrajectory saved to: "
        f"{trajectory_file}"
    )

    return metrics, hist_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    run_simulation()
