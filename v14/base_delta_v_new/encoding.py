"""
Laneless, vehicle-type-aware encoding of the traffic around the ego.

No lanes are used anywhere: traffic here is sub-lane (SL2015, 0.25 m lateral
resolution), bikes filter between lanes, and vehicles range from 1.8 m bikes to
12 m buses/trucks. Every vehicle is therefore described relative to the EGO'S
OWN BODY and PATH, using body-to-body (bounding-box) gaps, so a bike half a
metre to the side and a truck in the next "lane" are treated by geometry, not
by lane membership.

Geometry (ego frame: x along the road, y lateral, body centres, actual sizes):
    gx = |dx| - (L_ego + L_veh)/2     longitudinal box gap (< 0: bodies overlap in x)
    gy = |dy| - (W_ego + W_veh)/2     lateral box gap      (< 0: bodies overlap in y)

Sectors (8), from the longitudinal and lateral relation:
    longitudinal: ahead (gx >= 0, dx > 0) | alongside (gx < 0) | behind (gx >= 0, dx < 0)
    lateral:      path (gy < path_margin -- would touch if the x gap closed) | left (dy > 0) | right
    -> front, front_left, front_right, left, right, rear, rear_left, rear_right
       (alongside + path = bodies overlapping = contact; counted as "front")

Per sector, the MOST CRITICAL vehicle (highest inverse time-to-contact, then
smallest box distance) is encoded with 9 values:
    present        1 / 0
    gap_long       max(gx, 0), capped at long_range (free sector: long_range)
    gap_lat        max(gy, 0), capped at lat_range  (free sector: lat_range)
    dvx, dvy       vehicle velocity - ego velocity
    inv_ttc        1 / time until the two boxes would first touch at constant
                   velocities (0 if they never will); continuous, no cap needed
                   except a numerical one (inv_ttc_cap)
    drac           braking needed to avoid contact along x: closing^2 / (2 * gap_long)
                   (front: what the EGO must brake; rear: what the FOLLOWER must brake)
    length, width  vehicle size -> the vehicle type (bike / auto / car / bus / truck)

Global context (8):
    vx, vy, heading     ego velocity and its angle to the road (rad)
    y                   ego lateral position (road centre = 0)
    clear_left/right    body clearance to the left / right road edge
    density             vehicles within +-density_range (any lateral position)
    mean_dvx_ahead      mean relative speed of vehicles within density_range ahead
                        (negative = the traffic ahead is slower: congestion)

Total: 8 x 9 + 8 = 80 features (FEATURES gives the names, in order).

Vehicle tuples: (x_centre, y_centre, vx, vy, length, width).
"""

import math

SECTORS = ["front", "front_left", "front_right", "left", "right", "rear", "rear_left", "rear_right"]
PER_SECTOR = ["present", "gap_long", "gap_lat", "dvx", "dvy", "inv_ttc", "drac", "length", "width"]
GLOBAL = ["vx", "vy", "heading", "y", "clear_left", "clear_right", "density", "mean_dvx_ahead"]
FEATURES = [f"{s}_{f}" for s in SECTORS for f in PER_SECTOR] + GLOBAL

DEFAULTS = {"path_margin": 0.3, "long_range": 100.0, "lat_range": 10.0, "inv_ttc_cap": 10.0,
            "density_range": 50.0}


def box_ttc(dx, dy, dvx, dvy, sum_l, sum_w):
    """Time until two axis-aligned boxes (centre offset dx, dy; relative velocity
    dvx, dvy = vehicle - ego; summed sizes) first overlap. 0 if already
    overlapping, inf if never."""
    def interval(d, dv, half):
        if abs(dv) < 1e-9:
            return (-math.inf, math.inf) if abs(d) < half else None
        t1, t2 = (-half - d) / dv, (half - d) / dv
        return (min(t1, t2), max(t1, t2))

    ix, iy = interval(dx, dvx, sum_l / 2.0), interval(dy, dvy, sum_w / 2.0)
    if ix is None or iy is None:
        return math.inf
    t_in, t_out = max(ix[0], iy[0]), min(ix[1], iy[1])
    if t_in > t_out or t_out < 0.0:
        return math.inf
    return max(0.0, t_in)


def classify(dx, dy, gx, gy, path_margin):
    if gx >= 0.0:
        lon = "front" if dx > 0 else "rear"
    else:
        lon = "side"
    if gy < path_margin:
        lat = "path"
    else:
        lat = "left" if dy > 0 else "right"
    if lon == "side":
        return "front" if lat == "path" else lat
    return lon if lat == "path" else f"{lon}_{lat}"


def encode(ego, vehicles, road_edges, ego_length, ego_width, **kw):
    """
    ego        (x, y, vx, vy) ego body centre and velocity
    vehicles   iterable of (x, y, vx, vy, length, width) for every other vehicle
    road_edges (y_low, y_high)
    Returns a list of floats in FEATURES order.
    """
    p = {**DEFAULTS, **kw}
    xe, ye, vxe, vye = ego
    best = {}                                   # sector -> (key, values)
    density, dv_ahead = 0, []

    for (xo, yo, vxo, vyo, lo, wo) in vehicles:
        dx, dy = xo - xe, yo - ye
        dvx, dvy = vxo - vxe, vyo - vye
        if abs(dx) <= p["density_range"]:
            density += 1
            if dx > 0:
                dv_ahead.append(dvx)
        gx = abs(dx) - 0.5 * (ego_length + lo)
        gy = abs(dy) - 0.5 * (ego_width + wo)
        if gx > p["long_range"] or gy > p["lat_range"]:
            continue
        sector = classify(dx, dy, gx, gy, p["path_margin"])
        ttc = box_ttc(dx, dy, dvx, dvy, ego_length + lo, ego_width + wo)
        inv_ttc = min(p["inv_ttc_cap"], 1.0 / ttc) if ttc > 0 else p["inv_ttc_cap"]
        if ttc == math.inf:
            inv_ttc = 0.0
        gap_long = max(gx, 0.0)
        closing = -dvx if dx > 0 else dvx       # >0: the x gap is shrinking
        drac = (closing * closing) / (2.0 * max(gap_long, 0.1)) if closing > 0 else 0.0
        dist = math.hypot(gap_long, max(gy, 0.0))
        key = (-inv_ttc, dist)
        values = [1.0, min(gap_long, p["long_range"]), min(max(gy, 0.0), p["lat_range"]),
                  dvx, dvy, inv_ttc, drac, lo, wo]
        if sector not in best or key < best[sector][0]:
            best[sector] = (key, values)

    feats = []
    for s in SECTORS:
        if s in best:
            feats += best[s][1]
        else:
            feats += [0.0, p["long_range"], p["lat_range"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    y_low, y_high = road_edges
    half_w = 0.5 * ego_width
    heading = math.atan2(vye, vxe) if math.hypot(vxe, vye) > 1e-6 else 0.0
    feats += [vxe, vye, heading, ye, y_high - (ye + half_w), (ye - half_w) - y_low,
              float(density), (sum(dv_ahead) / len(dv_ahead)) if dv_ahead else 0.0]
    return [float(v) for v in feats]
