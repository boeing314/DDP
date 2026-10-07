"""
Slot-based traffic-situation features (shared by make_situations.py and
cluster_controller.py, so offline data and live driving use the same code).

Slots: the nearest vehicle AHEAD and BEHIND the ego in its own lane, the lane to
its left and the lane to its right (6 slots). A vehicle belongs to the lane whose
centre is nearest to its body-centre y; "ahead" = its centre x is greater than
the ego's.

Per slot:
    gap  bumper-to-bumper longitudinal gap (m), capped at slot_range_m;
         a free slot (no vehicle within range) has gap = slot_range_m;
         a slot in a lane that does not exist (ego in an outer lane) has gap = 0
         -- a wall is as blocking as a vehicle touching the ego
    dv   vehicle speed - ego vx (m/s); 0 for free / non-existent slots

Ego / context:
    vx, vy               controller-held ego velocity
    lat_offset           ego y - own lane centre (m)
    clear_left/right     clearance from the ego's body to the left/right road edge (m)
    left/right_exists    1 if that lane exists, else 0
    ttc_ahead            time to contact with the own-lane leader (s), capped at ttc_cap_s
                         (= cap when not closing or no leader)
    density              number of vehicles within +-density_range_m (any lane)

Vehicle tuples: (x_centre, y_centre, vx, vy, length).
"""

import math

SLOTS = [f"{lane}_{side}" for lane in ("own", "left", "right") for side in ("ahead", "behind")]
FEATURES = ([f"gap_{s}" for s in SLOTS] + [f"dv_{s}" for s in SLOTS]
            + ["vx", "vy", "lat_offset", "clear_left", "clear_right",
               "left_exists", "right_exists", "ttc_ahead", "density"])


def nearest_lane(y, lane_centres):
    return min(range(len(lane_centres)), key=lambda i: abs(y - lane_centres[i]))


def situation_features(ego, vehicles, lane_centres, road_edges, ego_length, ego_width,
                       slot_range=100.0, ttc_cap=20.0, density_range=50.0):
    """
    ego          (x, y, vx, vy)  ego body centre + velocity
    vehicles     iterable of (x, y, vx, vy, length) for every other vehicle
    lane_centres y of each lane centre, index 0 = lowest y (SUMO lane 0)
    road_edges   (y_low_edge, y_high_edge)
    Returns a list of floats in FEATURES order.

    "left" = towards higher y (SUMO lane index + 1), matching SUMO's lane numbering
    on this road (lane 0 is the rightmost lane in the driving direction +x).
    """
    xe, ye, vxe, vye = ego
    own = nearest_lane(ye, lane_centres)
    lanes = {"own": own, "left": own + 1, "right": own - 1}
    exists = {k: 0 <= v < len(lane_centres) for k, v in lanes.items()}

    best = {s: None for s in SLOTS}                       # slot -> (gap, dv)
    density = 0
    for (xo, yo, vxo, vyo, lo) in vehicles:
        if abs(xo - xe) <= density_range:
            density += 1
        lane = nearest_lane(yo, lane_centres)
        for name, idx in lanes.items():
            if lane != idx or not exists[name]:
                continue
            if xo > xe:
                side, gap = "ahead", (xo - 0.5 * lo) - (xe + 0.5 * ego_length)
            else:
                side, gap = "behind", (xe - 0.5 * ego_length) - (xo + 0.5 * lo)
            if gap > slot_range:
                continue
            key = f"{name}_{side}"
            if best[key] is None or gap < best[key][0]:
                best[key] = (gap, vxo - vxe)

    gaps, dvs = [], []
    for s in SLOTS:
        lane_name = s.split("_")[0]
        if not exists[lane_name]:
            gaps.append(0.0); dvs.append(0.0)              # wall
        elif best[s] is None:
            gaps.append(float(slot_range)); dvs.append(0.0)  # free
        else:
            gaps.append(float(best[s][0])); dvs.append(float(best[s][1]))

    lead = best["own_ahead"]
    if lead is not None and lead[1] < -1e-6:              # leader slower -> closing
        ttc = min(ttc_cap, max(0.0, lead[0]) / -lead[1])
    else:
        ttc = float(ttc_cap)

    y_low, y_high = road_edges
    half_w = 0.5 * ego_width
    return (gaps + dvs
            + [float(vxe), float(vye), float(ye - lane_centres[own]),
               float(y_high - (ye + half_w)), float((ye - half_w) - y_low),
               float(exists["left"]), float(exists["right"]), float(ttc), float(density)])
