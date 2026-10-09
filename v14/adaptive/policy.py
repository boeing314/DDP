"""
v13 adaptive-parameter policy (gain scheduling).

Every control step:
    f_t  = features(encoding)                       8 dynamics features, each in [0, 1]
    z*_t = clip(W f_t + b, -c, c)                   one exponent per adapted parameter
    z_t  = z_{t-1} + min(1, dt/T) (z*_t - z_{t-1})  first-order smoothing (time constant T)
    p_t  = clamp(p0 * exp(z_t), limits)             p0 = base value of the parameter

W = 0, b = 0 gives exactly the base parameters (the untrained policy = the fixed controller).

Features (all mirror-invariant: swapping left and right does not change them, like the controller):
    ttc_front       1 - exp(-1/TTC) to the most critical vehicle in the ego's path ahead
    ttc_side_front  same, worst of front-left / front-right
    ttc_rear_side   same, worst of rear / rear-left / rear-right / alongside (closing from behind or the side)
    gap_front       exp(-gap/20 m) to the vehicle ahead (0 = nothing ahead)
    gap_side_front  same, closest of front-left / front-right
    tight_lateral   exp(-room/1 m), room = smallest lateral space (road edge or vehicle alongside)
    speed           ego speed / 34 m/s
    density         1 - exp(-n/10), n = vehicles within +-50 m
"""

import math

import numpy as np

from encoding import FEATURES as ENC

FEATURES = ["ttc_front", "ttc_side_front", "ttc_rear_side", "gap_front", "gap_side_front",
            "tight_lateral", "speed", "density"]
_I = {name: i for i, name in enumerate(ENC)}
V_MAX = 34.0


def features(enc):
    """80-value encoding (encoding.FEATURES order) -> the 8 policy features."""
    e = lambda k: enc[_I[k]]                                    # noqa: E731

    def closeness(s):
        return math.exp(-e(f"{s}_gap_long") / 20.0) * e(f"{s}_present")

    room = min(e("clear_left"), e("clear_right"))
    for s in ("left", "right"):
        if e(f"{s}_present"):
            room = min(room, e(f"{s}_gap_lat"))
    return [
        1.0 - math.exp(-e("front_inv_ttc")),
        1.0 - math.exp(-max(e("front_left_inv_ttc"), e("front_right_inv_ttc"))),
        1.0 - math.exp(-max(e(f"{s}_inv_ttc") for s in ("rear", "rear_left", "rear_right", "left", "right"))),
        closeness("front"),
        max(closeness("front_left"), closeness("front_right")),
        math.exp(-max(room, 0.0) / 1.0),
        min(e("vx") / V_MAX, 1.0),
        1.0 - math.exp(-e("density") / 10.0),
    ]


def target_exponents(policy, f):
    """z* = clip(W f + b, -c, c) for one feature vector."""
    z = np.asarray(policy["W"]) @ np.asarray(f) + np.asarray(policy["b"])
    return np.clip(z, -policy["clip"], policy["clip"])


def apply(policy, base, z):
    """Parameter values for exponents z: base * exp(z), clamped to the policy's absolute limits."""
    out = {}
    for k, name in enumerate(policy["params"]):
        lo, hi = policy["limits"][name]
        out[name] = min(max(base[name] * math.exp(float(z[k])), lo), hi)
    return out


# ---- packing for the optimiser ------------------------------------------------

def n_dims(n_params, bias_only=False):
    return n_params if bias_only else n_params * (len(FEATURES) + 1)


def unpack(x, params, signs=None, bias_only=False):
    """Optimiser vector -> (W, b). signs[p][f] = +1 / -1 forces w >= 0 / w <= 0 (by reflection)."""
    P, F = len(params), len(FEATURES)
    x = np.asarray(x, float)
    if bias_only:
        return np.zeros((P, F)), x.copy()
    W, b = x[:P * F].reshape(P, F).copy(), x[P * F:].copy()
    for (p, f), s in sign_mask(params, signs).items():
        W[p, f] = s * abs(W[p, f])
    return W, b


def sign_mask(params, signs):
    out = {}
    for name, per in (signs or {}).items():
        if name in params:
            for feat, s in per.items():
                out[(params.index(name), FEATURES.index(feat))] = 1.0 if s > 0 else -1.0
    return out


def make_policy(W, b, params, limits, clip, smooth_s):
    return {"params": list(params), "features": FEATURES, "W": np.asarray(W).tolist(), "b": np.asarray(b).tolist(),
            "clip": float(clip), "smooth_s": float(smooth_s), "limits": {k: list(limits[k]) for k in params}}
