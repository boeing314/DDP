"""
Plot |F| = |-grad(U)| at a fixed (x, y) position as the relative velocity
magnitude v sweeps from 0 to 10, reusing the potential/force model in k9.py.

Usage:
    python force_vs_velocity_k9.py X Y
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k9 import force_at_point  # noqa: E402

# ============================================================
# Directions of the relative-velocity vector. v (swept 0..vmax) scales
# along each (dvx, dvy) direction, e.g. (1, 0) sweeps pure dvx, (0, 1)
# sweeps pure dvy, (1, 1) sweeps both equally. Edit this list directly --
# each entry is plotted as its own curve.
# ============================================================
DIRECTIONS = [(0.5,0.7), ( 1.5,0.7), ( 3.0,0.7), ( 10.0,0.7),(100,0.7)]

# Velocity range to sweep over. Edit these directly.
VMIN, VMAX = -10.0, 10.0


def force_vs_velocity(x, y, dvx, dvy, vmin=VMIN, vmax=VMAX, n=200):
    """Return (v, |F|) arrays at (x, y) as v sweeps from vmin to vmax along
    the (dvx, dvy) direction."""
    v = np.linspace(vmin, vmax, n)

    norm = np.hypot(dvx, dvy)
    ux, uy = (dvx / norm, dvy / norm) if norm > 0 else (0.0, 0.0)

    mags = np.empty_like(v)
    for i, vi in enumerate(v):
        _, _, mag = force_at_point(x, y, vi * ux, vi * uy)
        mags[i] = mag

    return v, mags


def plot_force_vs_velocity(x, y, vmin=VMIN, vmax=VMAX, n=200, directions=DIRECTIONS):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for dvx, dvy in directions:
        v, mags = force_vs_velocity(x, y, dvx, dvy, vmin=vmin, vmax=vmax, n=n)
        ax.plot(v, mags, linewidth=2, label=f"(dvx, dvy)=({dvx:g}, {dvy:g})")

    ax.set_xlabel("Velocity $v$ (kmph)")
    ax.set_ylabel("$|F|$")
    ax.set_title(f"Force magnitude vs. $v$ at (x={x:g}, y={y:g})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, nargs="?", default=3.0, help="X position of the point")
    parser.add_argument("y", type=float, nargs="?", default=30.0, help="Y position of the point")
    parser.add_argument("--vmin", type=float, default=VMIN,
                         help=f"min velocity (default {VMIN})")
    parser.add_argument("--vmax", type=float, default=VMAX,
                         help=f"max velocity (default {VMAX})")
    parser.add_argument("--n", type=int, default=200,
                         help="number of samples between vmin and vmax")
    args = parser.parse_args()

    plot_force_vs_velocity(args.x, args.y, vmin=args.vmin, vmax=args.vmax, n=args.n)
