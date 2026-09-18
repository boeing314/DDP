"""
Plot |F| = |-grad(U)| at a fixed (x, y) position as dvx sweeps from vmin to
vmax, with dvy held fixed, reusing the potential/force model in k8.py.

Usage:
    python force_vs_velocity.py X Y
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k8 import force_at_point  # noqa: E402

# Fixed dvy, and the dvx range to sweep over. Edit these directly.
DVY = 0.7
VMIN, VMAX = -10.0, 10.0


def force_vs_velocity(x, y, dvy=DVY, vmin=VMIN, vmax=VMAX, n=200):
    """Return (dvx, |F|) arrays at (x, y) as dvx sweeps from vmin to vmax
    with dvy held fixed."""
    dvx = np.linspace(vmin, vmax, n)

    mags = np.empty_like(dvx)
    for i, vi in enumerate(dvx):
        _, _, mag = force_at_point(x, y, vi, dvy)
        mags[i] = mag

    return dvx, mags


def plot_force_vs_velocity(x, y, dvy=DVY, vmin=VMIN, vmax=VMAX, n=200):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    dvx, mags = force_vs_velocity(x, y, dvy=dvy, vmin=vmin, vmax=vmax, n=n)
    ax.plot(dvx, mags, linewidth=2)

    ax.set_xlabel("$\\Delta v_x$ (kmph)")
    ax.set_ylabel("$|F|$")
    ax.set_title(f"Force magnitude vs. $\\Delta v_x$ at (x={x:g}, y={y:g}), $\\Delta v_y$={dvy:g}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, nargs="?", default=3.0, help="X position of the point")
    parser.add_argument("y", type=float, nargs="?", default=30.0, help="Y position of the point")
    parser.add_argument("--dvy", type=float, default=DVY,
                         help=f"fixed relative velocity dvy (default {DVY})")
    parser.add_argument("--vmin", type=float, default=VMIN,
                         help=f"min dvx (default {VMIN})")
    parser.add_argument("--vmax", type=float, default=VMAX,
                         help=f"max dvx (default {VMAX})")
    parser.add_argument("--n", type=int, default=200,
                         help="number of samples between vmin and vmax")
    args = parser.parse_args()

    plot_force_vs_velocity(args.x, args.y, dvy=args.dvy, vmin=args.vmin, vmax=args.vmax, n=args.n)
