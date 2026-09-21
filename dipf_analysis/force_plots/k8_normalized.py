r"""
k8 variant: force normalised by |grad rho|  ("delta_p").

Background
----------
For a potential that depends on position only through the elliptical
pseudo-distance rho, the chain rule splits the force magnitude in two:

    |grad U| = |U'(rho)| * |grad rho|
               \_________/   \________/
               function of    varies over [1/max(tx,ty), 1/min(tx,ty)]
               rho alone      around a single ellipse

The first factor is monotone in rho and can never create an off-axis ridge.
The second factor swings by the elongation ratio q = ty/tx as you go around
an ellipse (it equals 1/ty on the major axis and 1/tx on the minor axis), and
it is solely responsible for the U-shaped dip in |F| ahead of the obstacle
once q > sqrt(3).

This module divides that factor out:

    F_norm = -grad U / |grad rho|

which leaves |F| a monotone function of rho -- maximal on the axis, decaying
outward, exactly as an isotropic (circular) field behaves -- while leaving
the force DIRECTION completely unchanged, since |grad rho| > 0 is a scalar
and rescaling does not rotate a vector.

Set NORMALIZE_BY_GRAD_RHO = False to recover the plain k8 behaviour.

Caveat: F_norm is not conservative for q > 1, so U is no longer a Lyapunov
function for it. See dip_remedy.tex, section "The price of Remedy D".
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from queue import Empty, Queue
from threading import Thread

# ============================================================
# PARAMETERS
# ============================================================
w, l = 1.75, 4.5         # obstacle vehicle width, length (m)
host_w, host_l = 1.75, 4.5   # assumed host vehicle size, for the safety box
xo, yo = 0.0, 30.0       # obstacle vehicle centre position
PLOT_DVX, PLOT_DVY = 0, 3    # relative velocity for the plot (kmph)

lam = 10                 # scaling factor (lambda), Eq. 1
dv_max = 10.0            # normalising max relative velocity, Eq. 2
tau_x, tau_y = 1, 5
alpha = 1                # velocity-skew sensitivity, Eq. 6

# --- the two switches that distinguish this file from k8 -----------------
NORMALIZE_BY_GRAD_RHO = True   # Remedy D. False -> plain k8 force.
VELOCITY_INFLATION = False     # False reproduces k8's current
                               # `return tau_*0+tau0`, i.e. tau is left at
                               # tau0 and the field is Delta-v independent.
# -------------------------------------------------------------------------

U_CAP = 400000.0
N_LEVELS = 100
LEVEL_POWER = 1.0

QUIVER_STRIDE = 16
QUIVER_STEP = 1e-5

POINT_ARROW_LEN = 1.0
POINT_ARROW_COLOR = "cyan"

FORCE_CLIP_MIN = 1.0
FORCE_CLIP_MAX = 100.0

EPS = 1e-6


# ============================================================
# Eq. 2 -- f(Delta v)
# ============================================================
def f_dv(dx, dy, dvx, dvy):
    return dvx * 0 + 1


# ============================================================
# tau and the elliptical pseudo-distance rho, factored out of kprime so
# that |grad rho| can be formed independently of k'.
# ============================================================
def taus(dx, dy, dvx, dvy):
    """Return (tx, ty), the per-axis time constants of Eq. 6."""
    def tau(delta, dvel, tau0):
        if not VELOCITY_INFLATION:
            return np.full_like(np.asarray(delta, dtype=float), float(tau0))
        return np.where(-delta * dvel > 0, tau0 * (1 + alpha * abs(dvel)), tau0)

    tx = tau(dx, dvx, tau_x)
    ty = tau(dy, dvy, tau_y)
    tx = np.where(np.abs(tx) < EPS, EPS, tx)
    ty = np.where(np.abs(ty) < EPS, EPS, ty)
    return tx, ty


def rho(dx, dy, dvx, dvy):
    """Elliptical pseudo-distance (the `dist` term of Eq. 6)."""
    tx, ty = taus(dx, dy, dvx, dvy)
    return np.sqrt((dx / tx) ** 2 + (dy / ty) ** 2)


def grad_rho_mag(X, Y, dvx, dvy):
    """
    |grad rho| with respect to the HOST position (X, Y), evaluated
    analytically.

    With dx = xo - X and dy = yo - Y, and tx, ty piecewise constant in each
    quadrant,

        d(rho)/dX = -dx / (tx^2 * rho),   d(rho)/dY = -dy / (ty^2 * rho)

    so |grad rho| = sqrt( (dx/tx^2)^2 + (dy/ty^2)^2 ) / rho.

    The sign of dx/dX cancels in the magnitude. On the major axis this tends
    to 1/ty, on the minor axis to 1/tx -- the [1/ty, 1/tx] swing that causes
    the dip.
    """
    dx = xo - X
    dy = yo - Y
    tx, ty = taus(dx, dy, dvx, dvy)
    r = rho(dx, dy, dvx, dvy)
    r = np.where(r < EPS, EPS, r)
    return np.sqrt((dx / tx ** 2) ** 2 + (dy / ty ** 2) ** 2) / r


def grad_rho_mag_numeric(X, Y, dvx, dvy, step=1e-6):
    """Central-difference |grad rho|, used only to check grad_rho_mag."""
    d = lambda a, b: rho(xo - a, yo - b, dvx, dvy)
    drdX = (d(X + step, Y) - d(X - step, Y)) / (2 * step)
    drdY = (d(X, Y + step) - d(X, Y - step)) / (2 * step)
    return np.hypot(drdX, drdY)


# ============================================================
# Eq. 6 -- asymmetric, dimension-aware pseudo-distance k'
# ============================================================
def kprime(dx, dy, dvx, dvy):
    dist = rho(dx, dy, dvx, dvy)

    region13 = np.abs(dy) <= (l * np.abs(dx)) / w   # front/behind regions

    abs_dx = np.where(np.abs(dx) < EPS, EPS, np.abs(dx))
    abs_dy = np.where(np.abs(dy) < EPS, EPS, np.abs(dy))

    k_13 = (1 - w / abs_dx) * dist
    k_24 = (1 - l / abs_dy) * dist

    return np.where(region13, k_13, k_24)


# ============================================================
# Eq. 1 -- U = lambda * f(Delta v) / k'
# ============================================================
def potential(X, Y, dvx, dvy):
    dx = xo - X
    dy = yo - Y
    k = kprime(dx, dy, dvx, dvy)
    k = np.where(k <= 0, 1e-4, k)
    U = lam * f_dv(dx, dy, dvx, dvy) / np.abs(k)
    return np.clip(U, -10, U_CAP)


def gradient_field(X, Y, dvx, dvy, step=QUIVER_STEP):
    """Raw central-difference grad U (NOT normalised)."""
    dU_dX = (potential(X + step, Y, dvx, dvy) -
             potential(X - step, Y, dvx, dvy)) / (2 * step)
    dU_dY = (potential(X, Y + step, dvx, dvy) -
             potential(X, Y - step, dvx, dvy)) / (2 * step)
    return dU_dX, dU_dY


def force_field(X, Y, dvx, dvy, step=QUIVER_STEP, normalize=None):
    """
    F = -grad U, optionally divided by |grad rho|.

    Returns (Fx, Fy). This is the single place the normalisation is applied;
    everything downstream (quiver, background, point probes) goes through it.
    """
    if normalize is None:
        normalize = NORMALIZE_BY_GRAD_RHO

    dU_dX, dU_dY = gradient_field(X, Y, dvx, dvy, step)
    Fx, Fy = -dU_dX, -dU_dY

    if normalize:
        gn = grad_rho_mag(X, Y, dvx, dvy)
        gn = np.where(gn < EPS, EPS, gn)   # guard the origin only
        Fx, Fy = Fx / gn, Fy / gn

    return Fx, Fy


def force_at_point(X, Y, dvx, dvy, step=QUIVER_STEP, normalize=None):
    """Evaluate F at a single (X, Y). Returns (Fx, Fy, |F|)."""
    Fx, Fy = force_field(np.array(X, dtype=float), np.array(Y, dtype=float),
                         dvx, dvy, step, normalize)
    return float(Fx), float(Fy), float(np.hypot(Fx, Fy))


def gradient_magnitude_at(X, Y, dvx, dvy, step=1e-3):
    return force_at_point(X, Y, dvx, dvy, step)[2]


def plot_force_at_point(ax, X, Y, dvx, dvy, color=POINT_ARROW_COLOR,
                        arrow_len=POINT_ARROW_LEN, label_prefix=""):
    """Draw F at one point; direction exact, on-screen length fixed."""
    Fx, Fy, mag = force_at_point(X, Y, dvx, dvy)

    if mag > 1e-9:
        ux, uy = Fx / mag, Fy / mag
    else:
        ux, uy = 0.0, 0.0

    ax.plot(X, Y, marker="o", color=color, markersize=5, zorder=5)
    ax.annotate("", xy=(X + ux * arrow_len, Y + uy * arrow_len), xytext=(X, Y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=5)
    ax.annotate(f"{label_prefix}|F|={mag:.3g}", xy=(X, Y), xytext=(6, 6),
                textcoords="offset points", color=color, fontsize=8, zorder=5)
    return Fx, Fy, mag


def prompt_force_at_points(scenarios, fig, ax=None):
    """Terminal prompt for X,Y while keeping the figure responsive."""
    entries = Queue()

    def read_entries():
        print("\nEnter X,Y to evaluate F (enter 0,0 to exit).")
        while True:
            try:
                entries.put(input("X, Y: "))
            except (EOFError, KeyboardInterrupt):
                entries.put(None)
                return

    Thread(target=read_entries, daemon=True).start()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(scenarios), 1)))

    while plt.fignum_exists(fig.number):
        plt.pause(0.05)
        try:
            entry = entries.get_nowait()
        except Empty:
            continue

        if entry is None:
            print()
            return

        values = entry.replace(",", " ").split()
        if len(values) != 2:
            print("Please enter exactly two numbers, for example: 1.5, 28")
            continue
        try:
            X, Y = map(float, values)
        except ValueError:
            print("Please enter numeric X and Y values.")
            continue
        if X == 0 and Y == 0:
            return

        tag = "normalised" if NORMALIZE_BY_GRAD_RHO else "plain"
        print(f"At X={X:g}, Y={Y:g}  ({tag}):")
        for (dvx, dvy), color in zip(scenarios, colors):
            gn = float(grad_rho_mag(np.array(X), np.array(Y), dvx, dvy))
            if ax is not None:
                label = f"dv=({dvx:g},{dvy:g}) " if len(scenarios) > 1 else ""
                Fx, Fy, mag = plot_force_at_point(ax, X, Y, dvx, dvy,
                                                  color=color, label_prefix=label)
            else:
                Fx, Fy, mag = force_at_point(X, Y, dvx, dvy)
            print(f"  dvx={dvx:g}, dvy={dvy:g}: |grad rho|={gn:.4f}"
                  f"  F=({Fx:.4f}, {Fy:.4f})  |F|={mag:.4f}")

        if ax is not None:
            fig.canvas.draw_idle()


def draw_scene(ax, xo=xo, yo=yo, w=w, l=l, host_w=host_w, host_l=host_l):
    """Obstacle box, combined-footprint safety box, lane lines."""
    ax.add_patch(Rectangle((xo - w / 2, yo - l / 2), w, l,
                           fill=False, edgecolor="red", linewidth=1.5))
    safe_w, safe_l = w + host_w, l + host_l
    ax.add_patch(Rectangle((xo - safe_w / 2, yo - safe_l / 2), safe_w, safe_l,
                           fill=False, edgecolor="saddlebrown", linewidth=2.2))
    for lane_x in (-3.75, 0, 3.75):
        ax.axvline(lane_x, color="white", linestyle="--", linewidth=0.6, alpha=0.5)


# ============================================================
# Force-field plot
# ============================================================
def plot_force_quiver(dvx=PLOT_DVX, dvy=PLOT_DVY,
                      clip_min=FORCE_CLIP_MIN, clip_max=FORCE_CLIP_MAX,
                      arrow_color="cyan", arrow_len=0.6,
                      alpha_min=0.15, alpha_max=1.0,
                      normalize=None, ax=None, show_prompt=True):
    """
    Background = |F| on a log scale; arrows = direction only, with alpha
    encoding |F| on the same scale. Identical to k8 except that F comes from
    force_field(), which applies the |grad rho| normalisation.
    """
    if normalize is None:
        normalize = NORMALIZE_BY_GRAD_RHO

    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure

    Fx, Fy = force_field(X, Y, dvx, dvy, normalize=normalize)
    mag_full = np.hypot(Fx, Fy)

    mesh = ax.pcolormesh(X, Y, np.clip(mag_full, clip_min, clip_max),
                         cmap="inferno", norm=LogNorm(vmin=clip_min, vmax=clip_max),
                         shading="gouraud", zorder=0)
    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.04, label="|F| (log)")

    force_levels = np.geomspace(clip_min, clip_max, 8)
    cs = ax.contour(X, Y, np.clip(mag_full, clip_min, clip_max),
                    levels=force_levels, colors="white", linewidths=0.6,
                    alpha=0.6, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.3g}", fontsize=6, colors="white")

    Xq = X[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Yq = Y[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Fxq, Fyq = force_field(Xq, Yq, dvx, dvy, normalize=normalize)

    magq = np.hypot(Fxq, Fyq)
    uxq = Fxq / (magq + 1e-12)
    uyq = Fyq / (magq + 1e-12)

    mag_norm = (np.log(np.clip(magq, clip_min, clip_max) / clip_min)
                / np.log(clip_max / clip_min))
    alphas = alpha_min + mag_norm * (alpha_max - alpha_min)
    colors = np.tile(mcolors.to_rgba(arrow_color), (alphas.size, 1))
    colors[:, 3] = alphas.ravel()

    ax.quiver(Xq, Yq, uxq * arrow_len, uyq * arrow_len, color=colors,
              angles="xy", scale_units="xy", scale=1, pivot="mid",
              width=0.005, headwidth=3.5, headlength=4.5, zorder=3)

    draw_scene(ax)
    tag = r"$-\nabla U/|\nabla\rho|$" if normalize else r"$-\nabla U$"
    ax.set_title(f"Force field {tag}\n"
                 f"$\\Delta v_x$={dvx} kmph, $\\Delta v_y$={dvy} kmph", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal", adjustable="box")

    if standalone:
        plt.tight_layout()
        plt.show(block=False)
        if show_prompt:
            prompt_force_at_points([(dvx, dvy)], fig, ax=ax)
    return fig, ax


def plot_comparison(dvx=PLOT_DVX, dvy=PLOT_DVY):
    """Plain -grad U beside the |grad rho|-normalised force, same scale."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_force_quiver(dvx, dvy, normalize=False, ax=axes[0])
    plot_force_quiver(dvx, dvy, normalize=True, ax=axes[1])
    plt.tight_layout()
    plt.show(block=False)
    return fig, axes


if __name__ == "__main__":
    plot_force_quiver()
    plt.show()
