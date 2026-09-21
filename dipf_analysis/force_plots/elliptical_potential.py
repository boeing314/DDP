import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

# ============================================================
# PARAMETERS
# ============================================================
xo, yo = 0.0, 0.0        # obstacle position
tx, ty = 1,5        # elliptical pseudo-distance scales (elongation q = ty/tx)
lam = 10.0                # potential strength, U = lam / rho

NORMALIZE_FORCE = True   # False: plot F = -grad U
                           # True:  plot F = -grad U / |grad rho|   (dip removed)

XLIM, YLIM = (-6, 6), (-10, 10)  # tx > ty here -> ellipse stretched along X,
                                   # so the dip sits on a horizontal scan and
                                   # needs a Y-range this tight to be visible
GRID_N = 500

FORCE_CLIP_MIN, FORCE_CLIP_MAX = None, None  # log color scale clip.
                                              # None = auto-fit to the |F| actually
                                              # present in the current XLIM/YLIM
                                              # (see auto_force_clip()). A wide
                                              # fixed clip left over from a
                                              # different view can bury the whole
                                              # background in one color, or clip
                                              # it all to black/white; set both to
                                              # fixed numbers only if you want the
                                              # scale pinned across several plots.
QUIVER_STRIDE = 16
QUIVER_STEP = 1e-4
ARROW_LEN = 0.6   # fixed on-screen arrow length (data units), same as k8.py --
                  # |F| varies over orders of magnitude so a true-length arrow
                  # would be invisible or enormous; direction only, magnitude
                  # goes to alpha instead
EPS = 1e-9


# ============================================================
# Field definition
# ============================================================
def rho(X, Y):
    """Elliptical pseudo-distance from the obstacle."""
    return np.sqrt(((X - xo) / tx) ** 2 + ((Y - yo) / ty) ** 2)


def potential(X, Y):
    """U(x, y) = lam / rho(x, y)."""
    r = np.where(rho(X, Y) < EPS, EPS, rho(X, Y))
    return lam / r


def grad_field(f, X, Y, step=QUIVER_STEP):
    """Central-difference gradient of a scalar field f(X, Y)."""
    dfdX = (f(X + step, Y) - f(X - step, Y)) / (2 * step)
    dfdY = (f(X, Y + step) - f(X, Y - step)) / (2 * step)
    return dfdX, dfdY


def grad_rho_analytic(X, Y):
    """
    Analytic |grad rho|, used for the normalised force.

        rho = sqrt((dx/tx)^2 + (dy/ty)^2),  dx = X - xo, dy = Y - yo
        d(rho)/dX = (dx/tx^2) / rho,   d(rho)/dY = (dy/ty^2) / rho
    """
    dx, dy = X - xo, Y - yo
    r = np.where(rho(X, Y) < EPS, EPS, rho(X, Y))
    return np.hypot(dx / tx ** 2, dy / ty ** 2) / r


def auto_force_clip(mag):
    """
    Pick a log-scale (vmin, vmax) that actually spans the |F| values present,
    so the background always shows real color variation instead of being
    clipped to a single flat color by a stale fixed range.
    """
    finite = mag[np.isfinite(mag) & (mag > 0)]
    if finite.size == 0:
        return 1e-2, 1e2
    vmin = max(np.percentile(finite, 1), 1e-12)
    vmax = np.percentile(finite, 99.5)
    if vmax <= vmin:
        vmax = vmin * 10
    return vmin, vmax


def force_field(X, Y, normalize=None):
    """
    F = -grad U, optionally divided by |grad rho| (NORMALIZE_FORCE / normalize).
    Returns (Fx, Fy).
    """
    if normalize is None:
        normalize = NORMALIZE_FORCE

    dU_dX, dU_dY = grad_field(potential, X, Y)
    Fx, Fy = -dU_dX, -dU_dY

    if normalize:
        gn = grad_rho_analytic(X, Y)
        gn = np.where(gn < EPS, EPS, gn)
        Fx, Fy = Fx / gn, Fy / gn

    return Fx, Fy


# ============================================================
# Plotting
# ============================================================
def plot_field(normalize=None, ax=None, title=None):
    if normalize is None:
        normalize = NORMALIZE_FORCE

    x = np.linspace(*XLIM, GRID_N)
    y = np.linspace(*YLIM, GRID_N)
    X, Y = np.meshgrid(x, y)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure

    # --- background: |F| magnitude, log scale ---
    Fx, Fy = force_field(X, Y, normalize=normalize)
    mag = np.hypot(Fx, Fy)

    clip_min, clip_max = FORCE_CLIP_MIN, FORCE_CLIP_MAX
    if clip_min is None or clip_max is None:
        clip_min, clip_max = auto_force_clip(mag)

    mesh = ax.pcolormesh(X, Y, np.clip(mag, clip_min, clip_max),
                          cmap="inferno",
                          norm=LogNorm(vmin=clip_min, vmax=clip_max),
                          shading="gouraud", zorder=0)
    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.04, label="|F| (log)")

    # --- white contour lines of |F| itself (iso-force lines) ---
    # Same quantity as the background, same clipped range, so the contours
    # are literal isolines of the color underneath -- and, unlike contours of
    # U, these visibly pinch inward at the dip instead of staying clean nested
    # ellipses, since U has no dip but |F| does.
    force_levels = np.geomspace(clip_min, clip_max, 10)
    cs = ax.contour(X, Y, np.clip(mag, clip_min, clip_max), levels=force_levels,
                     colors="white", linewidths=0.5, alpha=0.5, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.2g}", fontsize=6, colors="white")

    # --- quiver: direction only, alpha encodes |F| on the clipped log scale ---
    Xq = X[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Yq = Y[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Fxq, Fyq = force_field(Xq, Yq, normalize=normalize)
    magq = np.hypot(Fxq, Fyq)
    uxq, uyq = Fxq / (magq + EPS), Fyq / (magq + EPS)

    mag_norm = (np.log(np.clip(magq, clip_min, clip_max) / clip_min)
                / np.log(clip_max / clip_min))
    alphas = 0.15 + mag_norm * 0.85
    colors = np.tile(mcolors.to_rgba("cyan"), (alphas.size, 1))
    colors[:, 3] = alphas.ravel()

    ax.quiver(Xq, Yq, uxq * ARROW_LEN, uyq * ARROW_LEN, color=colors,
              angles="xy", scale_units="xy", scale=1, pivot="mid",
              width=0.005, headwidth=3.5, headlength=4.5, zorder=3)

    ax.add_patch(Circle((xo, yo), 0.1, color="red", zorder=5))
    # NOTE: no crosshair lines through (xo, yo) -- at high elongation the dip
    # channel runs exactly along y=yo, and a line drawn there paints right
    # over it, hiding the very feature this plot exists to show.

    if title is None:
        tag = r"$F=-\nabla U/|\nabla\rho|$" if normalize else r"$F=-\nabla U$"
        title = f"Inverse elliptical potential  {tag}\n$t_x$={tx}, $t_y$={ty}  (q={max(tx,ty)/min(tx,ty):g})"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")

    if standalone:
        plt.tight_layout()
    return fig, ax


def plot_comparison():
    """Plain -grad U next to the normalised force, same parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_field(normalize=False, ax=axes[0])
    plot_field(normalize=True, ax=axes[1])
    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    plot_field()
    plt.show()
