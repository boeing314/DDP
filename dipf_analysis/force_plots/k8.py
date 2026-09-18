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
w, l = 1.75, 4.5         # obstacle vehicle width, length (m)  -- Eq. 6 uses only these
host_w, host_l = 1.75, 4.5   # assumed host vehicle size, only used to draw the
                              # white "combined footprint" safety box seen in the paper's figure
xo, yo = 0.0, 30.0       # obstacle vehicle centre position
# Selected relative velocity for the single plot (kmph).
PLOT_DVX, PLOT_DVY = 3,3

lam = 10              # scaling factor (lambda), Eq. 1
dv_max = 10.0             # normalising max relative velocity, Eq. 2
tau_x, tau_y = 1,3
alpha = 1           # velocity-skew sensitivity, Eq. 6

U_CAP = 4000.0               # clip potential for plotting only (paper caps the colour scale too)
N_LEVELS = 100          # many, finely-spaced levels -> the "fan" pattern near the vehicle
LEVEL_POWER = 1.0           # linear spacing: the field itself is sharply peaked near the
                            # vehicle, so plain fine levels are enough to reveal the "bullseye"

# Quiver settings
QUIVER_STRIDE = 16          # subsample the fine grid every N points for arrows
QUIVER_STEP = 1e-4          # finite-difference step for the gradient

# Single-point force-arrow settings (used by the interactive prompt)
POINT_ARROW_LEN = 1.0       # fixed on-screen arrow length (data units), since |F| varies wildly
POINT_ARROW_COLOR = "cyan"

# Force clipping bounds -- used for the background color scale, the quiver
# arrow alpha (transparency), and the point-arrow annotations.
FORCE_CLIP_MIN = 1.0
FORCE_CLIP_MAX = 100.0


# ============================================================
# Eq. 2 — f(Delta v): linear, normalised relative-velocity term.
# No directional/closing-speed gating -- just the magnitude.
# ============================================================
def f_dv(dx,dy,dvx, dvy):
    #return (np.hypot(dvx, dvy)**(alpha*np.tanh(-dvx*dx - dvy*dy))) / dv_max
    return dvx*0+1
    return np.hypot(dvx, dvy) / dv_max  


# ============================================================
# Eq. 6 — final asymmetric, dimension-aware pseudo-distance k'
# ============================================================
def kprime(dx, dy, dvx, dvy):
    def tau(delta, dvel, tau0):
        # return tau0
        return tau0 * ((1 + alpha * abs(dvel)) + (alpha * abs(dvel) -1) * np.tanh(-delta * dvel)) / 2

    tx = tau(dx, dvx, tau_x)
    ty = tau(dy, dvy, tau_y)

    eps = 1e-6
    tx = np.where(np.abs(tx) < eps, eps, tx)
    ty = np.where(np.abs(ty) < eps, eps, ty)

    dist = np.sqrt((dx / tx) ** 2 + (dy / ty) ** 2)

    region13 = np.abs(dy) <= (l * np.abs(dx)) / w   # front/behind regions (1 & 3)

    # Vehicle-dimension term is purely geometric (no velocity dependence),
    # so it must use the *magnitude* of the gap -- not the signed value --
    # or the field becomes left/right (or front/back) asymmetric even when
    # Delta v = 0, which the paper's figures show it should not be.
    abs_dx = np.where(np.abs(dx) < eps, eps, np.abs(dx))
    abs_dy = np.where(np.abs(dy) < eps, eps, np.abs(dy))

    k_13 = (1 - w / abs_dx) * dist
    k_24 = (1 - l / abs_dy) * dist

    return np.where(region13, k_13, k_24)


# ============================================================
# Eq. 1 — U = lambda * f(Delta v) / k'
# ============================================================
def potential(X, Y, dvx, dvy):
    dx = xo - X   # Delta x = xo - xh
    dy = yo - Y   # Delta y = yo - yh

    k = kprime(dx, dy, dvx, dvy)
    k = np.where(k <= 0, 1e-4, k)   # guard divide-by-zero only, no artificial floor/mask
    U = lam  / np.abs(k)
    U = lam * f_dv(dx, dy, dvx, dvy) / np.abs(k)
    return np.clip(U, -10, U_CAP)


def gradient_field(X, Y, dvx, dvy, step=QUIVER_STEP):
    """Central-difference gradient of U over an array of (X, Y) points.
    Returns dU/dX, dU/dY, same shape as X/Y."""
    dU_dX = (potential(X + step, Y, dvx, dvy) -
             potential(X - step, Y, dvx, dvy)) / (2 * step)
    dU_dY = (potential(X, Y + step, dvx, dvy) -
             potential(X, Y - step, dvx, dvy)) / (2 * step)
    return dU_dX, dU_dY


def gradient_magnitude_at(X, Y, dvx, dvy, step=1e-3):
    """Numerically evaluate |∇U| at one position using central differences."""
    dU_dX, dU_dY = gradient_field(np.array(X), np.array(Y), dvx, dvy, step)
    return float(np.hypot(dU_dX, dU_dY))


def force_at_point(X, Y, dvx, dvy, step=QUIVER_STEP):
    """Evaluate F = -grad(U) at a single (X, Y). Returns (Fx, Fy, |F|)."""
    dU_dX, dU_dY = gradient_field(np.array(X), np.array(Y), dvx, dvy, step)
    Fx, Fy = -float(dU_dX), -float(dU_dY)
    mag = float(np.hypot(Fx, Fy))
    return Fx, Fy, mag


def plot_force_at_point(ax, X, Y, dvx, dvy, color=POINT_ARROW_COLOR,
                         arrow_len=POINT_ARROW_LEN, label_prefix=""):
    """
    Draw the force vector F=-grad(U) at a single point on an existing axes.
    Direction is exact; on-screen length is fixed (arrow_len) since |F| can
    span orders of magnitude, so a true-length arrow would often be invisible
    or huge. The actual magnitude is annotated as text instead.
    Returns (Fx, Fy, |F|).
    """
    Fx, Fy, mag = force_at_point(X, Y, dvx, dvy)

    if mag > 1e-9:
        ux, uy = Fx / mag, Fy / mag
    else:
        ux, uy = 0.0, 0.0

    ax.plot(X, Y, marker="o", color=color, markersize=5, zorder=5)
    ax.annotate(
        "", xy=(X + ux * arrow_len, Y + uy * arrow_len), xytext=(X, Y),
        arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=5,
    )
    ax.annotate(
        f"{label_prefix}|F|={mag:.3g}", xy=(X, Y), xytext=(6, 6),
        textcoords="offset points", color=color, fontsize=8, zorder=5,
    )
    return Fx, Fy, mag


def prompt_force_at_points(scenarios, fig, ax=None):
    """
    Keep the figure responsive while accepting X, Y coordinates in the
    terminal. Prints |grad U| for each scenario, and -- if `ax` is given --
    also draws the force vector F=-grad(U) at that point directly on the
    figure (one arrow per scenario, since each has a different dvx, dvy).
    """
    entries = Queue()

    def read_entries():
        print("\nEnter X,Y to evaluate |∇U| / F (enter 0,0 to exit).")
        while True:
            try:
                entries.put(input("X, Y: "))
            except (EOFError, KeyboardInterrupt):
                entries.put(None)
                return

    Thread(target=read_entries, daemon=True).start()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(scenarios), 1)))

    while plt.fignum_exists(fig.number):
        # Process GUI events while the input thread waits on the terminal.
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

        print(f"At X={X:g}, Y={Y:g}:")
        for (dvx, dvy), color in zip(scenarios, colors):
            magnitude = gradient_magnitude_at(X, Y, dvx, dvy)
            print(f"  Δvx={dvx:g}, Δvy={dvy:g}: |∇U|={magnitude:.4f}")

            if ax is not None:
                label = f"Δv=({dvx:g},{dvy:g}) " if len(scenarios) > 1 else ""
                Fx, Fy, mag = plot_force_at_point(
                    ax, X, Y, dvx, dvy, color=color, label_prefix=label
                )
                print(f"    F=(-∂U/∂X, -∂U/∂Y)=({Fx:.4f}, {Fy:.4f}), |F|={mag:.4f}")

        if ax is not None:
            fig.canvas.draw_idle()


def draw_scene(ax, xo=xo, yo=yo, w=w, l=l, host_w=host_w, host_l=host_l):
    """Shared decorations: obstacle box, safety box, lane lines."""
    rect = Rectangle((xo - w / 2, yo - l / 2), w, l,
                      fill=False, edgecolor="red", linewidth=1.5)
    ax.add_patch(rect)

    safe_w, safe_l = w + host_w, l + host_l
    safe_rect = Rectangle((xo - safe_w / 2, yo - safe_l / 2), safe_w, safe_l,
                           fill=False, edgecolor="saddlebrown", linewidth=2.2)
    ax.add_patch(safe_rect)

    for lane_x in (-3.75, 0, 3.75):
        ax.axvline(lane_x, color="white", linestyle="--", linewidth=0.6, alpha=0.5)


# ============================================================
# Plotting — reproduces the paper's 1x4 panel layout (Fig. 7 style)
# ============================================================
def plot_scenarios(title="Eq. 6 potential field", fname="output.png"):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    levels = U_CAP * np.linspace(0, 1, N_LEVELS) ** LEVEL_POWER

    scenarios = [(PLOT_DVX, PLOT_DVY)]
    fig, axes = plt.subplots(1, 1, figsize=(5, 4.5), sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, (dvx, dvy) in zip(axes, scenarios):
        U = potential(X, Y, dvx, dvy)
        cf = ax.contourf(X, Y, U, levels=levels, cmap="viridis")
        ax.contour(X, Y, U, levels=levels, colors="white", linewidths=0.35, alpha=0.8)
        draw_scene(ax)

        ax.set_title(f"$\\Delta v_x$={dvx} kmph\n$\\Delta v_y$={dvy} kmph", fontsize=10)
        ax.set_xlabel("X")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Y")
    fig.colorbar(
        cf, ax=axes, orientation="vertical", label="Potential $U$",
        shrink=0.8, aspect=40, pad=0.04,
    )
    plt.show(block=False)
    prompt_force_at_points(scenarios, fig, ax=axes[0])


def plot_force_quiver(dvx=PLOT_DVX, dvy=PLOT_DVY, fname="force_quiver.png",
                       clip_min=FORCE_CLIP_MIN, clip_max=FORCE_CLIP_MAX,
                       arrow_color="cyan", arrow_len=0.6,
                       alpha_min=0.15, alpha_max=1.0):
    """
    Force field F = -grad(U), split across two visual channels so both the
    near-singularity corners and the near-zero far field stay legible:

      - background: |F| (log-scaled, clipped to [clip_min, clip_max]) as a
        smooth color field -- shows the overall magnitude landscape.
      - quiver arrows: all drawn at the *same* on-screen length (direction
        only), with transparency (alpha) encoding |F| on the same clipped
        log scale as the background -- faint arrows = weak force, opaque
        arrows = strong force. A true-length arrow would be invisible in the
        far field or enormous near the vehicle corners, so length is freed
        up to just show direction cleanly.
    """
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # --- background: |F| magnitude over the full fine grid ---
    dU_dX, dU_dY = gradient_field(X, Y, dvx, dvy)
    mag_full = np.hypot(dU_dX, dU_dY)

    mesh = ax.pcolormesh(X, Y, np.clip(mag_full, clip_min, clip_max),
                          cmap="inferno", norm=LogNorm(vmin=clip_min, vmax=clip_max),
                          shading="gouraud", zorder=0)
    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.04,
                 label=f"|F| (log)")

    # iso-|F| contour lines (log-spaced) traced directly on the force
    # magnitude, labeled so the background color can be read off precisely
    force_levels = np.geomspace(clip_min, clip_max, 8)
    cs = ax.contour(X, Y, np.clip(mag_full, clip_min, clip_max), levels=force_levels,
                     colors="white", linewidths=0.6, alpha=0.6, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.3g}", fontsize=6, colors="white")

    # --- arrows: subsampled grid, uniform length, alpha encodes |F| ---
    Xq = X[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Yq = Y[::QUIVER_STRIDE, ::QUIVER_STRIDE]

    dU_dXq, dU_dYq = gradient_field(Xq, Yq, dvx, dvy)
    Fxq, Fyq = -dU_dXq, -dU_dYq  # force = -grad(U)

    magq = np.hypot(Fxq, Fyq)
    eps = 1e-12
    uxq = Fxq / (magq + eps)
    uyq = Fyq / (magq + eps)

    mag_norm = np.log(np.clip(magq, clip_min, clip_max) / clip_min) / np.log(clip_max / clip_min)
    alphas = alpha_min + mag_norm * (alpha_max - alpha_min)

    colors = np.tile(mcolors.to_rgba(arrow_color), (alphas.size, 1))
    colors[:, 3] = alphas.ravel()

    ax.quiver(Xq, Yq, uxq * arrow_len, uyq * arrow_len, color=colors,
              angles="xy", scale_units="xy", scale=1, pivot="mid",
              width=0.005, headwidth=3.5, headlength=4.5, zorder=3)

    draw_scene(ax)

    ax.set_title(f"Force field \n$\\Delta v_x$={dvx} kmph, $\\Delta v_y$={dvy} kmph",
                 fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show(block=False)

    prompt_force_at_points([(dvx, dvy)], fig, ax=ax)
    return fig, ax


if __name__ == "__main__":
    # plot_scenarios()
    plot_force_quiver()
    plt.show()