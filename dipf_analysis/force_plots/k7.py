import numpy as np
import matplotlib.pyplot as plt
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
PLOT_DVX, PLOT_DVY = 20,10

lam = 10              # scaling factor (lambda), Eq. 1
dv_max = 10.0             # normalising max relative velocity, Eq. 2
tau_x, tau_y = 1.721871478710501, 4.8648315932156687
alpha = 0.1           # velocity-skew sensitivity, Eq. 6

U_CAP = 40.0               # clip potential for plotting only (paper caps the colour scale too)
N_LEVELS = 100          # many, finely-spaced levels -> the "fan" pattern near the vehicle
LEVEL_POWER = 1.0           # linear spacing: the field itself is sharply peaked near the
                            # vehicle, so plain fine levels are enough to reveal the "bullseye"

# Quiver settings
QUIVER_STRIDE = 16          # subsample the fine grid every N points for arrows
QUIVER_STEP = 1e-3          # finite-difference step for the gradient
QUIVER_CLIP = 30.0          # clip |F| before normalizing (non-log mode only)


# ============================================================
# Eq. 2 — f(Delta v): linear, normalised relative-velocity term.
# No directional/closing-speed gating -- just the magnitude.
# ============================================================
def f_dv(dx,dy,dvx, dvy):
    #return (np.hypot(dvx, dvy)**(alpha*np.tanh(-dvx*dx - dvy*dy))) / dv_max
    # return np.hypot(dvx, dvy) / dv_max
    return (np.hypot(dvx, dvy))*0+1


# ============================================================
# Eq. 6 — final asymmetric, dimension-aware pseudo-distance k'
# ============================================================
def kprime(dx, dy, dvx, dvy):
    def tau(delta, dvel, tau0):
        return tau0 * ((2 + alpha * dvel) + (alpha * dvel ) * np.tanh(-delta * dvel)) / 2

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


def prompt_gradient_magnitudes(scenarios, fig):
    """Keep the figure responsive while accepting X, Y coordinates in the terminal."""
    entries = Queue()

    def read_entries():
        print("\nEnter X,Y to evaluate |∇U| (enter 0,0 to exit).")
        while True:
            try:
                entries.put(input("X, Y: "))
            except (EOFError, KeyboardInterrupt):
                entries.put(None)
                return

    Thread(target=read_entries, daemon=True).start()

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

        print(f"|∇U| at X={X:g}, Y={Y:g}:")
        for dvx, dvy in scenarios:
            magnitude = gradient_magnitude_at(X, Y, dvx, dvy)
            print(f"  Δvx={dvx:g}, Δvy={dvy:g}: {magnitude:.4f}")


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
    prompt_gradient_magnitudes(scenarios, fig)


def plot_force_quiver(dvx=PLOT_DVX, dvy=PLOT_DVY, fname="force_quiver.png",
                       overlay_contour=True, normalize=False, log_scale=True):
    """
    Quiver plot of the force field F = -grad(U).

    overlay_contour: draw the U contourf underneath the arrows for context.
    normalize: unit-length arrows (ignored if log_scale=True).
    log_scale: if True, both arrow length and arrow color use log(1+|F|)
               instead of raw |F| -- recommended since |F| spans orders of
               magnitude near the vehicle corners vs. the far field.
    """
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    Xq = X[::QUIVER_STRIDE, ::QUIVER_STRIDE]
    Yq = Y[::QUIVER_STRIDE, ::QUIVER_STRIDE]

    dU_dX, dU_dY = gradient_field(Xq, Yq, dvx, dvy)
    Fx, Fy = -dU_dX, -dU_dY  # force = -grad(U)

    mag = np.hypot(Fx, Fy)
    eps = 1e-12
    ux = Fx / (mag + eps)
    uy = Fy / (mag + eps)

    if log_scale:
        mag_scaled = np.log1p(mag)
        Fx_plot = ux * mag_scaled
        Fy_plot = uy * mag_scaled
        color_vals = mag
        quiver_kwargs = dict(scale=None)
        norm = LogNorm(vmin=max(mag.min(), 1e-6), vmax=mag.max())
        colorbar_label = "|F| (log scale)"
    elif normalize:
        Fx_plot, Fy_plot = ux, uy
        color_vals = np.clip(mag, 0, QUIVER_CLIP)
        quiver_kwargs = dict(scale=30, scale_units="xy")
        norm = None
        colorbar_label = "|F| (clipped)"
    else:
        Fx_plot, Fy_plot = Fx, Fy
        color_vals = np.clip(mag, 0, QUIVER_CLIP)
        quiver_kwargs = dict(scale=None)
        norm = None
        colorbar_label = "|F|"

    fig, ax = plt.subplots(figsize=(5.5, 5))

    if overlay_contour:
        levels = U_CAP * np.linspace(0, 1, N_LEVELS) ** LEVEL_POWER
        U = potential(X, Y, dvx, dvy)
        ax.contourf(X, Y, U, levels=levels, cmap="viridis", alpha=0.6)

    q = ax.quiver(Xq, Yq, Fx_plot, Fy_plot, color_vals,
                  cmap="inferno", norm=norm,
                  width=0.006, headwidth=3, headlength=4,
                  **quiver_kwargs)
    fig.colorbar(q, ax=ax, label=colorbar_label, shrink=0.8, pad=0.04)

    draw_scene(ax)

    ax.set_title(f"Force field $F=-\\nabla U$\n$\\Delta v_x$={dvx} kmph, $\\Delta v_y$={dvy} kmph",
                 fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


if __name__ == "__main__":
    # plot_scenarios()
    plot_force_quiver(log_scale=True,normalize=False)
    plt.show()