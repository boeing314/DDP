import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from queue import Empty, Queue
from threading import Thread

# ============================================================
# PARAMETERS
# ============================================================
w, l = 1.75, 4.5         # obstacle vehicle width, length (m)  -- Eq. 6 uses only these
host_w, host_l = 1.75, 4.5   # assumed host vehicle size, only used to draw the
                              # white "combined footprint" safety box seen in the paper's figure
xo, yo = 0.0, 30.0       # obstacle vehicle centre position

lam = 10              # scaling factor (lambda), Eq. 1
dv_max = 10.0             # normalising max relative velocity, Eq. 2
tau_x, tau_y = 1.721871478710501, 4.8648315932156687
alpha = 1e2            # velocity-skew sensitivity, Eq. 6

U_CAP = 100.0               # clip potential for plotting only (paper caps the colour scale too)
N_LEVELS = 300            # many, finely-spaced levels -> the "fan" pattern near the vehicle
LEVEL_POWER = 1.0           # linear spacing: the field itself is sharply peaked near the
                            # vehicle, so plain fine levels are enough to reveal the "bullseye"


# ============================================================
# Eq. 2 — f(Delta v): linear, normalised relative-velocity term.
# No directional/closing-speed gating -- just the magnitude.
# ============================================================
def f_dv(dvx, dvy, dx, dy):
    return (np.hypot(dvx, dvy)**np.tanh(-dvx*dx - dvy*dy)) / dv_max


# ============================================================
# Eq. 6 — final asymmetric, dimension-aware pseudo-distance k'
# ============================================================
def kprime(dx, dy, dvx, dvy):
    def tau(delta, dvel, tau0):
        return tau0 / (np.tanh(-alpha * delta * dvel)+1.01)

    tx = tau_x#tau(dx, dvx, tau_x)
    ty = tau_y#tau(dy, dvy, tau_y)

    eps = 1e-6
    # tx = np.where(np.abs(tx) < eps, eps, tx)
    # ty = np.where(np.abs(ty) < eps, eps, ty)

    dist = np.sqrt((dx / tx) ** 2 + (dy / ty) ** 2)

    region13 = np.abs(dy) <= (l * np.abs(dx)) / w   # front/behind regions (1 & 3)

    # Vehicle-dimension term is purely geometric (no velocity dependence),
    # so it must use the *magnitude* of the gap -- not the signed value --
    # or the field becomes left/right (or front/back) asymmetric even when
    # Delta v = 0, which the paper's figures show it should not be.
    abs_dx = np.where(np.abs(dx) < eps, eps, np.abs(dx))
    abs_dy = np.where(np.abs(dy) < eps, eps, np.abs(dy))

    rel_vel=dx*dvx+dy*dvy
    k_13 = (1 - w / abs_dx) * dist#*np.tanh(-alpha*rel_vel)
    k_24 = (1 - l / abs_dy) * dist#*np.tanh(-alpha*rel_vel)

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
    U = lam * f_dv(dvx, dvy,dx,dy) / np.abs(k)
    return np.clip(U, -10, U_CAP)


def gradient_magnitude_at(X, Y, dvx, dvy, step=1e-3):
    """Numerically evaluate |∇U| at one position using central differences."""
    dU_dX = (potential(X + step, Y, dvx, dvy) -
             potential(X - step, Y, dvx, dvy)) / (2 * step)
    dU_dY = (potential(X, Y + step, dvx, dvy) -
             potential(X, Y - step, dvx, dvy)) / (2 * step)
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


# ============================================================
# Plotting — reproduces the paper's 1x4 panel layout (Fig. 7 style)
# ============================================================
def plot_scenarios(title="Eq. 6 potential field", fname="output.png"):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    # Power-law spaced levels: dense near U_CAP (fine rings close to the
    # vehicle), sparse near 0 (smooth far field) -- this is what produces
    # the "fan"/asymptote look radiating from the vehicle corners.
    levels = U_CAP * np.linspace(0, 1, N_LEVELS) ** LEVEL_POWER

    scenarios = [(0, 0), (0, -0.5), (-1.5,0), (2.5, -1)]   # matches paper's Fig. 4-7 cases
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

    for ax, (dvx, dvy) in zip(axes, scenarios):
        U = potential(X, Y, dvx, dvy)
        cf = ax.contourf(X, Y, U, levels=levels, cmap="viridis")
        ax.contour(X, Y, U, levels=levels, colors="white", linewidths=0.35, alpha=0.8)

        # obstacle vehicle footprint
        rect = Rectangle((xo - w / 2, yo - l / 2), w, l,
                          fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)

        # combined host+obstacle "safety box" (illustrative, not part of U itself)
        safe_w, safe_l = w + host_w, l + host_l
        safe_rect = Rectangle((xo - safe_w / 2, yo - safe_l / 2), safe_w, safe_l,
                               fill=False, edgecolor="saddlebrown", linewidth=2.2)
        ax.add_patch(safe_rect)

        for lane_x in (-3.75, 0, 3.75):
            ax.axvline(lane_x, color="white", linestyle="--", linewidth=0.6, alpha=0.5)

        ax.set_title(f"$\\Delta v_x$={dvx} kmph\n$\\Delta v_y$={dvy} kmph", fontsize=10)
        ax.set_xlabel("X")

    axes[0].set_ylabel("Y")
    fig.colorbar(
        cf, ax=axes, orientation="horizontal", label="Potential $U$",
        shrink=0.8, aspect=40, pad=0.14,
    )
    plt.show(block=False)
    prompt_gradient_magnitudes(scenarios, fig)


if __name__ == "__main__":
    plot_scenarios()
