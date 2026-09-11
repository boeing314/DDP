import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from queue import Empty, Queue
from threading import Thread

# ============================================================
# PARAMETERS
# ============================================================
w, l = 1.75, 4.5         # obstacle vehicle width, length (m)  -- used in Delta x, Delta y (Eq. 5)
host_w, host_l = 1.75, 4.5   # assumed host vehicle size, only used to draw the
                              # white "combined footprint" safety box seen in the paper's figure
xo, yo = 0.0, 30.0       # obstacle vehicle centre position
# Selected relative velocity for the single plot (kmph).
PLOT_DVX, PLOT_DVY = 2.5, -1.5

lam = 10              # scaling factor (lambda), Eq. 1
dv_max = 10.0             # normalising max relative velocity, Eq. 2
tau_x, tau_y = 1.721871478710501, 4.8648315932156687
alpha = 1            # velocity-skew sensitivity, Eq. 5

U_CAP = 40.0               # clip potential for plotting only (paper caps the colour scale too)
N_LEVELS = 100            # many, finely-spaced levels -> the "fan" pattern near the vehicle
LEVEL_POWER = 1.0           # linear spacing: the field itself is sharply peaked near the
                            # vehicle, so plain fine levels are enough to reveal the "bullseye"


# ============================================================
# Eq. 2 — f(Delta v): linear, normalised relative-velocity term.
# No directional/closing-speed gating -- just the magnitude.
# ============================================================
def f_dv(dvx, dvy):
    return np.hypot(dvx, dvy) / dv_max


# ============================================================
# Eq. 5 (Table 1) — edge-to-edge Delta x, Delta y (center-shifting rationale).
# Delta x / Delta y collapse to 0 whenever the host sits within the obstacle's
# width/length band -- this is the "otherwise: 0" branch that produces the
# flat vertical strip in Fig. 6 once combined with the velocity-shifted
# pseudo-distance below.
# ============================================================
def delta_x_edge(X):
    xh = X
    return np.where(
        xh < (xo - w), (xh + w / 2) - (xo - w / 2),
        np.where(xh > (xo + w), (xh - w / 2) - (xo + w / 2), 0.0),
    )


def delta_y_edge(Y):
    yh = Y
    return np.where(
        yh < (yo - l), (yh + l / 2) - (yo - l / 2),
        np.where(yh > (yo + l), (yh - l / 2) - (yo + l / 2), 0.0),
    )


# ============================================================
# Eq. 5 — center-shifting pseudo-distance k'
#
#   k' = sqrt( ((Dx - a*dvx) / (tau_x * a * dvx))^2
#            + ((Dy - a*dvy) / (tau_y * a * dvy))^2 )
#
# When a relative velocity component is exactly 0, its denominator is 0:
#   - if the numerator is also ~0 (host inside the obstacle's band, so
#     Delta=0), the term is treated as 0 -- it simply drops out, leaving
#     the other axis to govern the potential (this is the "flat plateau
#     inside the band" behaviour discussed for Fig. 6b/c).
#   - if the numerator is not 0 (host outside the band), the term blows
#     up to +inf, sending k' -> inf and U -> 0 (the dark background
#     outside the vertical strip in Fig. 6b/c).
# ============================================================
def kprime(dx, dy, dvx, dvy):
    eps = 1e-9
    BIG = 1e6  # stand-in for "this term diverges to infinity"

    def term(delta, dvel, tau0):
        numer = delta - alpha * dvel
        denom = tau0 * alpha * dvel
        zero_denom = np.abs(denom) < eps
        zero_numer = np.abs(numer) < eps
        safe_denom = np.where(zero_denom, eps, denom)
        finite_term = (numer / safe_denom) ** 2
        # where denom ~ 0: 0/0 -> 0 (term drops out); nonzero/0 -> BIG (diverges)
        return np.where(zero_denom, np.where(zero_numer, 0.0, BIG), finite_term)

    term_x = term(dx, dvx, tau_x)
    term_y = term(dy, dvy, tau_y)
    return np.sqrt(term_x + term_y)


# ============================================================
# Eq. 1 — U = lambda * f(Delta v) / k'
# ============================================================
def potential(X, Y, dvx, dvy):
    dx = delta_x_edge(X)   # Delta x, Eq. 5 (edge-to-edge, center-shifting)
    dy = delta_y_edge(Y)   # Delta y, Eq. 5

    k = kprime(dx, dy, dvx, dvy)
    k = np.where(k <= 0, 1e-4, k)   # guard divide-by-zero only, no artificial floor/mask
    U = lam * f_dv(dvx, dvy) / np.abs(k)
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
# Plotting — reproduces the paper's Figure 6 (Eq. 5) panel layout
# ============================================================
def plot_scenarios(title="Eq. 5 potential field (center-shifting)", fname="output.png"):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    # Power-law spaced levels: dense near U_CAP (fine rings close to the
    # vehicle), sparse near 0 (smooth far field) -- this is what produces
    # the "fan"/asymptote look radiating from the vehicle corners.
    levels = U_CAP * np.linspace(0, 1, N_LEVELS) ** LEVEL_POWER

    scenarios = [(PLOT_DVX, PLOT_DVY)]
    fig, axes = plt.subplots(1, 1, figsize=(5, 4.5), sharey=True, squeeze=False)
    axes = axes.ravel()

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
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Y")
    fig.colorbar(
        cf, ax=axes, orientation="vertical", label="Potential $U$",
        shrink=0.8, aspect=40, pad=0.04,
    )
    fig.suptitle(title)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    prompt_gradient_magnitudes(scenarios, fig)


if __name__ == "__main__":
    plot_scenarios()
