import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# PARAMETERS
# ============================================================
w, l = 1.75, 4.5         # obstacle vehicle width, length (m)  -- Eq. 6 uses only these
host_w, host_l = 1.75, 4.5   # assumed host vehicle size, only used to draw the
                              # white "combined footprint" safety box seen in the paper's figure
xo, yo = 0.0, 30.0       # obstacle vehicle centre position

lam = 244.96347872189534               # scaling factor (lambda), Eq. 1
dv_max = 10.0             # normalising max relative velocity, Eq. 2
tau_x, tau_y = 4.721871478710501, 1.8648315932156687
alpha = 1.6422052478480245            # velocity-skew sensitivity, Eq. 6

U_CAP = 400000000000000000.0               # clip potential for plotting only (paper caps the colour scale too)
GRADIENT_CAP = 40.0        # maximum displayed |∇U|; larger values use the top colour
N_LEVELS = 150             # many, finely-spaced levels -> the "fan" pattern near the vehicle
LEVEL_POWER = 1.0           # linear spacing: the field itself is sharply peaked near the
                            # vehicle, so plain fine levels are enough to reveal the "bullseye"


# ============================================================
# Eq. 2 — f(Delta v): linear, normalised relative-velocity term.
# No directional/closing-speed gating -- just the magnitude.
# ============================================================
def f_dv(dvx, dvy):
    return np.hypot(dvx, dvy) / dv_max


# ============================================================
# Eq. 6 — final asymmetric, dimension-aware pseudo-distance k'
# ============================================================
def kprime(dx, dy, dvx, dvy):
    def tau(delta, dvel, tau0):
        return tau0 * ((1 + alpha * dvel) + (alpha * dvel - 1) * np.tanh(delta * dvel)) / 2

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

    U = lam * f_dv(dvx, dvy) / np.abs(k)
    return np.clip(U, -10, U_CAP)


# ============================================================
# Potential-gradient magnitude: |∇U| at every spatial grid point
# ============================================================
def gradient_magnitude(U, x, y):
    """Return |∇U|, with derivatives taken with respect to physical X and Y."""
    dU_dy, dU_dx = np.gradient(U, y, x)
    return np.hypot(dU_dx, dU_dy)


def plot_scenarios(title="Potential-gradient magnitude", fname="output.png"):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    scenarios = [(0, 0), (0, -0.5), (0, -1.5), (0.5, -1)]   # matches paper's Fig. 4-7 cases
    gradient_fields = [
        np.clip(gradient_magnitude(potential(X, Y, dvx, dvy), x, y), 0, GRADIENT_CAP)
        for dvx, dvy in scenarios
    ]
    max_gradient = max(np.max(field) for field in gradient_fields)
    # Keep one shared scale so magnitudes can be compared between scenarios.
    levels = np.linspace(0, max(max_gradient, 1.0), N_LEVELS)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

    for ax, (dvx, dvy), grad_U in zip(axes, scenarios, gradient_fields):
        cf = ax.contourf(X, Y, grad_U, levels=levels, cmap="turbo")
        if np.ptp(grad_U) > 0:
            ax.contour(X, Y, grad_U, levels=levels, colors="white", linewidths=0.35, alpha=0.8)

        # obstacle vehicle footprint
        rect = Rectangle((xo - w / 2, yo - l / 2), w, l,
                          fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)

        # combined host+obstacle "safety box" (illustrative, not part of U itself)
        safe_w, safe_l = w + host_w, l + host_l
        safe_rect = Rectangle((xo - safe_w / 2, yo - safe_l / 2), safe_w, safe_l,
                               fill=False, edgecolor="white", linewidth=2.2)
        ax.add_patch(safe_rect)

        for lane_x in (-3.75, 0, 3.75):
            ax.axvline(lane_x, color="white", linestyle="--", linewidth=0.6, alpha=0.5)

        ax.set_title(f"$\\Delta v_x$={dvx} kmph\n$\\Delta v_y$={dvy} kmph", fontsize=10)
        ax.set_xlabel("X")

    axes[0].set_ylabel("Y")
    fig.colorbar(
        cf, ax=axes, orientation="horizontal", label=r"Potential-gradient magnitude $|\nabla U|$",
        shrink=0.8, aspect=40, pad=0.14,
    )
    plt.show()


if __name__ == "__main__":
    plot_scenarios()
