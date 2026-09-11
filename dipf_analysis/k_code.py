import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# PARAMETERS — edit these
# ============================================================
w, l = 1.75, 4.5          # obstacle vehicle width, length (m)
ego_w, ego_l = 1.748, 4.583  # ego vehicle width, length (m), from run6 config
xo, yo = 0.0, 30.0       # obstacle vehicle centre position

# Calibrated obstacle-potential parameters from v0/run6/config.yaml.
lam = 244.96347872189534
tau_x, tau_y = 4.721871478710501, 1.8648315932156687
alpha = 1.6422052478480245
dv_max_front, dv_max_rear, dv_max_lat = 10.0, 10.0, 5.0
obs_radius_ahead, obs_radius_behind, lateral_band = 50.0, 20.0, 8.0

U_CAP = 30000               # clip potential for plotting


# ============================================================
# EQUATION — replace this function's body with any k' formula
# dx, dy   = position difference (obstacle - host)
# dvx, dvy = relative velocity (obstacle - host)
# ============================================================
def kprime(dx, dy, dvx, dvy):
    """Vectorised equivalent of ``k_prime_exact`` in the run6 controller."""
    def tau(delta, dvel, tau0):
        return tau0 * ((1 + alpha * dvel) + (alpha * dvel - 1) * np.tanh(delta * dvel)) / 2
 
    tx = tau(dx, dvx, tau_x)
    ty = tau(dy, dvy, tau_y)
 
    eps = 1e-3
    tx = np.where(np.abs(tx) < eps, eps, tx)
    ty = np.where(np.abs(ty) < eps, eps, ty)
 
    dist = np.sqrt((dx / tx) ** 2 + (dy / ty) ** 2)

    # The controller uses the combined effective vehicle dimensions.
    w_combined = (ego_w + w) / 2.0
    l_combined = (ego_l + l) / 2.0
    abs_dx = np.maximum(np.abs(dx), eps)
    abs_dy = np.maximum(np.abs(dy), eps)
    front_or_rear = abs_dy <= (l_combined * abs_dx) / w_combined

    term_x = np.maximum(0.1, 1.0 - w_combined / abs_dx)
    term_y = np.maximum(0.1, 1.0 - l_combined / abs_dy)
    return np.where(front_or_rear, term_x, term_y) * dist + 0.05
 
 
# ============================================================
# Everything below this line usually does not need to change
# ============================================================
def f_dv(dx, dy, dvx, dvy):
    """Vectorised ``f_closing_region_aware`` from the run6 controller."""
    dominant_x = np.abs(dx) >= np.abs(dy)
    front = dominant_x & (dx >= 0.0)
    rear = dominant_x & ~front

    closing_speed = np.where(
        front, np.maximum(0.0, -dvx),
        np.where(rear, np.maximum(0.0, dvx),
                 np.where(dy >= 0.0, np.maximum(0.0, -dvy),
                          np.maximum(0.0, dvy))),
    )
    dv_limit = np.where(front, dv_max_front,
                        np.where(rear, dv_max_rear, dv_max_lat))
    return np.clip(closing_speed / dv_limit, 0.0, 1.0)
 
 
def potential(X, Y, dvx, dvy):
    dx = xo - X   # Delta x = xo - xh
    dy = yo - Y   # Delta y = yo - yh
 
    k = kprime(dx, dy, dvx, dvy)
    within_range = (
        (dx <= obs_radius_ahead)
        & (dx >= -obs_radius_behind)
        & (np.abs(dy) <= lateral_band)
    )
    U = np.where(within_range, lam * f_dv(dx, dy, dvx, dvy) / k, 0.0)
    return np.clip(U, 0, U_CAP)
 
 
def plot_scenarios(title="Potential field", fname="output.png"):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(20, 40, 400)
    X, Y = np.meshgrid(x, y)

    # Each row is a lateral relative velocity and each column is longitudinal.
    velocities = (-5, 0, 5)
    scenarios = [(dvx, dvy) for dvy in velocities for dvx in velocities]
    potentials = [potential(X, Y, dvx, dvy) for dvx, dvy in scenarios]
    max_potential = max(np.max(U) for U in potentials)
    levels = np.linspace(0.0, max(max_potential, 1.0), 31)

    fig, axes = plt.subplots(
        3, 3, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True
    )
    fig.suptitle(title)

    for ax, (dvx, dvy), U in zip(axes.flat, scenarios, potentials):
        cf = ax.contourf(X, Y, U, levels=levels, cmap="jet", extend="max")
        ax.contour(X, Y, U, levels=15, colors="white", linewidths=0.3, alpha=0.6)
 
        rect = Rectangle((xo - w / 2, yo - l / 2), w, l,
                          fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)
 
        for lane_x in (-3.75, 0, 3.75):
            ax.axvline(lane_x, color="white", linestyle="--", linewidth=0.6, alpha=0.5)
 
        ax.set_title(f"$\\Delta v_x$ = {dvx}, $\\Delta v_y$ = {dvy}", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel("X")
    for ax in axes[:, 0]:
        ax.set_ylabel("Y")

    fig.colorbar(
        cf, ax=axes, orientation="horizontal", label="Obstacle potential U",
        shrink=0.8, aspect=40, pad=0.08,
    )
    plt.show()


if __name__ == "__main__":
    plot_scenarios(title=" potential field", fname="output.png")
