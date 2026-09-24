"""
Three-panel illustration of the factorisation |grad U| = |U'(rho)| * |grad rho|,
matching the numeric example in grad_rho_intuition.tex (Section "Where the dip
actually comes from"): scanning sideways at a fixed distance ahead of an
elliptical obstacle with t_x = 1, t_y = 15.

Panel 1: |grad rho|   - the exchange rate, climbs monotonically outward (the artefact)
Panel 2: |U'(rho)|    - the honest quantity, falls monotonically outward (monotone)
Panel 3: |grad U|     - their product, the force actually felt (non-monotone: the dip/ridge)
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Parameters (match the worked example in the note)
# ==========================================
T_X = 1.0        # ellipse half-width scale, minor (lateral) axis
T_Y = 15.0       # ellipse half-width scale, major (longitudinal) axis -> elongation q = 15
DY = 9.0         # fixed distance ahead of the obstacle (metres)
LAMBDA = 10.0    # U(rho) = LAMBDA / rho  =>  U'(rho) = -LAMBDA / rho^2

X_RANGE = np.linspace(-3.0, 3.0, 400)  # lateral offset x (metres)

# Reference palette (validated categorical slots 1-3)
COLOR_GRAD_RHO = "#2a78d6"   # blue
COLOR_U_PRIME = "#eb6834"    # orange
COLOR_GRAD_U = "#1baf7a"     # aqua
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def rho(x, dy=DY, tx=T_X, ty=T_Y):
    """Elliptical risk coordinate."""
    return np.sqrt((x / tx) ** 2 + (dy / ty) ** 2)


def grad_rho_mag(x, dy=DY, tx=T_X, ty=T_Y):
    """|grad rho| -- the exchange rate (metres of rho per metre of space)."""
    r = rho(x, dy, tx, ty)
    return np.sqrt((x / tx ** 2) ** 2 + (dy / ty ** 2) ** 2) / r


def u_prime_mag(x, lam=LAMBDA, dy=DY, tx=T_X, ty=T_Y):
    """|U'(rho)| -- the honest, monotone quantity."""
    r = rho(x, dy, tx, ty)
    return lam / r ** 2


def grad_u_mag(x, lam=LAMBDA, dy=DY, tx=T_X, ty=T_Y):
    """|grad U| = |U'(rho)| * |grad rho| -- the force actually felt."""
    return u_prime_mag(x, lam, dy, tx, ty) * grad_rho_mag(x, dy, tx, ty)


def style_axis(ax, title, ylabel, color):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=10)
    ax.set_xlabel("x", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)


def main():
    y_grad_rho = grad_rho_mag(X_RANGE)
    y_u_prime = u_prime_mag(X_RANGE)
    y_grad_u = grad_u_mag(X_RANGE)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.set_facecolor(SURFACE)

    # Panel 1: |grad rho| -- the artefact, climbs monotonically
    ax = axes[0]
    ax.plot(X_RANGE, y_grad_rho, color=COLOR_GRAD_RHO, linewidth=2, zorder=3)
    style_axis(ax, "|∇ρ|",
               "|∇ρ|", COLOR_GRAD_RHO)

    # Panel 2: |U'(rho)| -- the honest quantity, falls monotonically
    ax = axes[1]
    ax.plot(X_RANGE, y_u_prime, color=COLOR_U_PRIME, linewidth=2, zorder=3)
    style_axis(ax, "|U'(ρ)|",
               "|U'(ρ)|", COLOR_U_PRIME)

    # Panel 3: |grad U| -- the product, non-monotone (the dip and ridge)
    ax = axes[2]
    ax.plot(X_RANGE, y_grad_u, color=COLOR_GRAD_U, linewidth=2, zorder=3)
    peak_idx = np.argmax(y_grad_u)
    # ax.scatter(X_RANGE[peak_idx], y_grad_u[peak_idx], color=COLOR_GRAD_U,
    #            s=36, zorder=4, edgecolor=SURFACE, linewidth=1)
    # ax.annotate(f"ridge\nx≈{X_RANGE[peak_idx]:.2f}",
    #             xy=(X_RANGE[peak_idx], y_grad_u[peak_idx]),
    #             xytext=(X_RANGE[peak_idx] + 0.5, y_grad_u[peak_idx] * 0.85),
    #             fontsize=9, color=INK_SECONDARY)
    # ax.annotate("dip", xy=(0, grad_u_mag(np.array([0.0]))[0]),
    #             xytext=(0.15, grad_u_mag(np.array([0.0]))[0] * 1.6),
    #             fontsize=9, color=INK_SECONDARY,
    #             arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.8))
    style_axis(ax, "|∇U| = |U'(ρ)|×|∇ρ|",
               "|∇U|", COLOR_GRAD_U)

    # fig.suptitle(
    #     "Where the dip comes from: a monotone decay times a monotone climb",
    #     fontsize=14, color=INK_PRIMARY, y=1.03
    # )
    plt.tight_layout()
    plt.savefig("grad_rho_intuition_plots.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
