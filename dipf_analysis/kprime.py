import numpy as np
import matplotlib.pyplot as plt

a = 0.2
del_v_values = [3, 5, 7, 10, 15]
del_y = np.linspace(-8, 8, 1000)
plot_derivative = True  # Set to False to hide the first derivative plot

if plot_derivative:
    fig, (ax, ax_derivative) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1]
    )
else:
    fig, ax = plt.subplots(figsize=(10, 6))

for del_v in del_v_values:
    temp = 0.5 * ((1 + a * abs(del_v)) + (a * abs(del_v) - 1) * np.tanh(del_v * del_y))
    ky = del_y / temp
    u_raw = 1 / np.abs(ky)

    # Clip the potential for a cleaner visualization
    u = np.clip(u_raw, -1, 1)

    ax.plot(del_y, u, linewidth=2, label=f"del_v={del_v}")

    if plot_derivative:
        u_derivative = np.gradient(u_raw, del_y, edge_order=2)
        u_derivative = abs(np.clip(u_derivative, -0.3, 0.3))
        ax_derivative.plot(del_y, u_derivative, linewidth=2, label=f"del_v={del_v}")

ax.set_title(f"a={a}", fontsize=14, pad=15)
ax.set_xlabel('del_y', fontsize=12)
ax.set_ylabel('U', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

if plot_derivative:
    ax_derivative.set_ylabel('dU/d(del_y)', fontsize=12)
    ax_derivative.grid(True, alpha=0.3)
    ax_derivative.legend()

plt.show()