import numpy as np
import matplotlib.pyplot as plt

a = 0.2
del_v_values = [-5,-3,-1,1,3,5]
del_y = np.linspace(1, 8, 1000)

fig, ax = plt.subplots(figsize=(10, 6))
for del_v in del_v_values:
    temp = 0.5 * ((1 + a*del_v) + (a*del_v - 1) * np.tanh(del_v * del_y))
    ky = del_y / temp
    u = 1 / ky
    ax.plot(del_y, u, linewidth=2, label=f"del_v={del_v}")

ax.set_title(f"a={a}", fontsize=14, pad=15)
ax.set_xlabel('del_y', fontsize=12)
ax.set_ylabel('U', fontsize=12)
ax.legend()

plt.show()