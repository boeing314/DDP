import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 1.0
x_left = -5.0
x_right = 5.0

# Avoid the singularities at x_left and x_right
x = np.linspace(x_left + 0.01, x_right - 0.01, 2000)

# Potential
U = beta * (
    (1 / (x - x_left))**2
    + (1 / (x_right - x))**2
)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, U, linewidth=2)

plt.xlabel(r"$x$")
plt.ylabel(r"$U(x)$")
plt.title("Road Potential")

plt.grid(True, alpha=0.3)
plt.ylim(0, 10)  # Adjust/remove if you want to see the singular behavior

plt.tight_layout()
plt.show()