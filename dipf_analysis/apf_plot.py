import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import art3d

# ==========================================
# CUSTOMIZABLE PARAMETERS
# ==========================================
# Road dimensions
X_START = 0
X_END = 1000
Y_MIN = -200
Y_MAX = 200

# Start point and Goal Line
START_POS = np.array([0.0, 0.0])
GOAL_X = 1000.0  # Goal is anywhere along the line X = 1000

# Obstacles: Array of [x, y] coordinates
OBSTACLES = np.array([
[500,0],[650,-40],[850,60]
])

# Khatib's APF Parameters
XI = 0.01          # Attractive gain (pull towards goal line)
ETA = 3e6         # Repulsive gain (push away from obstacles)
RHO_0 = 1200.0      # Limit distance of obstacle influence (radius), used in the potential/force calculations
RHO_0_DISPLAY = 40.0  # Obstacle disk radius drawn in the floor projection (purely visual, independent of RHO_0)

# Road boundary (edge) potential parameters
# U_road = BETA_ROAD * [ (1/(y - Y_MIN))^2 + (1/(Y_MAX - y))^2 ]
BETA_ROAD = 3e6*1   # Repulsive gain for the road edges (Y_MIN / Y_MAX)

# Visualization parameters
MAX_POTENTIAL = 4000       # Applied ONLY to the repulsive spikes now
AXIS_RATIO = (10, 4, 5)
BACKGROUND_COLOR = "#eed47b"  # Plot background colour (hex)
LEGEND_BACKGROUND_COLOR = "#eed47b"  # Legend box background colour (hex)

# Trajectory simulation parameters
SHOW_TRAJECTORY = True        # Master switch: simulate & plot the trajectory at all
SHOW_TRAJECTORY_LINE = True   # Show the actual trajectory line elevated on the potential surface
SHOW_TRAJECTORY_PROJECTION = True  # Show the trajectory's projection onto the XY (floor) plane
PROJECTION_ONLY = True       # If True, render only the flat top-down XY projection (no 3D potential surface)
STEP_SIZE = 2.0
MAX_ITERATIONS = 2000
GOAL_TOLERANCE = 10.0      # Distance to the X=1000 line to consider "arrived"
# ==========================================


def calculate_attractive_potential(X, goal_x, xi):
    """Calculate the attractive potential toward the X=goal line. Y is ignored."""
    return 0.5 * xi * (X - goal_x)**2


def calculate_repulsive_potential(X, Y, obstacles, eta, rho_0):
    """Calculate the repulsive potential."""
    U_rep = np.zeros_like(X)
    for obs in obstacles:
        rho = np.sqrt((X - obs[0])**2 + (Y - obs[1])**2)
        rho = np.where(rho == 0, 0.1, rho)
        influence_mask = rho <= rho_0
        U_rep[influence_mask] += 0.5 * eta * (1.0 / rho[influence_mask] - 1.0 / rho_0)**2
    return U_rep


def calculate_road_potential(Y, y_min, y_max, beta_road):
    """Calculate the road-edge potential: U_road = beta * [(1/(y-y_min))^2 + (1/(y_max-y))^2]."""
    dist_to_min = np.where(Y - y_min <= 0, 0.1, Y - y_min)
    dist_to_max = np.where(y_max - Y <= 0, 0.1, y_max - Y)
    return beta_road * ((1.0 / dist_to_min)**2 + (1.0 / dist_to_max)**2)


def compute_forces(pos, goal_x, obstacles, xi, eta, rho_0, y_min, y_max, beta_road):
    """Calculate the negative gradient (forces) at a specific position."""
    # Attractive force: F_att = -∇U_att
    F_att = np.array([-xi * (pos[0] - goal_x), 0.0])

    F_rep = np.zeros(2)
    for obs in obstacles:
        diff = pos - obs
        rho = np.linalg.norm(diff)

        # Repulsive force: F_rep = -∇U_rep
        if 0 < rho <= rho_0:
            force_magnitude = eta * (1.0 / rho - 1.0 / rho_0) * (1.0 / rho**3)
            F_rep += force_magnitude * diff

    # Road edge force: F_road = -dU_road/dy, pushes the vehicle back toward the road center
    dist_to_min = max(pos[1] - y_min, 0.1)
    dist_to_max = max(y_max - pos[1], 0.1)
    F_road_y = beta_road * (2.0 / dist_to_min**3 - 2.0 / dist_to_max**3)
    F_road = np.array([0.0, F_road_y])

    return F_att + F_rep + F_road


def simulate_trajectory():
    """Simulate the path using gradient descent."""
    path = [START_POS.copy()]
    current_pos = START_POS.copy()
    
    for _ in range(MAX_ITERATIONS):
        # Stop if the X coordinate is close enough to the GOAL_X line
        if abs(current_pos[0] - GOAL_X) < GOAL_TOLERANCE:
            break
            
        F_tot = compute_forces(current_pos, GOAL_X, OBSTACLES, XI, ETA, RHO_0,
                                Y_MIN, Y_MAX, BETA_ROAD)
        
        # Normalize the force to prevent massive jumps near obstacles
        force_magnitude = np.linalg.norm(F_tot)
        if force_magnitude > 0:
            direction = F_tot / force_magnitude
            current_pos = current_pos + direction * STEP_SIZE
            
        path.append(current_pos.copy())
        
    return np.array(path)


def plot_projection_only():
    """Render just the flat, top-down XY-plane projection (road, obstacles, trajectory)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    road_surface = mpatches.Rectangle((X_START, Y_MIN), X_END - X_START, Y_MAX - Y_MIN,
                                       facecolor='dimgray', edgecolor='none', alpha=0.5, zorder=1)
    ax.add_patch(road_surface)

    ax.axvline(GOAL_X, color='lime', linewidth=4, label='Goal Line', zorder=3)
    ax.axhline(Y_MIN, color='saddlebrown', linewidth=3, label='Road Boundary', zorder=3)
    ax.axhline(Y_MAX, color='saddlebrown', linewidth=3, zorder=3)

    for obs in OBSTACLES:
        ax.scatter(obs[0], obs[1], color='darkred', s=20, zorder=4)
        disk = mpatches.Circle((obs[0], obs[1]), RHO_0_DISPLAY, facecolor='darkred',
                                edgecolor='darkred', alpha=0.25, zorder=3)
        ax.add_patch(disk)

    if SHOW_TRAJECTORY:
        path = simulate_trajectory()
        ax.plot(path[:, 0], path[:, 1], color='black', linestyle='--', linewidth=2,
                label='Trajectory', zorder=5)
        ax.scatter(START_POS[0], START_POS[1], color='cyan', s=80, edgecolor='k',
                   label='Start', zorder=6)

    ax.set_title("Trajectory Projection (XY Plane)", fontsize=16, pad=15)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_MIN - 20, Y_MAX + 20)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', facecolor=LEGEND_BACKGROUND_COLOR)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    if PROJECTION_ONLY:
        plot_projection_only()
        return

    # 1. Create the grid setup for the road
    x = np.linspace(X_START, X_END, 300)
    y = np.linspace(Y_MIN, Y_MAX, 150)
    X, Y = np.meshgrid(x, y)

    # 2. Compute potentials for the surface
    U_att = calculate_attractive_potential(X, GOAL_X, XI)
    U_rep = calculate_repulsive_potential(X, Y, OBSTACLES, ETA, RHO_0)
    U_road = calculate_road_potential(Y, Y_MIN-40, Y_MAX+40, BETA_ROAD)

    # Clip the repulsive spikes (obstacles + road edges) to cap the infinite spikes
    U_rep = np.clip(U_rep, 0, MAX_POTENTIAL)
    U_road = np.clip(U_road, 0, MAX_POTENTIAL)

    # Total potential is the unclipped attractive bowl + clipped repulsive spikes
    U_tot = U_att + U_rep + U_road

    # 3. Render the 3D Plot
    fig = plt.figure(figsize=(12, 8))
    fig.set_facecolor(BACKGROUND_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.xaxis.set_pane_color(BACKGROUND_COLOR)
    ax.yaxis.set_pane_color(BACKGROUND_COLOR)
    ax.zaxis.set_pane_color(BACKGROUND_COLOR)

    # Plot the surface
    surf = ax.plot_surface(X, Y, U_tot, cmap='viridis',
                           edgecolor='none', alpha=0.8, antialiased=True)

    # Shade the road region on the floor (z=0)
    road_surface = mpatches.Rectangle((X_START, Y_MIN), X_END - X_START, Y_MAX - Y_MIN,
                                       facecolor='dimgray', edgecolor='none', alpha=0.5, zorder=2)
    ax.add_patch(road_surface)
    art3d.pathpatch_2d_to_3d(road_surface, z=0, zdir='z')

    # Plot Goal Line, Road Boundary and Obstacles (projected onto the floor, z=0)
    ax.plot([GOAL_X, GOAL_X], [Y_MIN, Y_MAX], [0, 0], color='lime', linewidth=4, label='Goal Line', zorder=5)
    ax.plot([X_START, X_END], [Y_MIN, Y_MIN], [0, 0], color='saddlebrown', linewidth=3, label='Road Boundary', zorder=4)
    ax.plot([X_START, X_END], [Y_MAX, Y_MAX], [0, 0], color='saddlebrown', linewidth=3, zorder=4)
    for obs in OBSTACLES:
        ax.scatter(obs[0], obs[1], 0, color='darkred', s=20, zorder=5)
        disk = mpatches.Circle((obs[0], obs[1]), RHO_0_DISPLAY, facecolor='darkred', edgecolor='darkred', alpha=0.25, zorder=4)
        ax.add_patch(disk)
        art3d.pathpatch_2d_to_3d(disk, z=0, zdir='z')

    # 4. Trajectory Simulation & Plotting
    if SHOW_TRAJECTORY:
        path = simulate_trajectory()

        # Calculate Z values for the path so it lies accurately on the modified surface
        path_z_att = calculate_attractive_potential(path[:, 0], GOAL_X, XI)
        path_z_rep = calculate_repulsive_potential(path[:, 0], path[:, 1], OBSTACLES, ETA, RHO_0)
        path_z_road = calculate_road_potential(path[:, 1], Y_MIN, Y_MAX, BETA_ROAD)

        # Apply the exact same clipping logic to the path's Z coordinate
        path_z_rep = np.clip(path_z_rep, 0, MAX_POTENTIAL)
        path_z_road = np.clip(path_z_road, 0, MAX_POTENTIAL)
        path_z = path_z_att + path_z_rep + path_z_road

        if SHOW_TRAJECTORY_LINE:
            # Add a slight Z-offset (+100) so the line doesn't clip into the surface visually
            ax.plot(path[:, 0], path[:, 1], path_z + 100, color='red',
                    linewidth=3, label='Descent Direction', zorder=10)
            ax.scatter(START_POS[0], START_POS[1], path_z[0] + 100, color='cyan', s=80, label='Start', zorder=10)

        if SHOW_TRAJECTORY_PROJECTION:
            # Project the trajectory onto the XY (floor) plane of the same 3D view
            ax.plot(path[:, 0], path[:, 1], np.zeros_like(path_z), color='black', linestyle='--',
                    linewidth=2, label='Trajectory', zorder=9)
            ax.scatter(START_POS[0], START_POS[1], 0, color='cyan', s=80,
                       edgecolor='k', label='Start' if not SHOW_TRAJECTORY_LINE else None, zorder=9)

        ax.legend(loc='upper right', facecolor=LEGEND_BACKGROUND_COLOR)

    # Formatting
    ax.set_title("Artificial Potential Field (Unclipped Attraction)", fontsize=16, pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Potential Energy (U)', fontsize=12)
    ax.legend(loc='upper right', facecolor=LEGEND_BACKGROUND_COLOR)
    ax.set_box_aspect(AXIS_RATIO)
    ax.view_init(elev=50, azim=-115)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Potential Magnitude')
    plt.show()

if __name__ == '__main__':
    main()