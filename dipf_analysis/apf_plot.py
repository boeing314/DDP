import numpy as np
import matplotlib.pyplot as plt

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
[500,-90],[500,5]
])

# Khatib's APF Parameters
XI = 0.01          # Attractive gain (pull towards goal line)
ETA = 3e6          # Repulsive gain (push away from obstacles)
RHO_0 = 150.0      # Limit distance of obstacle influence (radius)

# Visualization parameters
MAX_POTENTIAL = 3000       # Applied ONLY to the repulsive spikes now
AXIS_RATIO = (10, 4, 5) 

# Trajectory simulation parameters
SHOW_TRAJECTORY = True     
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


def compute_forces(pos, goal_x, obstacles, xi, eta, rho_0):
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
            
    return F_att + F_rep


def simulate_trajectory():
    """Simulate the path using gradient descent."""
    path = [START_POS.copy()]
    current_pos = START_POS.copy()
    
    for _ in range(MAX_ITERATIONS):
        # Stop if the X coordinate is close enough to the GOAL_X line
        if abs(current_pos[0] - GOAL_X) < GOAL_TOLERANCE:
            break
            
        F_tot = compute_forces(current_pos, GOAL_X, OBSTACLES, XI, ETA, RHO_0)
        
        # Normalize the force to prevent massive jumps near obstacles
        force_magnitude = np.linalg.norm(F_tot)
        if force_magnitude > 0:
            direction = F_tot / force_magnitude
            current_pos = current_pos + direction * STEP_SIZE
            
        path.append(current_pos.copy())
        
    return np.array(path)


def main():
    # 1. Create the grid setup for the road
    x = np.linspace(X_START, X_END, 300)
    y = np.linspace(Y_MIN, Y_MAX, 150)
    X, Y = np.meshgrid(x, y)

    # 2. Compute potentials for the surface
    U_att = calculate_attractive_potential(X, GOAL_X, XI)
    U_rep = calculate_repulsive_potential(X, Y, OBSTACLES, ETA, RHO_0)
    
    # Clip ONLY the repulsive potential to cap the infinite spikes
    U_rep = np.clip(U_rep, 0, MAX_POTENTIAL)
    
    # Total potential is the unclipped attractive bowl + clipped obstacle spikes
    U_tot = U_att + U_rep

    # 3. Render the 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, U_tot, cmap='viridis', 
                           edgecolor='none', alpha=0.8, antialiased=True)
    
    # Plot Goal Line and Obstacles
    ax.plot([GOAL_X, GOAL_X], [Y_MIN, Y_MAX], [0, 0], color='lime', linewidth=4, label='Goal Line', zorder=5)
    for obs in OBSTACLES:
        ax.scatter(obs[0], obs[1], 0, color='darkred', s=50, zorder=5)

    # 4. Trajectory Simulation & Plotting
    if SHOW_TRAJECTORY:
        path = simulate_trajectory()
        
        # Calculate Z values for the path so it lies accurately on the modified surface
        path_z_att = calculate_attractive_potential(path[:, 0], GOAL_X, XI)
        path_z_rep = calculate_repulsive_potential(path[:, 0], path[:, 1], OBSTACLES, ETA, RHO_0)
        
        # Apply the exact same clipping logic to the path's Z coordinate
        path_z_rep = np.clip(path_z_rep, 0, MAX_POTENTIAL)
        path_z = path_z_att + path_z_rep
        
        # Add a slight Z-offset (+100) so the line doesn't clip into the surface visually
        ax.plot(path[:, 0], path[:, 1], path_z + 100, color='red', 
                linewidth=3, label='Trajectory', zorder=10)
        ax.scatter(START_POS[0], START_POS[1], path_z[0] + 100, color='cyan', s=80, label='Start', zorder=10)
        
        ax.legend(loc='upper right')
    
    # Formatting
    ax.set_title("Artificial Potential Field (Unclipped Attraction)", fontsize=16, pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Potential Energy (U)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_box_aspect(AXIS_RATIO) 
    ax.view_init(elev=50, azim=-115)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Potential Magnitude')
    plt.show()

if __name__ == '__main__':
    main()