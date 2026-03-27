# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Visualization module for docking performance metrics. 
#              Generates histograms and statistical summaries for mission indices.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R


def plot_merit_indices_histograms(Energy_index_mat1,
                                   Time_index_mat2,
                                   Time_index_mat,
                                   Constraint_index_mat,
                                   output_dir=None,
                                   bins=15,
                                   save_stats=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(current_dir, "risultati_docking")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    indices = {
         "ΔV [m/s]": Energy_index_mat1.flatten() * 1000,
        "Constraint_Index []": Constraint_index_mat.flatten(),
        "Mean relative CPU time  [%]": Time_index_mat2.flatten(),
        "Mean CPU time [s]": Time_index_mat.flatten()
        
    }

    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:cyan']
    stats = []
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    axs = axs.flatten()

    for i, (name, data) in enumerate(indices.items()):
        axs[i].hist(data, bins=bins, color=colors[i], edgecolor='black', alpha=0.75)
        axs[i].set(title=name, ylabel="Frequency")
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].tick_params(labelsize=10)

        mean = np.mean(data)
        std = np.std(data)
        q25 = np.percentile(data, 25)
        q50 = np.percentile(data, 50)
        q75 = np.percentile(data, 75)

        stats_text = (
            f"Mean: {mean:.2e}\n"
            f"Q25: {q25:.2e}\n"
            f"Q50: {q50:.2e}\n"
            f"Q75: {q75:.2e}"
        )
        axs[i].text(0.98, 0.95, stats_text,
                    transform=axs[i].transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8))

        if save_stats:
            stats.append(
                f"{name}:\n"
                f"  Avg: {mean:.4e}, Std: {std:.4e}\n"
                f"  Q25: {q25:.4e}, Q50: {q50:.4e}, Q75: {q75:.4e}\n"
            )

    fig.tight_layout()
    #fig.savefig(os.path.join(output_dir, f"merit_indices_histograms_{timestamp}.png"))

    if save_stats:
        with open(os.path.join(output_dir, f"merit_indices_stats_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write("".join(stats))


# Trajectory plot
def plot_traj_cone(error_log, theta, cone_idx, last_idx, out_of_cone, overshoots):
    """
    Plots the approach trajectory within a conical constraint region, 
    highlighting any violations.

    Parameters:
        error_log: np.ndarray [3 x N] - relative position errors in ECI or LVLH [km]
        theta: float - semi-aperture angle of the cone [radians]
        cone_idx: int - starting index for constraint checking
        last_idx: int - ending index (included in the plot)
        out_of_cone: boolean array - flags points that violate cone constraint
        overshoots: boolean array - flags points that pass the docking plane
    """

    # Cone parameters
    h = 100  # Cone height in km (100 m)
    n = 20   # Number of mesh points
    x = np.linspace(0, h, n)
    r = x * np.tan(theta)
    phi = np.linspace(0, 2 * np.pi, n)

    # Create cylindrical mesh aligned with +X axis
    X, P = np.meshgrid(x, phi)
    Y = r[np.newaxis, :] * np.cos(P)
    Z = r[np.newaxis, :] * np.sin(P)
    cone = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Rotate the cone: align with negative Y axis (X → Y, invert Y)
    rotation = R.from_euler('z', -90, degrees=True)
    cone_rotated = rotation.apply(cone)

    # Reshape the rotated cone back to meshgrid format
    Xc = cone_rotated[:, 0].reshape(n, n)
    Yc = cone_rotated[:, 1].reshape(n, n)
    Zc = cone_rotated[:, 2].reshape(n, n)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot relative trajectory (converted to meters)
    ax.plot(error_log[0, :last_idx]*1000,
            error_log[1, :last_idx]*1000,
            error_log[2, :last_idx]*1000,
            color='b', linewidth=2, label='Chaser trajectory')

    # Highlight start and end points
    ax.scatter(error_log[0, 0]*1000,
               error_log[1, 0]*1000,
               error_log[2, 0]*1000,
               color='green', s=25, label='Start')

    ax.scatter(error_log[0, last_idx-1]*1000,
               error_log[1, last_idx-1]*1000,
               error_log[2, last_idx-1]*1000,
               color='black', s=15, label='End')


    # Draw the cone surface
    ax.plot_surface(Xc, Yc, Zc, alpha=0.25, color='orange', linewidth=0.3)

    # Labels
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')

    # Equal scaling and aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-60)  # Consistent camera angle

    # Legend and layout
    ax.legend(
    loc='upper center',       # place legend at top
    bbox_to_anchor=(0.5, 0.85),  # center above the axes
    ncol=3,                   # number of columns in the legend
    fontsize=10
    )
    ax.set_box_aspect([1,2.5,1])




def plot_errors_and_control(time_log, error_log, control_log, last_idx, u_max):
    """
    Plots position error, velocity error, and control action over time.

    Parameters:
    - time_log: array of time values
    - error_log: 6×N array, with position (0:3) and velocity (3:6) errors
    - control_log: 1D array of control magnitude values
    - last_idx: integer index up to which the data should be plotted
    - u_max: scalar maximum control acceleration (in km/s²)
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Position Error Plot ---
    pos_error = np.linalg.norm(error_log[0:3, :last_idx] * 1000, axis=0)  # in meters
    axs[0].plot(time_log[:last_idx], pos_error, linestyle='-', color='blue')
    axs[0].set_ylabel('Error [m]')
    axs[0].set_title('Position Error over Time')
    #axs[0].set_yscale('log')
    axs[0].grid(True, which='both')

    # --- Velocity Error Plot ---
    vel_error = np.linalg.norm(error_log[3:6, :last_idx] * 1000, axis=0)  # in m/s
    axs[1].plot(time_log[:last_idx], vel_error, linestyle='-', color='blue')
    axs[1].set_ylabel('Error [m/s]')
    axs[1].set_title('Velocity Error over Time')
    #axs[1].set_yscale('log')
    axs[1].grid(True, which='both')

    # --- Control Plot ---
    axs[2].plot(time_log[:last_idx], control_log[:last_idx] * 1000, linestyle='-', color='red', label='|u|')
    axs[2].plot(time_log[:last_idx], np.sqrt(3) * u_max * 1000 * np.ones(last_idx), linestyle='--', color='orange', label='√3·u_max')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control [m/s²]')
    axs[2].set_title('Control Action over Time')
    #axs[2].set_yscale('log')
    axs[2].grid(True, which='both')
    axs[2].legend()

    plt.tight_layout()



def plot_position_velocity_components(time_log, dock_err, last_idx):
    """
    Plots the components of position and velocity errors over time.

    Parameters:
    - time_log: array of time values
    - dock_err: 6×N array, with position (0:3) and velocity (3:6) errors
    - last_idx: integer index up to which the data should be plotted
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.suptitle('Position and Velocity Components Over Time')

    # --- Velocity Components ---
    axs[0, 0].plot(time_log[:last_idx], dock_err[3, :last_idx] * 1000, color='r', label='Vx')
    axs[0, 0].set_ylabel('Vx [m/s]')
    axs[0, 0].set_title('Velocity X')
    axs[0, 0].grid(True)

    axs[1, 0].plot(time_log[:last_idx], -dock_err[4, :last_idx] * 1000, color='g', label='Vy')
    axs[1, 0].set_ylabel('Vy [m/s]')
    axs[1, 0].set_title('Velocity Y')
    axs[1, 0].grid(True)

    axs[2, 0].plot(time_log[:last_idx], dock_err[5, :last_idx] * 1000, color='b', label='Vz')
    axs[2, 0].set_ylabel('Vz [m/s]')
    axs[2, 0].set_title('Velocity Z')
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].grid(True)

    # --- Position Components ---
    axs[0, 1].plot(time_log[:last_idx], dock_err[0, :last_idx] * 1000, color='r', label='X')
    axs[0, 1].set_ylabel('X [m]')
    axs[0, 1].set_title('Position X')
    axs[0, 1].grid(True)

    axs[1, 1].plot(time_log[:last_idx], -dock_err[1, :last_idx] * 1000, color='g', label='Y')
    axs[1, 1].set_ylabel('Y [m]')
    axs[1, 1].set_title('Position Y')
    axs[1, 1].grid(True)

    axs[2, 1].plot(time_log[:last_idx], dock_err[2, :last_idx] * 1000, color='b', label='Z')
    axs[2, 1].set_ylabel('Z [m]')
    axs[2, 1].set_title('Position Z')
    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].grid(True)



def plot_weight_log(time_log, weights_log, last_idx, labels=('Qp weight', 'Qv weight', 'R weight')):
    """
    Plots the evolution of Q and R weights over time.

    Parameters:
    - time_log: array of time values
    - weights_log: 2 x T array, where each row is a weight over time
    - last_idx: int, index up to which to plot
    - labels: tuple of str, labels for the weights 
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_log[:last_idx], weights_log[0, :last_idx], label=labels[0], color='#1f77b4', linewidth=2)
    ax.plot(time_log[:last_idx], weights_log[1, :last_idx], label=labels[1], color='#FFD43B', linewidth=2)
    ax.plot(time_log[:last_idx], weights_log[2, :last_idx], label=labels[2], color='#d62728', linewidth=2)


    ax.set_title('Weight Evolution Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Weight Value [-]', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig, ax




def plot_mpc_indices(Vect, Energy_index_mat1, Time_index_mat, Constraint_index_mat):
    """
    Plots three MPCC performance indices (Energy, Time, Constraint) over prediction steps.

    Parameters:
    - Energy_index_mat1: 2D array with energy index values
    - Time_index_mat: 2D array with computation time values
    - Constraint_index_mat: 2D array with constraint satisfaction metrics
    """
    plt.figure(figsize=(18, 5))  # Wider figure for 3 subplots

    for i, (matrix, label, base_cmap) in enumerate([
        (Energy_index_mat1, 'Energy index', 'plasma'),
        (Time_index_mat, 'Computation time [s]', 'plasma'),
        (Constraint_index_mat, 'Constraint index', 'plasma')
    ]):
        plt.subplot(1, 3, i + 1)

        n_rows = matrix.shape[0]
        cmap = plt.get_cmap(base_cmap)

        for row_idx in range(n_rows):
            y_vals = matrix[row_idx]
            theta = 360 * row_idx / n_rows  # Assuming theta spans 0–360°
            color = cmap(row_idx / (n_rows - 1))
            plt.plot(Vect, y_vals, 'o-', color=color, label=f'Theta = {theta:.1f}°')

        plt.xlabel('Initial distance []')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()


def plot_indices_vs_R(R_vect, Energy_index_mat1, Time_index_mat, Constraint_index_mat):
    """
    Plots three MPCC performance indices (Energy, Time, Constraint) vs diagonal element of R.

    Parameters:
    - R_vect: array of diagonal elements of matrix R (cost weight on control effort)
    - Energy_index_mat1: 2D array with energy index values
    - Time_index_mat: 2D array with computation time values
    - Constraint_index_mat: 2D array with constraint satisfaction metrics
    """
    plt.figure(figsize=(18, 5))  # 3 subplots horizontally

    for i, (matrix, label, base_cmap) in enumerate([
        (Energy_index_mat1, 'Energy index', 'plasma'),
        (Time_index_mat, 'Computation time [s]', 'plasma'),
        (Constraint_index_mat, 'Constraint index', 'plasma')
    ]):
        plt.subplot(1, 3, i + 1)

        n_rows = matrix.shape[0]
        cmap = plt.get_cmap(base_cmap)

        for row_idx in range(n_rows):
            y_vals = matrix[row_idx]
            theta = 360 * row_idx / n_rows  # Assuming θ spans 0–360°
            color = cmap(row_idx / (n_rows - 1))
            plt.plot(R_vect, y_vals, 'o-', color=color, label=f'Theta = {theta:.1f}°')

        plt.xlabel('Diagonal element of R')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()



def plot_dt_over_time_segment(time_log, dt_log, last_idx):
    """
    Plots dt_log against time_log up to a specified index.

    Parameters:
    - time_log: array of time stamps
    - dt_log: array of control time steps (same length as time_log)
    - last_idx: index up to which to plot the data
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time_log[:last_idx], dt_log[:last_idx], linestyle='-', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Timestep dt [s]')
    plt.title('Timestep over Time')
    plt.grid(True)
    plt.tight_layout()