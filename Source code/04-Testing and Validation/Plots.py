# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: A suite of plotting functions for satellite docking 
#              simulations, including merit index histograms, 3D trajectory 
#              analysis within conical constraints, and control effort logs.
# =============================================================================

from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from datetime import datetime

def plot_merit_indices_histograms(Energy_index_mat1,
                                   Time_index_mat2,
                                   Time_index_mat,
                                   Constraint_index_mat,
                                   output_dir=None,
                                   bins=15,
                                   save_stats=True):
    # === 1. Get current script directory ===
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # === 2. Define docking_results subfolder ===
    if output_dir is None:
        output_dir = os.path.join(current_dir, "docking_results")

    # === 3. Create directory if it doesn't exist ===
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === 4. Prepare data and charts ===
    indices = {
        "Energy_Index_1 [m/s]": Energy_index_mat1.flatten() * 1000,
        "Time_index 1 [%]": Time_index_mat2.flatten(),
        "Time_Index 2 [s]": Time_index_mat.flatten(),
        "Constraint_Index []": Constraint_index_mat.flatten()
    }

    colors = ['tab:green', 'tab:blue', 'tab:cyan', 'tab:red']
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
    fig.savefig(os.path.join(output_dir, f"merit_indices_histograms_{timestamp}.png"))

    if save_stats:
        with open(os.path.join(output_dir, f"merit_indices_stats_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write("".join(stats))


def plot_traj_cone(error_log, theta, cone_idx, last_idx, out_of_cone, overshoots):
    """
    Plots the approach trajectory within a conical constraint region, 
    highlighting any violations.
    """
    # Cone parameters
    h = 100  # Cone height in meters
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot relative trajectory (converted to meters)
    ax.plot(error_log[0, :last_idx]*1000,
            error_log[1, :last_idx]*1000,
            error_log[2, :last_idx]*1000, color='b', label='Trajectory')

    # Highlight out-of-cone violations (in red)
    bad_pts = error_log[:3, cone_idx:last_idx][:, out_of_cone] * 1000
    ax.scatter(bad_pts[0], bad_pts[1], bad_pts[2], color='red', s=20, label='Out of cone')

    # Highlight overshoots (in black)
    bad_pt = error_log[:3, cone_idx:last_idx][:, overshoots] * 1000
    ax.scatter(bad_pt[0], bad_pt[1], bad_pt[2], color='black', s=20, label='Overshoot')

    # Draw the cone surface
    ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='orange')

    ax.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
    ax.legend()
    plt.tight_layout()


def plot_errors_and_control(time_log, error_log, control_log, last_idx, u_max):
    """
    Plots position error, velocity error, and control action over time.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Position Error Plot ---
    pos_error = np.linalg.norm(error_log[0:3, :last_idx] * 1000, axis=0)  # in meters
    axs[0].plot(time_log[:last_idx], pos_error, linestyle='-', color='blue')
    axs[0].set_ylabel('Error [m]')
    axs[0].set_title('Position Error over Time')
    axs[0].grid(True, which='both')

    # --- Velocity Error Plot ---
    vel_error = np.linalg.norm(error_log[3:6, :last_idx] * 1000, axis=0)  # in m/s
    axs[1].plot(time_log[:last_idx], vel_error, linestyle='-', color='blue')
    axs[1].set_ylabel('Error [m/s]')
    axs[1].set_title('Velocity Error over Time')
    axs[1].grid(True, which='both')

    # --- Control Plot ---
    axs[2].plot(time_log[:last_idx], control_log[:last_idx] * 1000, linestyle='-', color='red', label='|u|')
    axs[2].plot(time_log[:last_idx], np.sqrt(3) * u_max * 1000 * np.ones(last_idx), linestyle='--', color='orange', label='√3·u_max')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control [m/s²]')
    axs[2].set_title('Control Action over Time')
    axs[2].grid(True, which='both')
    axs[2].legend()

    plt.tight_layout()


def plot_position_velocity_components(time_log, dock_err, last_idx):
    """
    Plots the components of position and velocity errors over time.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.suptitle('Position and Velocity Components Over Time')

    # --- Velocity Components ---
    axs[0, 0].plot(time_log[:last_idx], dock_err[3, :last_idx] * 1000, color='r', label='Vx')
    axs[0, 0].set_ylabel('Vx [m/s]')
    axs[0, 0].set_title('Velocity X')
    axs[0, 0].grid(True)

    axs[1, 0].plot(time_log[:last_idx], dock_err[4, :last_idx] * 1000, color='g', label='Vy')
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

    axs[1, 1].plot(time_log[:last_idx], dock_err[1, :last_idx] * 1000, color='g', label='Y')
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
    """
    plt.figure(figsize=(18, 5))

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
            theta = 360 * row_idx / n_rows
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
    """
    plt.figure(figsize=(18, 5))

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
            theta = 360 * row_idx / n_rows
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
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time_log[:last_idx], dt_log[:last_idx], linestyle='-', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Timestep dt [s]')
    plt.title('Timestep over Time')
    plt.grid(True)
    plt.tight_layout()


def plot_merit_indices_barcharts(Energy_index_mat1,
                                 Time_index_mat2,
                                 Time_index_mat,
                                 Constraint_index_mat,
                                 output_dir=None,
                                 save_stats=True):
    """
    Generates bar charts for merit indices, grouping X-axis labels by orbit,
    and saves a text file with statistics.
    """
    # === 1. Prepare output environment ===
    if output_dir is None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "docking_results")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === 2. Extract dimensions and define plot parameters ===
    n_orbits, n_points_per_orbit, _ = Energy_index_mat1.shape
    total_points = n_orbits * n_points_per_orbit

    orbit_names = ["GTO1", "GTO2", "GTO3", "Molniya1", "Molniya2", "Tundra1", "Tundra2"]
    if len(orbit_names) != n_orbits:
        print(f"WARNING: Number of orbit names ({len(orbit_names)}) does not match number of orbits ({n_orbits}). Using generic labels.")
        orbit_names = [f"Orbit {i+1}" for i in range(n_orbits)]

    # Color palettes
    energy_colors = plt.get_cmap('Greens')(np.linspace(0.3, 0.9, n_points_per_orbit))
    relative_time_colors = plt.get_cmap('Blues')(np.linspace(0.3, 0.9, n_points_per_orbit))
    absolute_time_colors = plt.get_cmap('Purples')(np.linspace(0.3, 0.9, n_points_per_orbit))
    constraint_colors = plt.get_cmap('Reds')(np.linspace(0.3, 0.9, n_points_per_orbit))
    
    color_palettes = {
        "Energy Consumption (Total Impulse) [m/s]": energy_colors,
        "Relative Computation Time [%]": relative_time_colors,
        "Mean Computation Time per Step [s]": absolute_time_colors,
        "Cone Constraint Violation (Integral)": constraint_colors
    }
    
    indices_data = {
        "Energy Consumption (Total Impulse) [m/s]": Energy_index_mat1 * 1000,
        "Relative Computation Time [%]": Time_index_mat2,
        "Mean Computation Time per Step [s]": Time_index_mat,
        "Cone Constraint Violation (Integral)": Constraint_index_mat
    }

    all_stats_text = []

    # === 3. Generate chart for each merit index ===
    for name, data_matrix in indices_data.items():
        
        current_colors = color_palettes[name]
        means = np.mean(data_matrix, axis=2)
        stds = np.std(data_matrix, axis=2)
        bar_values = means.flatten()
        error_bar_values = stds.flatten()
        
        bar_colors = [current_colors[p % n_points_per_orbit] for p in range(total_points)]
        
        fig, ax = plt.subplots(figsize=(18, 9))
        
        x_positions = np.arange(total_points)
        ax.bar(x_positions, bar_values, yerr=error_bar_values, 
               color=bar_colors, capsize=4, alpha=0.85, edgecolor='black', width=0.8)
        
        ax.set_ylabel("Mean Value")
        ax.set_title(f"Merit Index: {name}", fontsize=16, pad=20)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        tick_positions = [i * n_points_per_orbit + (n_points_per_orbit - 1) / 2.0 for i in range(n_orbits)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(orbit_names, rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis='x', which='minor', bottom=False)

        for i in range(n_orbits - 1):
            separator_pos = (i + 1) * n_points_per_orbit - 0.5
            ax.axvline(x=separator_pos, color='grey', linestyle='--', linewidth=1.0, alpha=0.8)

        legend_labels = [f"Theta = {i*90}°" for i in range(n_points_per_orbit)]
        legend_patches = [Patch(color=current_colors[i], label=legend_labels[i]) for i in range(n_points_per_orbit)]
        ax.legend(handles=legend_patches, title="Orbital Point", bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.tight_layout(rect=[0, 0, 0.92, 1])
        
        safe_name = name.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "").replace("%", "")
        fig.savefig(os.path.join(output_dir, f"barchart_{safe_name}_{timestamp}.png"))
        plt.close(fig)

    # === 4. Save detailed statistics text file ===
    if save_stats:
        for o in range(n_orbits):
            orbit_name_for_stats = orbit_names[o]
            all_stats_text.append(f"================== ORBIT {o+1} ({orbit_name_for_stats}) ==================\n")
            for p in range(n_points_per_orbit):
                theta_deg = p * 90
                all_stats_text.append(f"--- Point {p+1} (Theta = {theta_deg}°) ---\n")
                
                for name, data_matrix in indices_data.items():
                    data_slice = data_matrix[o, p, :]
                    mean_val = np.mean(data_slice)
                    std_val = np.std(data_slice)
                    clean_name = name.split('[')[0].strip()
                    all_stats_text.append(f"  - {clean_name}: Mean = {mean_val:.4e}, Std Dev = {std_val:.4e}\n")
                all_stats_text.append("\n")

        with open(os.path.join(output_dir, f"merit_indices_detailed_stats_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write("".join(all_stats_text))
            
    print(f"Charts and statistics saved in: {output_dir}")


def save_merit_indices_raw(Energy_index_mat1,
                            Time_index_mat2,
                            Time_index_mat,
                            Constraint_index_mat,
                            output_dir=None):
    """
    Saves raw data matrices into a single compressed .npz file.
    """
    if output_dir is None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "docking_results")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_data_for_plotting_{timestamp}.npz"
    output_path = os.path.join(output_dir, filename)

    try:
        np.savez_compressed(
            output_path,
            energy=Energy_index_mat1,
            time_relative=Time_index_mat2,
            time_absolute=Time_index_mat,
            constraint=Constraint_index_mat
        )
        print(f"Raw data successfully saved in: {output_path}")
    except Exception as e:
        print(f"Error while saving raw data: {e}")