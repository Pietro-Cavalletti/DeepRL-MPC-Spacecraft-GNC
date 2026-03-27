"""
Author: Pietro Cavalletti (2025)
Project: Model Predictive Control (MPC) for Autonomous Spacecraft Docking
Description: This script implements an MPC-based GNC (Guidance, Navigation, and Control) 
             system for a chaser spacecraft performing proximity operations and docking 
             with a target in an elliptical orbit. It accounts for orbital perturbations 
             (J2, Atmospheric Drag) and ensures safety-critical constraint satisfaction.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom modules
from Dynamics import RHS
from MPC import mpc_control
from Utils import (kep2car, rk8_step, normalize, random_chaser_state, 
                   control_pert, sun_pos, eci2lvlh)
from Plots import (plot_traj_cone, plot_errors_and_control, 
                   plot_position_velocity_components, plot_weight_log, 
                   plot_mpc_indices, plot_indices_vs_R, 
                   plot_dt_over_time_segment, plot_merit_indices_histograms)


def check_approach_cone(error_log, dock_err, theta, current_idx):
    """
    Checks whether the trajectory violates the approach cone or if there is overshoot.
    Returns: violates_cone (bool), overshoot (bool), out_of_cone (array), overshoots (array), cone_idx (int)
    """
    last_idx = current_idx + 1
    dists = np.linalg.norm(error_log[:3, :last_idx], axis=0)
    
    # Entrance in the cone
    cone_idx = np.argmax(dists < 0.100)
    cone2_idx = np.argmax(dists < 0.005)
    
    # Fallback: If it never enters, the index is the end
    if cone_idx == 0 and dists[0] >= 0.100: cone_idx = last_idx - 1
    if cone2_idx == 0 and dists[0] >= 0.005: cone2_idx = last_idx - 1

    # Range 1: 100m -> 5m (standard cone)
    y1 = dock_err[1, cone_idx:cone2_idx]
    r2_1 = dock_err[0, cone_idx:cone2_idx]**2 + dock_err[2, cone_idx:cone2_idx]**2
    out1 = r2_1 > (y1 * np.tan(theta))**2
    
    # Range 2: < 5m (offset cone for final approach)
    y2 = dock_err[1, cone2_idx:last_idx]
    r2_2 = dock_err[0, cone2_idx:last_idx]**2 + dock_err[2, cone2_idx:last_idx]**2
    out2 = r2_2 > ((y2 + 5.67e-4) * np.tan(theta))**2
    
    out_of_cone = np.concatenate([out1, out2]) if len(out1) or len(out2) else np.array([False])
    overshoots = np.concatenate([y1, y2]) < 0 if len(y1) or len(y2) else np.array([False])
    
    violates_cone = np.any(out_of_cone)
    overshoot = np.any(overshoots)
    
    return violates_cone, overshoot, out_of_cone, overshoots, cone_idx


def main():
    print("🎬 The script is running")
    t_start = time.perf_counter()
    
    # ---------------------------------------------------------
    # 1. PARAMETER CONFIGURATION
    #    n_orbit_points and n_simulations may be changed
    # ---------------------------------------------------------
    config = {
        "n_orbit_points": 2,
        "n_simulations": 2,
        "mu": 398600.4418,
        "AU": 1.495978707e8,
        "J2": 1,
        "Drag": 1,
        "u_max": 2e-5,
        "theta": np.radians(10),       # Approach cone angle
        "tol_vy": 1e-4,                # Docking tolerances
        "tol_r": 1e-4,
        "tol_vr": 4e-5,
        "tol_y": 2e-4,
        "N_min": 8, "N_max": 40,       # MPC parameter range
        "dt_min": 0.4, "dt_max": 0.7,
        "R_min": 0.5, "R_max": 1e7,
        "max_steps": 3000
    }

    success_counter = 0
    runtime_counter = 0
    tolerance_counter = 0
    thrust_exceed_counter = 0
    total_stepcounter = 0

    # ---------------------------------------------------------
    # 2. MATRIX AND VECTOR PREALLOCATION
    # ---------------------------------------------------------
    n_pts = config["n_orbit_points"]
    n_sims = config["n_simulations"]
    
    energy_index_mat1 = np.zeros((n_pts, n_sims))
    time_index_mat2 = np.zeros((n_pts, n_sims))
    time_index_mat = np.zeros((n_pts, n_sims))
    constraint_index_mat = np.zeros((n_pts, n_sims))
    
    N_vect = np.linspace(config["N_min"], config["N_max"], n_sims)
    dt_vect = np.linspace(config["dt_min"], config["dt_max"], n_sims)
    R_vect = np.linspace(config["R_min"], config["R_max"], n_sims)
    r_sun = sun_pos(np.random.uniform(0, 365.25))
    r_vect = np.linspace(100, 300, n_sims)

    # ---------------------------------------------------------
    # 3. MAIN SIMULATION LOOP
    # ---------------------------------------------------------
    for j in range(0, n_pts, 1):
        
        # Target orbital parameters and state
        a, e = 24000, 0.73
        i, Omega, omega, the = map(np.radians, [0, 0, 0, 360 * j / n_pts])
        state0_target = kep2car(a, e, i, Omega, omega, the, config["mu"])
        state0_chaser = random_chaser_state(state0_target, 9, 100, 300)
        
        for q in range(0, n_sims, 1):
            initial_dist = np.linalg.norm(state0_chaser - state0_target) * 1000
            print(f"\nStarting simulation number {q+1} in point {j+1}")
            print(f"Initial distance = {initial_dist:.2f} m")

            step_vect = np.arange(0, config["max_steps"], 1)
            n_steps = len(step_vect)
            docking_time = 0

            # MPC Parameters
            Q = np.eye(6)
            R = np.eye(3) * 10000 
            dt = 1.5
            Th = 15

            # Preallocate local simulation variables
            traj_target = np.zeros((6, n_steps))
            traj_chaser = np.zeros((6, n_steps))
            traj_target[:, 0] = state0_target
            traj_chaser[:, 0] = state0_chaser
            
            error_log = np.zeros((6, n_steps))
            dock_err = np.zeros((6, n_steps))
            control_log = np.zeros(n_steps)
            ratio_log = np.zeros(n_steps)
            weights_log = np.zeros((3, n_steps))
            dt_log = np.zeros(n_steps)
            time_log = np.zeros(n_steps)
            
            control = np.array([1, 1, 1])
            err_r2 = 1

            # --- DYNAMICS LOOP ---
            for i in range(0, n_steps - 1, 1):
                
                # Simulate Navigation noise
                pos_err = np.linalg.norm(traj_target[0:3, i] - traj_chaser[0:3, i])
                vel_err = np.linalg.norm(traj_target[3:6, i] - traj_chaser[3:6, i])
                pos_std = min(1, pos_err / 40)
                vel_std = min(1e-2, vel_err / 30)
                noise = np.hstack([pos_std * np.random.randn(3), vel_std * np.random.randn(3)])

                if i == 0:
                    Rs = R
                    
                dist_ratio = np.clip(pos_err / np.linalg.norm(state0_chaser - state0_target), 0.0, 1.0)
                v_max = config["u_max"] * (1 + 200 * dist_ratio**1.5)
                pos_weight = 0.5 * (1 / dist_ratio)**2
                vel_weight = 1e5 

                Qs = Q.copy() 
                Qs[0:3, 0:3] = np.eye(3) * pos_weight 
                Qs[3:6, 3:6] = np.eye(3) * vel_weight 

                weights_log[0, i] = pos_weight
                weights_log[1, i] = vel_weight
                weights_log[2, i] = 10000
                dt_log[i] = dt

                # Calculate control action with MPC
                h = np.linalg.norm(np.cross(traj_target[0:3, i], traj_target[3:6, i]))
                r = np.linalg.norm(traj_target[0:3, i])
                
                t_mom = time.perf_counter()
                control = mpc_control(traj_chaser[:, i], traj_target[:, i] + noise, Th, dt, Qs, Rs, 
                                      config["mu"], h, r, config["u_max"], control, v_max)
                time_i = time.perf_counter() - t_mom
                
                control = control_pert(control)
                if np.linalg.norm(control) > config["u_max"] * np.sqrt(3):
                    control = config["u_max"] * np.sqrt(3) * normalize(control)
                    thrust_exceed_counter += 1

                # Propagate trajectories
                traj_target[:, i+1] = rk8_step(traj_target[:, i], dt, RHS, config["mu"], 
                                               r_sun, config["J2"], config["Drag"], None)
                traj_chaser[:, i+1] = rk8_step(traj_chaser[:, i], dt, RHS, config["mu"], 
                                               r_sun, config["J2"], config["Drag"], control)
                docking_time += dt
                time_log[i] = docking_time

                # Debug and plot data
                Rot = eci2lvlh(traj_target[:, i])
                error_log[:, i] = traj_chaser[:, i] - traj_target[:, i]
                error_log[:, i] = np.hstack((Rot @ error_log[:3, i], Rot @ error_log[3:, i]))
                
                control_log[i] = np.linalg.norm(control)
                ratio_log[i] = vel_weight / pos_weight
                
                energy1 = control_log[i] * dt 
                energy_index_mat1[j, q] += energy1
                time_index_mat2[j, q] += (time_i / dt) * 100
                time_index_mat[j, q] += time_i

                # Compute error in docking frame
                docking_axis = np.array([0, -1, 0])
                x_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                
                err_y = error_log[0:3, i] @ docking_axis
                err_vy = error_log[3:6, i] @ docking_axis
                err_r2 = (error_log[0:3, i] @ x_axis)**2 + (error_log[0:3, i] @ z_axis)**2
                err_vr2 = (error_log[3:6, i] @ x_axis)**2 + (error_log[3:6, i] @ z_axis)**2
                
                dock_err[1, i] = err_y
                dock_err[4, i] = err_vy 
                dock_err[0, i] = error_log[0:3, i] @ x_axis
                dock_err[2, i] = error_log[0:3, i] @ z_axis
                dock_err[3, i] = error_log[3:6, i] @ x_axis
                dock_err[5, i] = error_log[3:6, i] @ z_axis
                
                constraint_index_mat[j, q] += (err_r2) / ((err_y + 5.67e-4) * np.tan(config["theta"]))**2 * dt

                # Console Logging
                if (i + 1) % 50 == 0 or i == n_steps - 1:
                    print(f"Step {i+1}/{n_steps} ({(i+1)/n_steps*100:.1f}%), d={err_y*1000:.2f}")

                # --- EXIT CONDITIONS ---
                
                # Condition 1: Docking tolerances satisfied
                cond_docking = (np.abs(err_vy) < config["tol_vy"] and 
                                err_r2 < config["tol_r"]**2 and 
                                err_vr2 < config["tol_vr"]**2 and 
                                err_y < config["tol_y"])
                
                if i > 0 and cond_docking:
                    print("✅ Docking tolerances are satisfied")
                    violates_cone, overshoot, out_of_cone, overshoots, cone_idx = check_approach_cone(
                        error_log, dock_err, config["theta"], i
                    )
                    
                    if violates_cone:
                        print(f"⚠️ {np.sum(out_of_cone)} points might violate the approach cone.")
                    if overshoot:
                        print("⚠️ Overshoot: trajectory goes beyond docking plane!")
                    if not (violates_cone or overshoot):
                        print("✅ Trajectory valid: inside cone and no overshoot.")
                        success_counter += 1
                        
                    print(f"⏱️  Docking time: {docking_time / 60:.2f} [m]")
                    last_idx = i + 1
                    break
                
                # Condition 2: Stuck (No movement)
                if i > 0 and np.linalg.norm(error_log[:, i] - error_log[:, i-1]) < 1e-6:
                    tolerance_counter += 1
                    violates_cone, overshoot, out_of_cone, overshoots, cone_idx = check_approach_cone(
                        error_log, dock_err, config["theta"], i
                    )
                    
                    if violates_cone:
                        print(f"⚠️ {np.sum(out_of_cone)} points might violate the approach cone.")
                    if overshoot:
                        print("⚠️ Overshoot: trajectory goes beyond docking plane!")
                    
                    last_idx = i + 1
                    break
                
                # Condition 3: Time out (End of simulation reached without docking)
                if i == n_steps - 2:
                    runtime_counter += 1
                    violates_cone, overshoot, out_of_cone, overshoots, cone_idx = check_approach_cone(
                        error_log, dock_err, config["theta"], i
                    )
                    
                    if violates_cone:
                        print(f"⚠️ {np.sum(out_of_cone)} points might violate the approach cone.")
                    
                    print("❌ Docking not achieved in given time")
                    last_idx = i + 1
            
            # Post-simulation metric updates
            time_index_mat[j, q] /= last_idx
            time_index_mat2[j, q] /= last_idx
            constraint_index_mat[j, q] /= docking_time
            total_stepcounter += last_idx

    # ---------------------------------------------------------
    # 4. PLOTS AND RESULTS
    # ---------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    total_runs = n_sims * n_pts

    print("\n" + "="*40)
    print(f"⏱️  Simulation time: {elapsed/60:.6f} [m]")
    print(f"Success:         {(success_counter / total_runs) * 100:.2f} %")
    print(f"Exceeded time:   {(runtime_counter / total_runs) * 100:.2f} %")
    print(f"Got stuck:       {(tolerance_counter / total_runs) * 100:.2f} %")
    print(f"Thrust exceeded: {(thrust_exceed_counter / total_stepcounter) * 100:.2f} %")
    print("="*40 + "\n")

    # Call external plot functions
    plot_traj_cone(error_log, config["theta"], cone_idx, last_idx, out_of_cone, overshoots)
    plot_errors_and_control(time_log, error_log, control_log, last_idx, config["u_max"])
    plot_position_velocity_components(time_log, dock_err, last_idx)
    plot_merit_indices_histograms(energy_index_mat1, time_index_mat2, time_index_mat, constraint_index_mat)
    plot_weight_log(time_log, weights_log, last_idx)

    plt.show()

if __name__ == "__main__":
    main()