# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Main execution script for multi-orbit satellite docking 
#              simulations. Evaluates classical MPC control performance across  
#              orbital parameters (GTO, Molniya, Tundra) and orbital points.
# =============================================================================

import numpy as np
from Dynamics import RHS
from MPC_RL import mpc_control
from Utils import kep2car, rk8_step, normalize, random_chaser_state, control_pert, sun_pos, eci2lvlh
from Plots import save_merit_indices_raw, plot_mpc_indices, plot_merit_indices_barcharts, plot_dt_over_time_segment, plot_merit_indices_histograms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# --- Simulation Parameters ---
N_ORBITS = 7
N_POINTS_PER_ORBIT = 4
N_SIMULATIONS_PER_POINT = 100
SEED_FOR_REPRODUCIBILITY = 42 
np.random.seed(SEED_FOR_REPRODUCIBILITY)
t_start = time.perf_counter()

success_counter = 0; runtime_counter = 0; tolerance_counter = 0
thrust_exceed_counter = 0; total_stepcounter = 0
mu = 398600.4418; AU = 1.495978707e8; J2 = 1; Drag = 1
u_max = 2e-5
theta = np.radians(10) # Approach cone angle
# Docking tolerances
tol_vy = 1e-4; tol_r = 1e-4; tol_vr = 4e-5; tol_y = 2e-4
r_sun = sun_pos(np.random.uniform(0, 365.25))

# Orbital parameters for the 7 orbits 
orbital_params = [ {'a': 24414, 'e': 0.7267, 'i': 7.0, 'Omega': 0, 'omega': 0},
                    {'a': 24414, 'e': 0.7267, 'i': 28.5, 'Omega': 0, 'omega': 0},
                    {'a': 24364, 'e': 0.7303, 'i': 7.0, 'Omega': 0, 'omega': 0},
                    {'a': 27300, 'e': 0.73, 'i': 63.4, 'Omega': 0, 'omega': 0},
                    {'a': 25448, 'e': 0.73, 'i': 63.4, 'Omega': 0, 'omega': 0},
                    {'a': 42164, 'e': 0.3, 'i': 63.4, 'Omega': 0, 'omega': 0},
                    {'a': 42164, 'e': 0.4, 'i': 63.4, 'Omega': 0, 'omega': 0} ]

Energy_index_mat1 = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Time_index_mat = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Time_index_mat2 = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Constraint_index_mat = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))

def random_chaser_state(state_target, theta_max_deg=9, r_min=100, r_max=300):
    """
    Generates a random initial state for the chaser relative to the target.
    Uses np.random, thus depends on the global seed.
    """
    # Assuming eci2lvlh is defined elsewhere
    Rot = eci2lvlh(state_target)
    
    # Initial distance uniformly distributed
    d0 = np.random.uniform(r_min, r_max) / 1000  # in km
    
    # Position inside the approach cone
    theta_rand = np.random.uniform(0, np.radians(theta_max_deg))
    phi_rand = np.random.uniform(0, 2 * np.pi)
    r0 = d0 * np.array([np.sin(theta_rand) * np.cos(phi_rand), -np.cos(theta_rand), np.sin(theta_rand) * np.sin(phi_rand)])
    
    # Relative velocity with Gaussian distribution
    v0 = np.random.normal(0, 0.1, 3) / 1000  # in km/s
    
    x_rel_lvlh = np.hstack((r0, v0))
    x_rel_eci = np.hstack((Rot.T @ x_rel_lvlh[:3], Rot.T @ x_rel_lvlh[3:]))
    
    return state_target + x_rel_eci

# --- Main Simulation Loop ---

# Iterate over each of the 7 orbits
for orbit_idx, params in enumerate(orbital_params):
    print(f"--- Starting analysis for orbit {orbit_idx + 1} ---")
    a, e, i_p, Omega, omega = params['a'], params['e'], params['i'], params['Omega'], params['omega']

    # Iterate over the 4 orbital points (theta = 0, 90, 180, 270)
    for j in range(N_POINTS_PER_ORBIT):
        theta_deg = 90 * j
        print(f"\n-- Point {j+1}/4 (Theta={theta_deg} deg) of orbit {orbit_idx+1} --")

        thet = np.radians(theta_deg)
        i_rad, Omega_rad, omega_rad = map(np.radians, [i_p, Omega, omega])
        state0_target = kep2car(a, e, i_rad, Omega_rad, omega_rad, thet, mu)
        
        np.random.seed(SEED_FOR_REPRODUCIBILITY)
        print("Pre-generating initial chaser states for this point...")
        initial_chaser_states_bank = []
        for _ in range(N_SIMULATIONS_PER_POINT):
            initial_chaser_states_bank.append(random_chaser_state(state0_target, 9, 100, 300))

        # Run simulations using pre-generated states
        for q in range(N_SIMULATIONS_PER_POINT):
            
            state0_chaser = initial_chaser_states_bank[q]
            print(f"  Simulation {orbit_idx+1}.{j+1} {q+1}/{N_SIMULATIONS_PER_POINT}  ({(orbit_idx*400+j*100+q)/(2800)*100:.1f}%) | Initial Distance: {np.linalg.norm(state0_chaser[:3] - state0_target[:3]) * 1000:.2f} m")
            
            # Time span
            Max_steps_per_simulation = 500
            step_vect = np.arange(0, Max_steps_per_simulation, 1)
            n_steps = len(step_vect)
            docking_time = 0
            
            # MPC Parameters
            Q = np.eye(6)
            R_mat = np.eye(3)*10000   
            dt = 1.5
            Th = 15
            
            # Preallocate variables
            traj_target = np.zeros((6, len(step_vect))); traj_chaser = np.zeros((6, len(step_vect)))
            traj_target[:, 0] = state0_target; traj_chaser[:, 0] = state0_chaser
            error_log = np.zeros((6, len(step_vect))); dock_err = np.zeros((6, len(step_vect)))
            control_log = np.zeros(len(step_vect)); ratio_log = np.zeros(len(step_vect))
            control = np.array([1, 1, 1]); weights_log = np.zeros((3, len(step_vect)))
            dt_log = np.zeros(len(step_vect)); time_log = np.zeros(len(step_vect))

            for i in range(0, len(step_vect) - 1, 1):
                
                # Simulate Navigation noise
                pos_err_val = np.linalg.norm(traj_target[0:3, i] - traj_chaser[0:3, i])
                vel_err_val = np.linalg.norm(traj_target[3:6, i] - traj_chaser[3:6, i])
                pos_std = min(1, pos_err_val / 40); vel_std = min(1e-2, vel_err_val / 30)
                noise = np.hstack([pos_std*np.random.randn(3), vel_std*np.random.randn(3)])

                if i == 0:
                    Rs = R_mat
                dist_ratio = np.clip(pos_err_val / np.linalg.norm(state0_chaser-state0_target), 0.0, 1.0)
                v_max = 2e-5 * (1 + 200 * dist_ratio**1.5)

                pos_weight = min(0.5 * (1 / (dist_ratio + 1e-9))**2, 1e9) 
                vel_weight = 1e5 

                Qs = Q.copy() 
                Qs[0:3, 0:3] = np.eye(3) * pos_weight 
                Qs[3:6, 3:6] = np.eye(3) * vel_weight 

                dt_log[i] = dt

                # Calculate control action with MPC
                h_param = np.linalg.norm(np.cross(traj_target[0:3, i], traj_target[3:6, i]))
                r_param = np.linalg.norm(traj_target[0:3, i])
                t_mom = time.perf_counter()
                control = mpc_control(traj_chaser[:,i], traj_target[:,i]+noise, Th, dt, Qs, Rs, mu, h_param, r_param, u_max, control, v_max)
                time_i = time.perf_counter() - t_mom
                control = control_pert(control)
                
                if np.linalg.norm(control) > u_max*np.sqrt(3):
                    control = u_max*np.sqrt(3) * normalize(control)
                    thrust_exceed_counter += 1

                # Propagate trajectories
                traj_target[:,i+1] = rk8_step(traj_target[:,i], dt, RHS, mu, r_sun, J2, Drag, None)
                traj_chaser[:,i+1] = rk8_step(traj_chaser[:,i], dt, RHS, mu, r_sun, J2, Drag, control)
                docking_time += dt
                time_log[i] = docking_time

                # Debug and plot data
                Rot_m = eci2lvlh(traj_target[:,i])
                error_log[:,i] = traj_chaser[:,i] - traj_target[:,i]
                error_log[:,i] = np.hstack((Rot_m @ error_log[:3,i], Rot_m @ error_log[3:,i]))
                control_log[i] = np.linalg.norm(control)
                
                energy1 = control_log[i] * dt 
                Energy_index_mat1[orbit_idx,j,q] += energy1
                Time_index_mat2[orbit_idx,j,q] += (time_i/dt)*100
                Time_index_mat[orbit_idx,j,q] += time_i

                # Compute error in docking frame
                docking_axis = np.array([0, -1, 0])
                x_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                err_y = error_log[0:3,i] @ docking_axis; err_vy = error_log[3:6,i] @ docking_axis
                err_r2 = (error_log[0:3,i] @ x_axis)**2 + (error_log[0:3,i] @ z_axis) **2
                err_vr2 = (error_log[3:6,i] @ x_axis)**2 + (error_log[3:6,i] @ z_axis) **2
                
                dock_err[1,i] = err_y; dock_err[4,i] = err_vy 
                dock_err[0,i] = error_log[0:3,i] @ x_axis; dock_err[2,i] = error_log[0:3,i] @ z_axis
                dock_err[3,i] = error_log[3:6,i] @ x_axis; dock_err[5,i] = error_log[3:6,i] @ z_axis
                Constraint_index_mat[orbit_idx,j,q] += (err_r2)/((err_y + 5.67e-4)*np.tan(theta))**2 * dt

                if (i + 1) % 500 == 0 or i == n_steps - 1:
                    print(f"Step {i+1}/{n_steps} ({(i+1)/n_steps*100:.1f}%), d={err_y*1000:.2f}")

                if i > 0 and np.abs(err_vy) < (tol_vy) and err_r2 < tol_r**2 and err_vr2 < tol_vr**2 and err_y < tol_y:
                    print("✅ Docking tolerances are satisfied")
                    last_idx = i + 1
                    dists = np.linalg.norm(error_log[:3, :last_idx], axis=0)
                    cone_idx = np.argmax(dists < 0.100); cone2_idx = np.argmax(dists < 0.005)
                    
                    y1 = dock_err[1, cone_idx:cone2_idx]
                    r2_1 = dock_err[0, cone_idx:cone2_idx]**2 + dock_err[2, cone_idx:cone2_idx]**2
                    out1 = r2_1 > (y1 * np.tan(theta))**2
                    
                    y2 = dock_err[1, cone2_idx:last_idx]
                    r2_2 = dock_err[0, cone2_idx:last_idx]**2 + dock_err[2, cone2_idx:last_idx]**2
                    out2 = r2_2 > ((y2 + 5.67e-4) * np.tan(theta))**2
                    
                    out_of_cone = np.concatenate([out1, out2]); overshoots = np.concatenate([y1, y2]) < 0
                    violates_cone = np.any(out_of_cone); overshoot_flag = np.any(overshoots)
                    
                    if violates_cone:
                        print(f"⚠️ {np.sum(out_of_cone)} points might violate the approach cone.")
                    if overshoot_flag:
                        print("⚠️ Overshoot: trajectory goes beyond docking plane!")
                    if not (violates_cone or overshoot_flag):
                        print("✅ Trajectory valid: inside cone and no overshoot.")
                        success_counter += 1
                    print(f"⏱️  Docking time: {docking_time / 60:.2f} [m]")
                    break
                
                if i > 0 and np.linalg.norm(error_log[:,i]-error_log[:,i-1]) < (1e-6):
                    tolerance_counter += 1
                    last_idx = i + 1
                    break
                    
                if i == len(step_vect) - 2:
                    runtime_counter += 1
                    last_idx = i + 1
                    print("❌ Docking not achieved in given time")

            Time_index_mat[orbit_idx,j,q] /= last_idx
            Time_index_mat2[orbit_idx,j,q] /= last_idx
            Constraint_index_mat[orbit_idx,j,q] /= docking_time
            total_stepcounter += last_idx
            
print("\n--- All simulations completed. ---")
elapsed = time.perf_counter() - t_start
print(f"⏱️  Simulation time: {elapsed/60:.6f} [m]")
print("Success: ", success_counter/(N_ORBITS*N_POINTS_PER_ORBIT*N_SIMULATIONS_PER_POINT)*100, "%")
print("Exceeded time: ", runtime_counter/(N_ORBITS*N_POINTS_PER_ORBIT*N_SIMULATIONS_PER_POINT)*100, "%")
print("Got stuck: ", tolerance_counter/(N_ORBITS*N_POINTS_PER_ORBIT*N_SIMULATIONS_PER_POINT)*100, "%")

print("\n--- Saving raw data and generating bar charts ---")

# Save raw data to .npz file
save_merit_indices_raw(
    Energy_index_mat1,
    Time_index_mat2,
    Time_index_mat,
    Constraint_index_mat
)

# Attempt plotting (might fail on server without GUI)
try:
    plot_merit_indices_barcharts(
        Energy_index_mat1,
        Time_index_mat2,
        Time_index_mat,
        Constraint_index_mat
    )
except Exception as e:
    print(f"\nNOTICE: Chart generation failed (expected on headless server).")
    print(f"Error: {e}")
    print("Raw data was still saved. Use the local script for plotting.")

print("\n--- Analysis completed. ---")