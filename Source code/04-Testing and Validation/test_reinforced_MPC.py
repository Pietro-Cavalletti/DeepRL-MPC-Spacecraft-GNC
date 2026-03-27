# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Main execution script for multi-orbit satellite docking 
#              simulations. Evaluates RL-tuned MPC control performance across  
#              orbital parameters (GTO, Molniya, Tundra) and orbital points.
# =============================================================================

import numpy as np
import time
import os
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- MODIFICATION: Import from your libraries and final plotting utilities ---
from Dynamics import RHS
from MPC_RL import mpc_control
from Utils_RL_QR import kep2car, rk8_step, normalize, random_chaser_state, control_pert, sun_pos, eci2lvlh
# Ensure these functions are available and correct for the WS
from Plots import save_merit_indices_raw, plot_merit_indices_barcharts

# --- MODIFICATION: Assuming environments are correctly defined ---
# Adjust paths if these files are not in the same directory
from Docking_env_QR import DockingEnv
from Docking_env_t import DockingEnv as DockingEnv_t

# ==============================================================================
# NEURAL NETWORKS SETUP SECTION (Adapted from RL-MPC script)
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_t = os.path.join(base_dir, "models", "final_model_dt_Th.zip")
VECNORM_PATH_t = os.path.join(base_dir, "models", "final_model_dt_Th_vec.pkl")
MODEL_PATH_Q = os.path.join(base_dir, "models", "final_model_QR.zip")
VECNORM_PATH_Q = os.path.join(base_dir, "models", "final_model_QR_vec.pkl")

# Check if model files exist before proceeding
if not all(os.path.exists(p) for p in [MODEL_PATH_t, VECNORM_PATH_t, MODEL_PATH_Q, VECNORM_PATH_Q]):
    print("ERROR: One or more model files not found. Check paths.")
    # exit() # Uncomment to stop the script if models are missing

def make_env_t(): return DockingEnv_t()
def make_env_q(): return DockingEnv()

# Load Model for dt and Th
try:
    base_env_t = DummyVecEnv([make_env_t])
    env_t = VecNormalize.load(VECNORM_PATH_t, base_env_t)
    env_t.training = False
    env_t.norm_reward = False
    _model_t = RecurrentPPO.load(MODEL_PATH_t, env=env_t)
    _policy_t = _model_t.policy
    _policy_t.eval()
except FileNotFoundError:
    print(f"ERROR: File not found during dt/Th model loading. Check paths: {MODEL_PATH_t}, {VECNORM_PATH_t}")
    _policy_t = None # Set to None to handle the error

# Load Model for Q and R
try:
    base_env_q = DummyVecEnv([make_env_q])
    env_q = VecNormalize.load(VECNORM_PATH_Q, base_env_q)
    env_q.training = False
    env_q.norm_reward = False
    _model_q = RecurrentPPO.load(MODEL_PATH_Q, env=env_q)
    _policy_q = _model_q.policy
    _policy_q.eval()
except FileNotFoundError:
    print(f"ERROR: File not found during Q/R model loading. Check paths: {MODEL_PATH_Q}, {VECNORM_PATH_Q}")
    _policy_q = None # Set to None to handle the error


# Helper functions for prediction with LSTM state management
_lstm_state_t = None
_first_step_t = True
def reset_lstm_t():
    global _lstm_state_t, _first_step_t
    if _policy_t:
        hidden_size = _policy_t.lstm_actor.hidden_size
        device = next(_policy_t.parameters()).device
        _lstm_state_t = (torch.zeros((2, 1, hidden_size), device=device), torch.zeros((2, 1, hidden_size), device=device))
    _first_step_t = True

def predict_dt_th(obs):
    global _lstm_state_t, _first_step_t
    obs = obs.reshape((1, -1))
    episode_start = np.array([_first_step_t], dtype=bool)
    with torch.no_grad():
        action, _lstm_state_t = _policy_t.predict(obs, state=_lstm_state_t, episode_start=episode_start, deterministic=True)
    _first_step_t = False
    return action[0]

_lstm_state_q = None
_first_step_q = True
def reset_lstm_q():
    global _lstm_state_q, _first_step_q
    if _policy_q:
        hidden_size = _policy_q.lstm_actor.hidden_size
        device = next(_policy_q.parameters()).device
        _lstm_state_q = (torch.zeros((2, 1, hidden_size), device=device), torch.zeros((2, 1, hidden_size), device=device))
    _first_step_q = True

def predict_q_weights(obs):
    global _lstm_state_q, _first_step_q
    obs = obs.reshape((1, -1))
    episode_start = np.array([_first_step_q], dtype=bool)
    with torch.no_grad():
        action, _lstm_state_q = _policy_q.predict(obs, state=_lstm_state_q, episode_start=episode_start, deterministic=True)
    _first_step_q = False
    return action[0]

# ==============================================================================
# TEST CONFIGURATION SECTION (Adapted from classical script)
# ==============================================================================

# --- Simulation Parameters ---
N_ORBITS = 7
N_POINTS_PER_ORBIT = 4
N_SIMULATIONS_PER_POINT = 100
SEED_FOR_REPRODUCIBILITY = 42
np.random.seed(SEED_FOR_REPRODUCIBILITY)
t_start = time.perf_counter()

success_counter, runtime_counter, tolerance_counter = 0, 0, 0
thrust_exceed_counter, total_stepcounter = 0, 0

# Physical and Controller parameters
mu = 398600.4418; J2 = 1; Drag = 1
u_max = 2e-5
theta = np.radians(10)  # Approach cone angle

# Docking tolerances
tol_vy=1e-4; tol_r=1e-4; tol_vr=4e-5; tol_y=2e-4
r_sun = sun_pos(np.random.uniform(0, 365.25))

# RL Parameters (for network output de-normalization)
dt_range=1.7; dt_avg=2.3
Th_range=13; Th_avg=17
Qp_range=4; Qp_avg=4
R_range=3; R_avg=9

# --- MODIFICATION: Definition of 7 orbits as in the classical script ---
orbital_params = [
    {'a': 24414, 'e': 0.7267, 'i': 7.0, 'Omega': 0, 'omega': 0},
    {'a': 24414, 'e': 0.7267, 'i': 28.5, 'Omega': 0, 'omega': 0},
    {'a': 24364, 'e': 0.7303, 'i': 7.0, 'Omega': 0, 'omega': 0},
    {'a': 27300, 'e': 0.73, 'i': 63.4, 'Omega': 0, 'omega': 0},
    {'a': 25448, 'e': 0.73, 'i': 63.4, 'Omega': 0, 'omega': 0},
    {'a': 42164, 'e': 0.3, 'i': 63.4, 'Omega': 0, 'omega': 0},
    {'a': 42164, 'e': 0.4, 'i': 63.4, 'Omega': 0, 'omega': 0}
]

# --- MODIFICATION: Initialization of 3D result matrices ---
Energy_index_mat1 = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Time_index_mat = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Time_index_mat2 = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))
Constraint_index_mat = np.zeros((N_ORBITS, N_POINTS_PER_ORBIT, N_SIMULATIONS_PER_POINT))

# ==============================================================================
# MAIN SIMULATION LOOP (Classical structure with RL logic)
# ==============================================================================
# Iterate over each of the 7 orbits
for orbit_idx, params in enumerate(orbital_params):
    print(f"--- Starting analysis for orbit {orbit_idx + 1}/{N_ORBITS} ---")
    a, e, i, Omega, omega = params['a'], params['e'], params['i'], params['Omega'], params['omega']

    # Iterate over 4 orbital points (theta = 0, 90, 180, 270)
    for j in range(N_POINTS_PER_ORBIT):
        theta_deg = 90 * j
        print(f"\n-- Point {j+1}/{N_POINTS_PER_ORBIT} (Theta={theta_deg} deg) of orbit {orbit_idx+1} --")

        thet = np.radians(theta_deg)
        i_rad, Omega_rad, omega_rad = map(np.radians, [i, Omega, omega])
        state0_target = kep2car(a, e, i_rad, Omega_rad, omega_rad, thet, mu)
        
        # --- MODIFICATION: Pre-generation of initial states for reproducibility ---
        np.random.seed(SEED_FOR_REPRODUCIBILITY)
        initial_chaser_states_bank = [random_chaser_state(state0_target, 9, 100, 300) for _ in range(N_SIMULATIONS_PER_POINT)]

        # Execute N simulations for this orbital point
        for q in range(N_SIMULATIONS_PER_POINT):
            state0_chaser = initial_chaser_states_bank[q]
            dist_iniziale = np.linalg.norm(state0_chaser[:3] - state0_target[:3]) * 1000
            print(f"  Simulation {orbit_idx+1}.{j+1}.{q+1}/{N_SIMULATIONS_PER_POINT} ({(orbit_idx*400+j*100+q)/(2800)*100:.1f}%)| Initial distance: {dist_iniziale:.2f} m")

            # Reset simulation variables
            Max_steps_per_simulation = 10 * 60  # Increased for safety
            step_vect = np.arange(0, Max_steps_per_simulation, 1)
            n = len(step_vect)
            docking_time, last_idx = 0, n - 1

            # Preallocation
            traj_target = np.zeros((6, n)); traj_chaser = np.zeros((6, n))
            traj_target[:, 0], traj_chaser[:, 0] = state0_target, state0_chaser
            error_log = np.zeros((6, n)); dock_err = np.zeros((6, n))
            control_log = np.zeros(n);
            control = np.array([0.0, 0.0, 0.0])

            # --- MODIFICATION: Reset LSTM states at the beginning of each simulation ---
            reset_lstm_t()
            reset_lstm_q()
            dt_norm, Th_norm, Qp_norm, R_norm = 0, 0, 0, 0

            # Normalization for network input
            d0 = np.linalg.norm(state0_target - state0_chaser) * 1000
            y_max = d0 * np.sin(np.radians(9))
            delta_x_max = np.array([y_max, d0, y_max, 0.2, 1, 0.2]) / 1000

            # Main trajectory loop
            for i in range(n - 1):
                # Relative state and noise calculation
                Rot = eci2lvlh(traj_target[:, i])
                error_log[:, i] = traj_chaser[:, i] - traj_target[:, i]
                error_log[:, i] = np.hstack((Rot @ error_log[:3, i], Rot @ error_log[3:, i]))
                pos_err = np.linalg.norm(error_log[:3, i]); vel_err = np.linalg.norm(error_log[3:, i])
                pos_std = min(1e-3, pos_err / 40); vel_std = min(1e-5, vel_err / 30)
                noise = np.hstack([pos_std * np.random.randn(3), vel_std * np.random.randn(3)])
                noise_lvlh = np.hstack((Rot @ noise[:3], Rot @ noise[3:]))

                # --- RL-MPC LOGIC ---
                # Prepare observation for dt/Th network
                state_obs = np.array((error_log[:, i] + noise_lvlh) / delta_x_max, dtype=np.float32)
                u_last_obs = np.array([np.linalg.norm(control / u_max)], dtype=np.float32)
                obs_t = np.concatenate([state_obs, np.array([dt_norm, Th_norm], dtype=np.float32), u_last_obs])
                
                dt_norm, Th_norm = np.clip(predict_dt_th(obs_t), -1, 1)
                dt = dt_norm * dt_range + dt_avg
                Th = Th_norm * Th_range + Th_avg

                # Prepare observation for Q/R network
                q_input = np.array([np.clip(Qp_norm, -1, 1), np.clip(R_norm, -1, 1)], dtype=np.float32)
                obs_q = np.concatenate([state_obs, q_input, u_last_obs])
                
                Qp_norm, R_norm = predict_q_weights(obs_q)
                Qp_weight = 10**(np.clip(Qp_norm, -1, 1) * Qp_range + Qp_avg)
                Qv_weight = 10**(2)
                R_weight  = 10**(np.clip(R_norm, -1, 1) * R_range + R_avg)
                
                Qs = np.diag([Qp_weight]*3 + [Qv_weight]*3)
                Rs = np.eye(3) * R_weight
                # --- END RL-MPC LOGIC ---

                dist_ratio = np.clip(pos_err / (d0 / 1000), 0.0, 1.0)
                v_max = u_max * (1 + 200 * dist_ratio**1.5)

                # MPC Control calculation
                h = np.linalg.norm(np.cross(traj_target[0:3, i], traj_target[3:6, i]))
                r = np.linalg.norm(traj_target[0:3, i])
                t_mom = time.perf_counter()
                control = mpc_control(traj_chaser[:, i], traj_target[:, i] + noise, Th, dt, Qs, Rs, mu, h, r, u_max, control, v_max)
                time_i = time.perf_counter() - t_mom

                if np.linalg.norm(control) > u_max * np.sqrt(3):
                    control = u_max * np.sqrt(3) * normalize(control)
                    thrust_exceed_counter += 1

                # Propagation
                traj_target[:, i+1] = rk8_step(traj_target[:, i], dt, RHS, mu, r_sun, J2, Drag, None)
                traj_chaser[:, i+1] = rk8_step(traj_chaser[:, i], dt, RHS, mu, r_sun, J2, Drag, control)
                docking_time += dt

                # --- MODIFICATION: Update merit indices with 3D indexing ---
                control_norm = np.linalg.norm(control)
                Energy_index_mat1[orbit_idx, j, q] += control_norm * dt
                Time_index_mat2[orbit_idx, j, q] += (time_i / dt) * 100
                Time_index_mat[orbit_idx, j, q] += time_i
                
                # Docking frame error and constraint calculation
                docking_axis = np.array([0, -1, 0]); x_axis = np.array([1, 0, 0]); z_axis = np.array([0, 0, 1])
                err_y = error_log[0:3, i] @ docking_axis; err_vy = error_log[3:6, i] @ docking_axis
                err_r2 = (error_log[0:3, i] @ x_axis)**2 + (error_log[0:3, i] @ z_axis)**2
                err_vr2 = (error_log[3:6, i] @ x_axis)**2 + (error_log[3:6, i] @ z_axis)**2
                Constraint_index_mat[orbit_idx, j, q] += (err_r2) / ((err_y + 5.67e-4) * np.tan(theta))**2 * dt

                # Termination conditions (successful docking, stall, max time)
                if abs(err_vy) < tol_vy and err_r2 < tol_r**2 and err_vr2 < tol_vr**2 and err_y < tol_y:
                    print("  ✅ Docking successful.")
                    success_counter += 1
                    last_idx = i + 1
                    break
                if i > 0 and np.linalg.norm(error_log[:, i] - error_log[:, i-1]) < 1e-6:
                    print("  ⚠️ Stall detected. Interrupting.")
                    tolerance_counter += 1
                    last_idx = i + 1
                    break
            
            if i == n - 2: # If loop terminated due to max time
                print("  ❌ Max time reached.")
                runtime_counter += 1
                last_idx = i + 1

            # Final normalization of average indices
            if last_idx > 0:
                Time_index_mat[orbit_idx, j, q] /= last_idx
                Time_index_mat2[orbit_idx, j, q] /= last_idx
            if docking_time > 0:
                Constraint_index_mat[orbit_idx, j, q] /= docking_time
            total_stepcounter += last_idx

# ==============================================================================
# FINAL SECTION: STATISTICS AND SAVING (Adapted from classical script)
# ==============================================================================
print("\n--- All simulations completed. ---")
elapsed = time.perf_counter() - t_start
total_sims = N_ORBITS * N_POINTS_PER_ORBIT * N_SIMULATIONS_PER_POINT
print(f"⏱️  Total simulation time: {elapsed/60:.2f} [min]")
print(f"Success rate: {success_counter/total_sims*100:.2f}%")
print(f"Timeout rate: {runtime_counter/total_sims*100:.2f}%")
print(f"Stall rate: {tolerance_counter/total_sims*100:.2f}%")

# --- MODIFICATION: Final calls to save raw data and generate bar charts ---
print("\n--- Saving raw data and generating bar charts ---")

# Save raw data in an .npz file (safe for WS)
save_merit_indices_raw(
    Energy_index_mat1,
    Time_index_mat2,
    Time_index_mat,
    Constraint_index_mat
)

# Attempt to plot charts (may fail on WS without graphical backend)
try:
    plot_merit_indices_barcharts(
        Energy_index_mat1,
        Time_index_mat2,
        Time_index_mat,
        Constraint_index_mat
    )
except Exception as e:
    print(f"\nNOTICE: Chart generation failed (expected on headless servers).")
    print(f"Error: {e}")
    print("Raw data has been saved successfully. Use a local script for plotting.")

print("\n--- Analysis completed. ---")