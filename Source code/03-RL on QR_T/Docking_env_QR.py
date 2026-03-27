# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Gymnasium Environment for Autonomous Satellite Docking.
#              The RL agent tunes MPC parameters (Q, R) to optimize energy,
#              safety constraints, and computational effort.
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Dynamics_RL import integrate
from MPC_RL import mpc_control
from Utils_RL_QR import random_chaser_state, sun_pos, kep2car, eci2lvlh, control_pert

import warnings

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor.*",
    category=UserWarning
)


class DockingEnv(gym.Env):
    def __init__(self, mu=398600.4418, record=False):
        super().__init__()
        self.episode_count = 1
        self.success_count = 0
        self.total_step_count = 0
        self.episode_step_count = 0
        self.time_count = 0
        self.total_steps = 1_000_000/16
        self.mu = mu
        self.start_time = time.time()
        self.episode_end_steps = []
        self.record = record

       
        self.Qp_min = 0
        self.Qp_max = 8
        self.Qp_range=(self.Qp_max-self.Qp_min)/2
        self.Qp_avg=(self.Qp_min+self.Qp_max)/2

        self.Qv_weight=2

        self.R_min = 6
        self.R_max = 12
        self.R_range=(self.R_max-self.R_min)/2
        self.R_avg=(self.R_min+self.R_max)/2

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Costants
        self.J2 = 1; self.Drag = 1; self.srp= 1; self.noise = 1; self.dist_contr = 1 #0 to disable disturbances, 1 to enable
        self.u_max = 2e-5
        self.Q = np.eye(6)
        self.Q[3:6, 3:6] = np.eye(3) * 10**self.Qv_weight
        self.R = np.eye(3)
        self.theta = np.radians(10)

        # State variables
        self.target = None
        self.chaser = None
        self.trajectory = []
        self.control_log = []
        self.weights_log = []
        self.dt_log = []
        self.time_log = []
        self.error_log = []
        self.dock_err_log = []

        # Reward indices
        self.energy_index = 0.0
        self.constraint_index = 0.0
        self.time_index = 0.0
        self.time_elapsed = 0.0
        self.reward_accumulator = 0.0
        self.reward_interval = 0.4   #higher for more sparse reward
        self.docked = False
        self.total_r_energy = 0
        self.total_r_constraint =0
        self.total_r_computation  =0
        self.total_r_pos =0
        self.total_r_vel =0
        self.total=0


        

        self.err_lvlh=np.array([1,1,1,1,1,1])/100
        self.prev_reward = 0.0
        self.cone_violation=0
        self.theta0_max=9 #deg
        self.d0_min=100
        self.d0_max=105

        self.y_max=1
        self.delta_x_max= np.array([self.y_max, self.d0_max, self.y_max, 0.2, 1, 0.2 ])/1000

   

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.r_sun = sun_pos(np.random.uniform(0, 365.25))*self.srp
        self.start_time = time.time()

        # Stato target
        a, e = 24000, 0.73
        i, Ω, ω, θ = map(np.radians, [0, 0, 0, np.random.uniform(0, 360)])
        self.target = kep2car(a, e, i, Ω, ω, θ, self.mu)

        #Stato chaser
        frac = min(self.total_step_count / self.total_steps, 1)  #curriculum learning 
        self.chaser = random_chaser_state(self.target, self.theta0_max, self.d0_min, self.d0_max)
        self.d0 = np.linalg.norm(self.target-self.chaser)

        self.d0_max=self.d0*1000
        self.y_max=self.d0_max*np.sin(self.theta0_max*np.pi/180)
        self.delta_x_max= np.array([self.y_max, self.d0_max, self.y_max, 0.2, 1, 0.2 ])/1000

        # Reset log
        self.trajectory = [self.chaser.copy()] if self.record else []
        self.control_log = []
        self.weights_log = []
        self.dt_log = []
        self.time_log = []
        self.error_log = []
        self.dock_err_log = []

        self.energy_index = 0.0
        self.constraint_index = 0.0
        self.time_index = 0.0
        self.time_elapsed = 0.0
        self.time_count = 0.0
        self.episode_step_count = 0
        self.docked = False
        self.y_min = self.d0
        self.total_r_energy = 0
        self.total_r_constraint =0
        self.total_r_computation  =0
        self.total_r_pos =0
        self.total_r_vel =0
        self.total=0

        self.u_last = np.zeros(3, dtype=np.float32)  
        self.mpc_failure_count = 0
        Rot = eci2lvlh(self.target)
        initial_err = self.chaser - self.target
        initial_err_lvlh = np.hstack((Rot @ initial_err[:3], Rot @ initial_err[3:]))

        obs = np.concatenate([initial_err_lvlh/self.delta_x_max, np.zeros(3, dtype=np.float32)])
        
        return obs.astype(np.float32), {}

    def step(self, action, dt , Th):
        info = {}
        self.episode_step_count +=1
        #MPC tuning
        self.Qp_weight = 10**(np.clip(action[0], -1, 1) * self.Qp_range + self.Qp_avg )
        self.R_weight  = 10**(np.clip(action[1], -1, 1) * self.R_range + self.R_avg )

        Qs = self.Q.copy()
        Qs[0:3, 0:3] = np.eye(3) * self.Qp_weight
        
        Rs =  np.eye(3) * self.R_weight

        self.dt = dt
        self.Th = Th
        
        pos_err = np.linalg.norm(self.target[0:3] - self.chaser[0:3])
        dist_ratio = np.clip(pos_err / self.d0, 0.0, 1.0)
        v_max = 2e-5 * (1+200*dist_ratio**1.5)

        #Simulate Navigation noise
        vel_err = np.linalg.norm(self.target[3:6] - self.chaser[3:6])
        pos_std = min(1,  pos_err / 40); vel_std = min(1e-2,  vel_err / 30)
        noise = self.noise * np.hstack([pos_std*np.random.randn(3), vel_std*np.random.randn(3)])

        # Compute control
        t0 = time.perf_counter()
        u = mpc_control(
            self.chaser, self.target+noise,
            self.Th, self.dt,
            Q=Qs, R=Rs, mu=self.mu,
            h=np.linalg.norm(np.cross(self.target[0:3], self.target[3:6])),
            r=np.linalg.norm(self.target[0:3]),
            u_max=self.u_max, u_last=self.u_last, v_max=v_max
        )
        t1 = time.perf_counter()
        comp_time = t1 - t0

        if u is None or np.any(np.isnan(u)):
            self.mpc_failure_count += 1
            u =0*self.u_last.copy() 
        else:
            self.mpc_failure_count = 0
        if self.mpc_failure_count >= 3:
            done = True
            reward = -5  
            info = {'mpc_failure': True}
            return obs.astype(np.float32), reward, done, False, {}
        if self.dist_contr==1:
            u=control_pert(u)
        u_norm = np.linalg.norm(u)
        u_clipped = u if u_norm <= self.u_max * np.sqrt(3) else self.u_max * np.sqrt(3)* u / u_norm
        self.u_last = u_clipped.astype(np.float32)

        # Propagate
        new_chaser = integrate(self.chaser, u_clipped, self.dt, self.mu, self.r_sun, self.J2, self.Drag)
        new_target = integrate(self.target, None, self.dt, self.mu, self.r_sun, self.J2, self.Drag)

        # Error LVLH
        Rot = eci2lvlh(new_target)
        err = new_chaser - new_target
        self.err_lvlh = np.hstack((Rot @ err[:3], Rot @ err[3:]))
        dock_err = self.compute_docking_error(self.err_lvlh)
        noise_lvlh=np.hstack((Rot @ noise[:3], Rot @ noise[3:]))

        # Update logs
        self.energy_index += np.linalg.norm(u_clipped) * self.dt
        self.constraint_index += (dock_err['r2']) / ((dock_err['y'] + 5.67e-4) * np.tan(self.theta))**2 * self.dt
        self.time_index += comp_time 
        self.time_elapsed += self.dt
        self.time_count += self.dt

        

        if self.record:
            self.trajectory.append(new_chaser.copy())
        self.control_log.append(u_norm)
        self.time_log.append(self.time_elapsed)
        self.error_log.append(self.err_lvlh)
        self.dock_err_log.append(dock_err)
        self.chaser = new_chaser
        self.target = new_target

        # --- Termination checks ---
        Max_docking_time = 10 #m
        self.cone_violation = dock_err['r2'] / (((dock_err['y'] + 0.6e-3) * np.tan(self.theta))**2)

        self.docked = False
        done = False

        if self.time_elapsed > Max_docking_time*60:
            print(f"[x] Timeout: no docking in {Max_docking_time:.0f} min.")
            done = True
        elif self.check_docked(self.err_lvlh):
            self.docked = True
            done = True
        elif dock_err['y'] < 0:
            print("[x] Overshoot: surpassed docking plane.")
            done = True
        elif self.cone_violation >= 1:
            print("[x] Cone violation: out of approach corridor.")
            done = True
        

        
        #Calculate reward
        reward= self.compute_intermediate_reward(dock_err, comp_time, np.linalg.norm(u_clipped), pos_err, vel_err)

        if done:
            self.episode_count += 1
            reward += self.compute_final_reward(done)
            self.episode_end_steps.append(self.total_step_count)
            info["episode_perf/energy_index"] = self.energy_index
            info["episode_perf/time_index"] = self.time_index /self.episode_step_count 
            info["episode_perf/constraint_index"] = self.constraint_index /self.time_elapsed

        if  dock_err['y'] <= self.y_min:
            self.y_min = dock_err['y']
        
        self.prev_action = np.clip(np.array(action, dtype=np.float32), -1, 1)
        obs = np.concatenate([(self.err_lvlh + noise_lvlh)/self.delta_x_max, self.prev_action,  [np.linalg.norm(u_clipped) / self.u_max]])
        self.total_step_count+=1
       
       

        return obs.astype(np.float32), reward, done, False, info



    def compute_docking_error(self, err):
        # Docking frame: Y = forward, XZ = radial
        y_axis = np.array([0, -1, 0])
        x_axis = np.array([1, 0, 0])
        z_axis = np.array([0, 0, 1])
        err_y = err[:3] @ y_axis
        err_vy = err[3:] @ y_axis
        err_vr2= (err[3:] @ x_axis)**2 + (err[3:] @ z_axis)**2
        err_r2 = (err[:3] @ x_axis)**2 + (err[:3] @ z_axis)**2
        return {'y': err_y, 'vy': err_vy, 'r2': err_r2, 'vr2': err_vr2}
    

    def check_docked(self, err_lvlh):
        dock_err = self.compute_docking_error(err_lvlh)
        tol_vy=1e-4; tol_r=1e-4; tol_vr=4e-5; tol_y=2e-4
        if (np.abs((dock_err['vy'])) < tol_vy
            and (dock_err['r2']) < tol_r**2
            and (dock_err['vr2']) < tol_vr**2
            and (dock_err['y']) < tol_y):
            self.success_count += 1
            elapsed = time.time() - self.start_time
            print(f"[✓] Docking achieved! Successes: {self.success_count}/{self.episode_count}| N_steps: {self.episode_step_count:1f}  -  Docking time: {self.time_elapsed/60:.2f}m")
            return True
        else: 
            if self.total_step_count%100==0:
                print(f"Step {self.total_step_count}: y_dist={dock_err['y']*1000:.2f},  Qp= {self.Qp_weight:.2e}, R= {self.R_weight:.2e}")
                print(f"Th= {self.Th:.2f}, dt= {self.dt:.2f}")
            return False
        

    def check_done(self, err):
        if self.check_docked(err): return True
        if np.linalg.norm(err[:3]) > 1.0: 
             print(f"[!] Terminating episode — chaser too far: {np.linalg.norm(err[:3]):.2f} m")
             return True
        return False



    def compute_intermediate_reward(self, dock_err, comp_time, u, pos_err, vel_err):
        # 1. Reward Weights (primary point for hyperparameter tuning)
        w_pos1, w_pos2, w_angle = 100, 1.5, 3
        w_control = 2

        # --- GRADUAL SCALING OF THE ANGULAR PENALTY ---
        # Reduces the impact of angular errors as the chaser gets very close to the target
        angle_transition_dist = 10e-3  
        min_angle_weight_factor = 0.3
        angle_weight_scaler = np.clip(dock_err["y"] / angle_transition_dist, min_angle_weight_factor, 1.0)
        w_angle_effective = w_angle * angle_weight_scaler

        # 2. Calculation of Normalized Error Terms
        norm_vel_ratio = np.linalg.norm(self.err_lvlh[3:] / self.delta_x_max[3:]) / np.sqrt(3)
        current_angle_rad = np.arctan2(np.sqrt(dock_err['r2']), np.abs(dock_err['y']))
        angular_error_ratio = np.clip(current_angle_rad / 0.17, 0, 3)
        control_term = np.linalg.norm(u) / self.u_max

        # 3. Penalty Calculation
        # r_pos combines a linear distance term and an exponential shaping term for precision
        r_pos =  w_pos1 * ((self.y_min - dock_err["y"])/ self.delta_x_max[1]) + w_pos2 * np.exp(-50 * dock_err["y"]/ self.delta_x_max[1]) 
        r_ang = -w_angle_effective * angular_error_ratio**2
        r_ctr = -w_control * control_term
    
        reward = r_pos + r_ang + r_ctr

        # 4. Accumulation for End-of-Episode Logging
        self.total_r_pos += r_pos
        self.total_r_constraint += r_ang  
        self.total_r_energy += r_ctr      
        self.total += reward

        # 5. Intermediate Debug Logging 
        if self.total_step_count % 100 == 0:
            print(f"r_pos={r_pos:.2e}, r_ang={r_ang:.2e}, r_ctrl={r_ctr:.2e},  -> r={reward:.2e}")
        return reward

    def compute_final_reward(self, done):
        if not done:
            return 0.0
        
        # Sparse final reward/penalty to provide a clear objective (sparse signal)
        zeta = 100   # Bonus for successful docking
        kappa = 100  # Penalty for mission failure
        
        if self.docked:
            reward = zeta
            print(f"[🏁 Docked: {self.docked}]")
            print(f"Total position reward   ≈ {self.total_r_pos:.2e}")
            print(f"Total constraint reward ≈ {self.total_r_constraint:.2e}")
            print(f"Total energy reward     ≈ {self.total_r_energy:.2e}")
            print(f"Total intermediate reward   ≈ {self.total:.2e}")
            print(f"Mean intermediate reward    ≈ {self.total/self.episode_step_count:.2e}")
        else:
            reward = -kappa
            print(f"[Docking FAIL] → - {kappa:.1e}")
            print(f"Total position reward   ≈ {self.total_r_pos:.2e}")
            print(f"Total constraint reward ≈ {self.total_r_constraint:.2e}")
            print(f"Total energy reward     ≈ {self.total_r_energy:.2e}")
            print(f"Total intermediate reward  ≈ {self.total:.2e}")
            print(f"Mean intermediate reward   ≈ {self.total/self.episode_step_count:.2e}")
        return reward