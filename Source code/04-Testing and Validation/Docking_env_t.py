import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from Dynamics_RL import integrate
from MPC_RL import mpc_control
from Utils_RL_QR import random_chaser_state, sun_pos, kep2car, eci2lvlh, control_pert

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
        self.dt_history = []
        self.Th_history = []
        self.episode_end_steps = []
        self.record = record

        self.dt_min = 0.6
        self.dt_max = 4.0
        self.dt_range=(self.dt_max-self.dt_min)/2
        self.dt_avg=(self.dt_min+self.dt_max)/2

        self.Th_min = 4.0
        self.Th_max = 30.0
        self.Th_range=(self.Th_max-self.Th_min)/2
        self.Th_avg=(self.Th_min+self.Th_max)/2

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
        self.R = np.eye(3) * 10000
        self.theta = np.radians(10)
        self.theta0_max=7 #deg
        self.d0_min=100
        self.d0_max=105

        self.y_max=self.d0_max*np.sin(self.theta0_max*np.pi/180)
        self.delta_x_max= np.array([self.y_max, self.d0_max, self.y_max, 0.2, 1, 0.2 ])/1000

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
        self.reward_time=0.0
        self.reward_interval = 0.0   #higher for more sparse reward
        self.total_inter_r_energy = 0.0
        self.total_inter_r_constraint = 0.0
        self.total_inter_r_computation = 0.0
        self.docked = False

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
        r_min = self.d0_min
        r_max = self.d0_max
        theta_max= 3 + frac * 6  
        self.chaser = random_chaser_state(self.target, theta_max, r_min, r_max)
        self.d0 = np.linalg.norm(self.target-self.chaser)

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
        self.total_inter_r_energy = 0.0
        self.total_inter_r_constraint = 0.0
        self.total_inter_r_computation = 0.0
        self.docked = False

        self.u_last = np.zeros(3, dtype=np.float32)  
        self.reward_time=0.0
        self.reward_accumulator=0.0
        self.mpc_failure_count = 0
        Rot = eci2lvlh(self.target)
        initial_err = self.chaser - self.target
        initial_err_lvlh = np.hstack((Rot @ initial_err[:3], Rot @ initial_err[3:]))

        obs = np.concatenate([initial_err_lvlh, np.zeros(3, dtype=np.float32)])
        return obs.astype(np.float32), {}

    def step(self, action):
        info = {}
        self.episode_step_count += 1
        #MPC tuning
        dt = np.clip(action[0], -1, 1) * self.dt_range + self.dt_avg
        Th = np.clip(action[1], -1, 1) * self.Th_range + self.Th_avg
        N = int(np.ceil(Th / dt))
        self.Th = N * dt   
        self.dt = dt
        # Adaptive weights
        pos_err = np.linalg.norm(self.target[0:3] - self.chaser[0:3])
        dist_ratio = np.clip(pos_err / self.d0, 0.0, 1.0)
        v_max=2e-5*(1+200*dist_ratio**1.5)
        pos_weight = 0.5 * (1/dist_ratio)**2
        vel_weight = 1e5 
        Qs =self.Q; 
        Qs[0:3, 0:3] = np.eye(3) * pos_weight 
        Qs[3:6, 3:6] = np.eye(3) * vel_weight
        Rs = self.R

        #Simulate Navigation noise
        vel_err = np.linalg.norm(self.target[3:6] - self.chaser[3:6])
        pos_std = min(1,  pos_err / 40); vel_std = min(1e-2,  vel_err / 30)
        noise = self.noise * np.hstack([pos_std*np.random.randn(3), vel_std*np.random.randn(3)])

        # Compute control
        t0 = time.perf_counter()
        u = mpc_control(
            self.chaser, self.target+noise,
            self.Th, self.dt,
            Qs, Rs, mu=self.mu,
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
        Rot = eci2lvlh(self.target)
        err = new_chaser - new_target
        self.err_lvlh = np.hstack((Rot @ err[:3], Rot @ err[3:]))
        dock_err = self.compute_docking_error(self.err_lvlh)
        noise_lvlh=np.hstack((Rot @ noise[:3], Rot @ noise[3:]))

        # Update logs
        self.energy_index += np.linalg.norm(u_clipped) * self.dt
        self.constraint_index += (dock_err['r2']) / ((dock_err['y'] + 5.67e-4) * np.tan(self.theta))**2 * self.dt
        self.time_index += comp_time/self.dt
        self.time_elapsed += self.dt
        self.time_count += self.dt

        if self.record:
            self.trajectory.append(new_chaser.copy())
        self.control_log.append(u_norm)
        self.time_log.append(self.time_elapsed)
        self.error_log.append(self.err_lvlh)
        self.dock_err_log.append(dock_err)
        self.dt_history.append(self.dt)
        self.Th_history.append(np.clip(action[1], self.Th_min, self.Th_max))  # Store raw Th
        self.chaser = new_chaser
        self.target = new_target

        # --- Termination checks ---
        Max_docking_time = 800.0 #s
        self.cone_violation = dock_err['r2'] / (((dock_err['y'] + 1e-3) * np.tan(self.theta))**2)

        self.docked = False
        done = False

        if self.time_elapsed > Max_docking_time:
            print(f"[x] Timeout: no docking in {Max_docking_time/60:.0f} min.")
            done = True
        elif self.check_docked(self.err_lvlh):
            self.docked = True
            done = True
        elif self.cone_violation >= 1:
            print("[x] Cone violation: out of approach corridor.")
            done = True
        elif dock_err['y'] < 0:
            print("[x] Overshoot: surpassed docking plane.")
            done = True

        #Calculate reward
        rew = self.compute_intermediate_reward(dock_err, comp_time, np.linalg.norm(u_clipped))
        reward = rew/1000/self.dt
        self.reward_accumulator += rew
        self.reward_time +=self.dt
        if self.reward_time >= self.reward_interval:
            reward=self.reward_accumulator
            self.reward_accumulator=0
            self.reward_time -= self.reward_interval

        if done:
            self.episode_count += 1
            self.time_index /= self.episode_step_count
            self.constraint_index /= self.time_elapsed
            reward += self.compute_final_reward(done)
            self.episode_end_steps.append(self.total_step_count)
            info["episode_perf/energy_index"] = self.energy_index
            info["episode_perf/time_index"] = self.time_index
            info["episode_perf/constraint_index"] = self.constraint_index

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
            if self.total_step_count%50==0:
                print(f"Step {self.total_step_count}: y_dist={dock_err['y']*1000:.2f},  dt= {self.dt:.2f}, Th= {self.Th:.2f}")
            return False
        

    def check_done(self, err):
        if self.check_docked(err): return True
        if np.linalg.norm(err[:3]) > 1.0: 
             print(f"[!] Terminating episode — chaser too far: {np.linalg.norm(err[:3]):.2f} m")
             return True
        return False
    

    def compute_intermediate_reward(self, dock_err, comp_time, u):

        epsilon = 5.67e-4
        cone_violation = (dock_err['r2']) / ((dock_err['y'] + epsilon) * np.tan(self.theta))**2
        w_e = 1.1e10
        w_t = 30
        w_c = 1200
        e_term = w_e * u**2 * self.dt
        c_term = w_c * cone_violation**4  * self.dt
        t_term = w_t * (comp_time / self.dt)**2
        reward = - (e_term + c_term + t_term)
        self.total_inter_r_energy +=e_term
        self.total_inter_r_constraint +=c_term
        self.total_inter_r_computation += t_term
        return reward


    def compute_final_reward(self, done):
        if not done:
            return 0.0

        if self.docked:
            w_e, w_c, w_t = 5e1, 2e0, 5e-1
            e_term = w_e * self.energy_index/self.d0
            c_term = w_c * self.constraint_index 
            t_term = w_t * self.time_index 
            reward = 1
            print(f"[🏁 Docked: {self.docked}] Energy: {e_term:.2e}, "
            f"Constraint: {c_term:.2e} , "
            f"Time: {t_term:.2e} → Reward: {reward:.2e}")
            print(f"Total intermediate energy reward     ≈ -{1.1*self.total_inter_r_energy:.2e}")
            print(f"Total intermediate constraint reward  ≈ -{1.1*self.total_inter_r_constraint:.2e}")
            print(f"Total intermediate computation reward  ≈ -{1.1*self.total_inter_r_computation:.2e}")
            total = self.total_inter_r_energy + self.total_inter_r_constraint + self.total_inter_r_computation
            print(f"Total intermediate reward             ≈ -{1.1*total:.2e}")
        else:
            reward = -100
        return reward
