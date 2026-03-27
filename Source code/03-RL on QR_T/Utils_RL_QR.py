# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Reinforcement Learning utility classes and orbital mechanics functions.
#              Includes custom Stable-Baselines3 callbacks for performance logging,
#              action analysis, and coordinate transformation utilities.
# =============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import io
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class EvalCallbackWithPerfLogging(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations_infos = []
        self.current_eval_infos = []
        self.current_eval_rewards = []
        self.episode_rewards = []  # To collect rewards per episode
        self.is_evaluating = False
    
    def _on_rollout_start(self) -> None:
        """Called at the start of each evaluation"""
        self.is_evaluating = True
        self.current_eval_infos.clear()
        self.current_eval_rewards.clear()
        self.episode_rewards.clear()
        print(f"[EVAL] Starting evaluation at step {self.num_timesteps}")
        super()._on_rollout_start()
    
    def _on_step(self) -> bool:
        if self.is_evaluating:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            rewards = self.locals.get("rewards", [])
            
            # Collect all step-by-step rewards
            if rewards is not None:
                if isinstance(rewards, np.ndarray):
                    self.current_eval_rewards.extend(rewards.flatten())
                else:
                    self.current_eval_rewards.extend([rewards] if not isinstance(rewards, list) else rewards)
            
            # Collect info when an episode finishes
            for i, (info, done) in enumerate(zip(infos, dones)):
                if done and isinstance(info, dict):
                    self.current_eval_infos.append(info)
                    
                    # If episode reward is in info, collect it
                    if 'episode' in info and 'r' in info['episode']:
                        self.episode_rewards.append(info['episode']['r'])
                    elif 'episode_reward' in info:
                        self.episode_rewards.append(info['episode_reward'])
        
        return super()._on_step()
    
    def _on_rollout_end(self):
        """Called at the end of each evaluation"""
        # FIRST call the parent class method to calculate eval/mean_reward
        super()._on_rollout_end()
        
        if self.is_evaluating:
            print(f"[EVAL] Evaluation completed at step {self.num_timesteps}")
            
            # Log custom performance metrics
            if self.current_eval_infos:
                self.evaluations_infos.append(self.current_eval_infos.copy())
                
                energy, time_, constraint = [], [], []
                
                for info in self.current_eval_infos:
                    if "episode_perf/energy_index" in info:
                        energy.append(info["episode_perf/energy_index"]/0.0031)
                    if "episode_perf/time_index" in info:
                        time_.append(info["episode_perf/time_index"]/0.17)
                    if "episode_perf/constraint_index" in info:
                        constraint.append(info["episode_perf/constraint_index"]/0.13)
                
                if energy:
                    self.logger.record("eval/episode_perf/energy_index", float(np.mean(energy)))
                    print(f"[EVAL] Energy Index: {np.mean(energy):.4f}")
                if time_:
                    self.logger.record("eval/episode_perf/time_index", float(np.mean(time_)))
                    print(f"[EVAL] Time Index: {np.mean(time_):.4f}")
                if constraint:
                    self.logger.record("eval/episode_perf/constraint_index", float(np.mean(constraint)))
                    print(f"[EVAL] Constraint Index: {np.mean(constraint):.4f}")
            
            # Detailed rewards log
            if self.current_eval_rewards:
                mean_eval_reward = np.mean(self.current_eval_rewards)
                std_eval_reward = np.std(self.current_eval_rewards)
                min_eval_reward = np.min(self.current_eval_rewards)
                max_eval_reward = np.max(self.current_eval_rewards)
                
                self.logger.record("eval/mean_reward_step_by_step", float(mean_eval_reward))
                self.logger.record("eval/std_reward_step_by_step", float(std_eval_reward))
                self.logger.record("eval/min_reward_step_by_step", float(min_eval_reward))
                self.logger.record("eval/max_reward_step_by_step", float(max_eval_reward))
                self.logger.record("eval/total_reward_steps", len(self.current_eval_rewards))
                
                print(f"[EVAL] Step-by-step Mean Reward: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
            
            # Log rewards per episode (if available)
            if self.episode_rewards:
                mean_episode_reward = np.mean(self.episode_rewards)
                std_episode_reward = np.std(self.episode_rewards)
                
                self.logger.record("eval/mean_episode_reward", float(mean_episode_reward))
                self.logger.record("eval/std_episode_reward", float(std_episode_reward))
                self.logger.record("eval/num_episodes_completed", len(self.episode_rewards))
                
                print(f"[EVAL] Episode Mean Reward: {mean_episode_reward:.2f} ± {std_episode_reward:.2f}")
            
            # Check if eval/mean_reward was set by parent class
            if hasattr(self, 'last_mean_reward'):
                print(f"[EVAL] Parent class mean reward: {self.last_mean_reward:.2f}")
            
            # Force logger dump
            self.logger.dump(step=self.num_timesteps)
        
        self.is_evaluating = False

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Override to also capture the parent class's mean_reward value"""
        super()._log_success_callback(locals_, globals_)
        
        # Capture the value calculated by parent class
        if hasattr(self, 'last_mean_reward'):
            self.logger.record("eval/mean_reward_parent_check", float(self.last_mean_reward))

class EpisodePerformanceLogger(BaseCallback):
    """
    Logs all `episode_perf/...` metrics returned by `info` at the end of an episode.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        # Normalization factors
        normalization_factors = {
            "episode_perf/energy_index": 0.0031,
            "episode_perf/time_index": 0.17,
            "episode_perf/constraint_index": 0.13
        }

        for info, done in zip(infos, dones):
            if done and isinstance(info, dict):
                for key in ["episode_perf/energy_index", 
                            "episode_perf/time_index", 
                            "episode_perf/constraint_index"]:
                    value = info.get(key)
                    if value is not None:
                        # Normalize value before logging
                        normalized_value = value / normalization_factors[key]
                        self.logger.record(key, normalized_value)
        return True

class ActionLoggerCallback(BaseCallback):
    """
    Callback to log action details in TensorBoard
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_buffer = deque(maxlen=10000)  # Buffer for the last N actions
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_count = 0
        self.Qp_min = 0
        self.Qp_max = 8
        self.Qp_range=(self.Qp_max-self.Qp_min)/2
        self.Qp_avg=(self.Qp_min+self.Qp_max)/2

        self.R_min = 6
        self.R_max = 14
        self.R_range=(self.R_max-self.R_min)/2
        self.R_avg=(self.R_min+self.R_max)/2
    
        
    def _on_step(self) -> bool:
        # Get current action
        if 'actions' in self.locals:
            actions = self.locals['actions']
            
            # If it's a numpy array, convert it
            if isinstance(actions, np.ndarray):
                actions = actions.flatten()
            elif torch.is_tensor(actions):
                actions = actions.cpu().numpy().flatten()
            
            # Add to buffer
            self.action_buffer.append(np.array(actions))
            self.episode_actions.extend(actions)
        
        # Rewards log
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if isinstance(rewards, np.ndarray):
                self.episode_rewards.extend(rewards.flatten())
        
        # Periodic log
        if self.n_calls % self.log_freq == 0:
            self._log_action_statistics()
        
        # End of episode log
        if 'dones' in self.locals and np.any(self.locals['dones']):
            self._log_episode_data()
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_count += 1

        # Debug entropy
        try:
            obs = torch.as_tensor(self.locals["obs"], device=self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs)
                ent = dist.entropy().mean().item()
                coef = self.model.ent_coef if isinstance(self.model.ent_coef, float) else self.model.ent_coef.item()
                self.logger.record("debug/policy_entropy", ent)
                self.logger.record("debug/entropy_loss_check", -ent * coef)
                self.logger.record("debug/ent_coef", coef)
            if self.n_calls % self.log_freq == 0:
                self.logger.dump(self.n_calls)
        except: pass
        return True
    
    def _log_action_statistics(self):
        """Log action statistics (supports multi-dimensional actions)"""
        if len(self.action_buffer) == 0:
            return

        actions_array = np.array(self.action_buffer)

        # Check correct dimensions
        if actions_array.ndim == 1:
            if self.verbose:
                print(f"[WARNING] action_buffer has 1D shape: {actions_array.shape}.")
            return  # Avoid crash: unable to distinguish dt and Th
        elif actions_array.shape[1] < 2:
            if self.verbose:
                print(f"[WARNING] Actions with insufficient dimension: shape = {actions_array.shape}")
            return

        # Extract components (assuming actions [dt, Th])
        Qp_array = (np.clip(actions_array[:,0], -1, 1) * self.Qp_range + self.Qp_avg)
        R_array  = (np.clip(actions_array[:,1], -1, 1) * self.R_range + self.R_avg)

        # Log for Qp
        self.logger.record("episode/real_actions_Qp_mean", np.mean(Qp_array))
        self.logger.record("episode/real_actions_Qp_std", np.std(Qp_array))
        self.logger.record("episode/real_actions_Qp_min", np.min(Qp_array))
        self.logger.record("episode/real_actions_Qp_max", np.max(Qp_array))

        # Log for R
        self.logger.record("episode/real_actions_R_mean", np.mean(R_array))
        self.logger.record("episode/real_actions_R_std", np.std(R_array))
        self.logger.record("episode/real_actions_R_min", np.min(R_array))
        self.logger.record("episode/real_actions_R_max", np.max(R_array))

        # Histogram image logging
        if self.n_calls % (self.log_freq * 10) == 0:
            self._log_action_histogram()

    
    def _log_episode_data(self):
        """Log data from the recently completed episode"""
        if len(self.episode_actions) > 0:
            episode_actions = np.array(self.episode_actions)
            
            # Episode statistics
            self.logger.record(f"episode/actions_mean", np.mean(episode_actions))
            self.logger.record(f"episode/actions_std", np.std(episode_actions))
            self.logger.record(f"episode/action_changes", 
                             np.sum(np.abs(np.diff(episode_actions))))
            
            # Action-reward correlation if available
            if len(self.episode_rewards) == len(self.episode_actions):
                correlation = np.corrcoef(episode_actions, self.episode_rewards)[0, 1]
                if not np.isnan(correlation):
                    self.logger.record(f"episode/action_reward_correlation", correlation)
    
    def _log_action_histogram(self):
        """Creates and logs action histogram"""
        try:
            actions_array = np.array(self.action_buffer)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if actions_array.ndim == 1:
                # 1D Actions
                ax.hist(actions_array, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title('Action Distribution')
                ax.set_xlabel('Action Value')
                ax.set_ylabel('Frequency')
            else:
                # Multi-dimensional actions - subplot per dimension
                n_dims = actions_array.shape[-1] if actions_array.ndim > 1 else 1
                fig, axes = plt.subplots(1, min(n_dims, 4), figsize=(15, 4))
                if n_dims == 1:
                    axes = [axes]
                
                actions_reshaped = actions_array.reshape(-1, n_dims)
                for i, ax in enumerate(axes[:n_dims]):
                    ax.hist(actions_reshaped[:, i], bins=30, alpha=0.7)
                    ax.set_title(f'Action Dim {i}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
            
            # Save as image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array for TensorBoard
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Log to TensorBoard
            self.logger.record("actions/histogram", 
                             img_array, 
                             exclude=("stdout", "log", "json", "csv"))
            
            plt.close(fig)
            buf.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error logging histogram: {e}")

class DetailedActionLogger(BaseCallback):
    """
    Detailed version for in-depth analysis
    """
    def __init__(self, log_freq=1000, save_raw_data=False, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_raw_data = save_raw_data
        self.raw_data = {'actions': [], 'rewards': [], 'observations': []} if save_raw_data else None
        
    def _on_step(self) -> bool:
        # Get all available data
        actions = self.locals.get('actions', None)
        rewards = self.locals.get('rewards', None)
        obs = self.locals.get('obs', None) if 'obs' in self.locals else None
        
        # Save raw data if requested
        if self.save_raw_data and actions is not None:
            self.raw_data['actions'].append(actions.copy() if isinstance(actions, np.ndarray) else actions)
            if rewards is not None:
                self.raw_data['rewards'].append(rewards.copy() if isinstance(rewards, np.ndarray) else rewards)
            if obs is not None:
                self.raw_data['observations'].append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        
        # Periodic detailed log
        if self.n_calls % self.log_freq == 0:
            self._detailed_logging()
        
        return True
    
    def _detailed_logging(self):
        """Detailed logging with advanced metrics"""
        if not self.save_raw_data or len(self.raw_data['actions']) < 100:
            return
        
        recent_actions = np.array(self.raw_data['actions'][-1000:])  # Last 1000 actions
        
        # Action entropy (measure of diversity)
        action_entropy = self._calculate_entropy(recent_actions)
        self.logger.record("actions/entropy", action_entropy)
        
        # Action stability (how much they change between consecutive steps)
        if len(recent_actions) > 1:
            action_stability = np.mean(np.abs(np.diff(recent_actions, axis=0)))
            self.logger.record("actions/stability", action_stability)
        
        # Action patterns
        if recent_actions.ndim > 1:
            for dim in range(recent_actions.shape[1]):
                dim_actions = recent_actions[:, dim]
                
                # Autocorrelation
                if len(dim_actions) > 1:
                    autocorr = np.corrcoef(dim_actions[:-1], dim_actions[1:])[0, 1]
                    if not np.isnan(autocorr):
                        self.logger.record(f"actions/dim_{dim}_autocorrelation", autocorr)
                
                # Dynamic range
                dynamic_range = np.max(dim_actions) - np.min(dim_actions)
                self.logger.record(f"actions/dim_{dim}_dynamic_range", dynamic_range)
    
    def _calculate_entropy(self, actions, bins=20):
        """
        Compute empirical entropy of recent actions.
        Works for both 1D and multi-dimensional continuous actions.
        """
        try:
            actions = np.asarray(actions)
            if actions.ndim == 1:
                hist, _ = np.histogram(actions, bins=bins, density=True)
            else:
                # Flatten over time, preserve dims
                hist, _ = np.histogramdd(actions, bins=bins, density=True)
                hist = hist.flatten()
            
            hist = hist[hist > 0]  # remove zeros to avoid log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            return float(entropy)
        except Exception as e:
            print(f"[Entropy calc error]: {e}")
            return 0.0

def kep2car(a, e, i, Ω, ω, theta, mu):
    cos, sin, sqrt = np.cos, np.sin, np.sqrt
    r = a * (1 - e**2) / (1 + e * cos(theta))
    h = sqrt(mu * a * (1 - e**2))
    
    # Position and velocity in perifocal frame
    r_pf = np.array([r * cos(theta), r * sin(theta), 0])
    v_pf = np.array([-sin(theta), e + cos(theta), 0]) * (mu / h)
    
    # Rotation matrix
    R = np.array([
        [cos(Ω)*cos(ω)-sin(Ω)*sin(ω)*cos(i), -cos(Ω)*sin(ω)-sin(Ω)*cos(ω)*cos(i), sin(Ω)*sin(i)],
        [sin(Ω)*cos(ω)+cos(Ω)*sin(ω)*cos(i), -sin(Ω)*sin(ω)+cos(Ω)*cos(ω)*cos(i), -cos(Ω)*sin(i)],
        [sin(ω)*sin(i),                      cos(ω)*sin(i),                      cos(i)]
    ])
    
    return np.concatenate((R @ r_pf, R @ v_pf))

def rk8_step(state, dt, RHS, mu, r_sun, J2, Drag, control):
    # RKF78 Butcher coefficients
    a = [0, 2/27, 1/9, 1/6, 5/12, 0.5, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]
    b = [
        [],
        [2/27],
        [1/36, 1/12],
        [1/24, 0, 1/8],
        [5/12, 0, -25/16, 25/16],
        [1/20, 0, 0, 1/4, 1/5],
        [-25/108, 0, 0, 125/108, -65/27, 125/54],
        [31/300, 0, 0, 0, 61/225, -2/9, 13/900],
        [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3],
        [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12],
        [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100,
         45/82, 45/164, 18/41],
        [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0],
        [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82,
         2193/4100, 51/82, 33/164, 12/41, 0, 1]
    ]
    c_high = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35,
              9/280, 9/280, 41/840, 0, 0]

    k = []
    for i in range(13):
        s = state.copy()
        for j in range(i):
            s += dt * b[i][j] * k[j]
        k.append(RHS(s, mu, r_sun, J2, Drag, control))

    # Combination for 8th order output
    y_next = state + dt * sum(c * ki for c, ki in zip(c_high, k))

    return y_next

def normalize(v):
    return v / np.linalg.norm(v)

from scipy.spatial.transform import Rotation as R

def random_chaser_state(state_target, theta_max_deg=9, r_min=100, r_max=300):
    Rot = eci2lvlh(state_target)
    d0=np.random.uniform(r_min, r_max)/1000
    theta, phi = np.random.uniform(0, np.radians(theta_max_deg)), np.random.uniform(0, 2*np.pi)
    r0=d0*np.array([np.sin(theta)*np.cos(phi), -np.cos(theta), np.sin(theta)*np.sin(phi)])
    v0 = np.random.normal(0, 0.1/2, 3) / 1000
    x_rel_lvlh = np.hstack((r0, v0 ))  # km, km/s
    x_rel_eci = np.hstack((Rot.T @ x_rel_lvlh[:3], Rot.T @ x_rel_lvlh[3:]))
    
    return state_target + x_rel_eci

def control_pert(u_ctrl_nom, std_theta_deg=3.0):

    u_ctrl_nom = np.array(u_ctrl_nom, dtype=float)
    u_dist = np.zeros(3)

    for i in range(3):
        ui = u_ctrl_nom[i]
        if ui == 0:
            continue
        # Intensity error (gaussian)
        scale_mag =  np.random.uniform( 0.96, 1.01)
        ui_pert = ui * scale_mag

        # Pointing error
        theta = np.radians(np.random.normal(loc=0.0, scale=std_theta_deg))
        axis = np.zeros(3)
        axis[(i+1)%3] = 1.0  # uses a fixed orthogonal axis for simplicity
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

        ei = np.zeros(3)
        ei[i] = 1.0
        u_pert_i = R @ ei * ui_pert

        u_dist += u_pert_i

    return u_dist

def sun_pos(day_of_year):
    # Approximated orbital angle (in radians)
    theta = 2 * np.pi * (day_of_year / 365.25)
    # Ecliptic obliquity (radians)
    epsilon = np.radians(23.44)
    # ECI coordinates (unit vector)
    x = np.cos(theta)
    y = np.cos(epsilon) * np.sin(theta)
    z = np.sin(epsilon) * np.sin(theta)
    return np.array([x, y, z])* 1.495978707e8

def eci2lvlh(x_target):
    """
    Returns the rotation matrix from ECI to Hill (LVLH) frame.
    Convention:
      - î: points from satellite to Earth center (radial inward)
      - ĵ: points along the velocity vector (prograde)
      - k̂: completes the right-handed frame, opposite to orbital angular momentum
    
    Input:
        x_target: target state in ECI [x, y, z, vx, vy, vz]
    
    Output:
        R: 3x3 rotation matrix such that v_hill = R.T @ v_eci
    """
    r, v = x_target[:3], x_target[3:]
    j = normalize(v)
    k = -normalize(np.cross(r, v))
    i = normalize(np.cross(j,k))

    return np.column_stack([i, j, k])