# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Custom Stable-Baselines3 callbacks for performance logging,
#              action distribution analysis, and TensorBoard integration
#              specifically tuned for MPC-Reinforced docking simulations.
# =============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import io
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class EvalCallbackWithPerfLogging(EvalCallback):
    """
    Custom EvalCallback that logs specific performance indices (Energy, Time, Constraints)
    during the evaluation phase.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations_infos = []
        self.current_eval_infos = []
        self.current_eval_rewards = []
        self.episode_rewards = []
        self.is_evaluating = False
    
    def _on_rollout_start(self) -> None:
        """Triggered at the start of the evaluation rollout."""
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
            
            # Step-by-step reward collection
            if rewards is not None:
                if isinstance(rewards, np.ndarray):
                    self.current_eval_rewards.extend(rewards.flatten())
                else:
                    self.current_eval_rewards.extend([rewards] if not isinstance(rewards, list) else rewards)
            
            # Collect info at episode completion
            for info, done in zip(infos, dones):
                if done and isinstance(info, dict):
                    self.current_eval_infos.append(info)
                    
                    if 'episode' in info and 'r' in info['episode']:
                        self.episode_rewards.append(info['episode']['r'])
                    elif 'episode_reward' in info:
                        self.episode_rewards.append(info['episode_reward'])
        
        return super()._on_step()
    
    def _on_rollout_end(self):
        """Triggered at the end of the evaluation rollout."""
        super()._on_rollout_end()
        
        if self.is_evaluating:
            print(f"[EVAL] Evaluation completed at step {self.num_timesteps}")
            
            if self.current_eval_infos:
                self.evaluations_infos.append(self.current_eval_infos.copy())
                energy, time_, constraint = [], [], []
                
                # Normalization and extraction
                for info in self.current_eval_infos:
                    if "episode_perf/energy_index" in info:
                        energy.append(info["episode_perf/energy_index"]/0.0031)
                    if "episode_perf/time_index" in info:
                        time_.append(info["episode_perf/time_index"]/0.17)
                    if "episode_perf/constraint_index" in info:
                        constraint.append(info["episode_perf/constraint_index"]/0.13)
                
                if energy:
                    self.logger.record("eval/episode_perf/energy_index", float(np.mean(energy)))
                if time_:
                    self.logger.record("eval/episode_perf/time_index", float(np.mean(time_)))
                if constraint:
                    self.logger.record("eval/episode_perf/constraint_index", float(np.mean(constraint)))
            
            # Statistics for step-by-step rewards
            if self.current_eval_rewards:
                self.logger.record("eval/mean_reward_step_by_step", float(np.mean(self.current_eval_rewards)))
                self.logger.record("eval/total_reward_steps", len(self.current_eval_rewards))
            
            if self.episode_rewards:
                self.logger.record("eval/mean_episode_reward", float(np.mean(self.episode_rewards)))
                self.logger.record("eval/num_episodes_completed", len(self.episode_rewards))
            
            self.logger.dump(step=self.num_timesteps)
        
        self.is_evaluating = False

class ActionLoggerCallback(BaseCallback):
    """
    Callback to log action statistics and distributions (histograms) to TensorBoard.
    Specifically tracks Qp and R parameters for the reinforced MPC.
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_buffer = deque(maxlen=10000)
        self.episode_actions = []
        self.episode_rewards = []
        
        # Scaling parameters for Qp and R
        self.Qp_min, self.Qp_max = 0, 8
        self.Qp_range = (self.Qp_max - self.Qp_min) / 2
        self.Qp_avg = (self.Qp_min + self.Qp_max) / 2

        self.R_min, self.R_max = 6, 14
        self.R_range = (self.R_max - self.R_min) / 2
        self.R_avg = (self.R_min + self.R_max) / 2
    
    def _on_step(self) -> bool:
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy().flatten()
            else:
                actions = actions.flatten()
            
            self.action_buffer.append(np.array(actions))
            self.episode_actions.extend(actions)
        
        if self.n_calls % self.log_freq == 0:
            self._log_action_statistics()
            # Policy Entropy Debug
            try:
                obs = torch.as_tensor(self.locals["obs"], device=self.model.device)
                with torch.no_grad():
                    dist = self.model.policy.get_distribution(obs)
                    ent = dist.entropy().mean().item()
                    self.logger.record("debug/policy_entropy", ent)
            except Exception:
                pass

        if 'dones' in self.locals and np.any(self.locals['dones']):
            self.episode_actions = []
            self.episode_rewards = []
            
        return True
    
    def _log_action_statistics(self):
        """Logs mean/std/min/max for Qp and R after de-normalization."""
        if len(self.action_buffer) == 0:
            return

        actions_array = np.array(self.action_buffer)
        if actions_array.ndim < 2 or actions_array.shape[1] < 2:
            return

        # De-normalize actions to real physical values
        Qp_array = (np.clip(actions_array[:,0], -1, 1) * self.Qp_range + self.Qp_avg)
        R_array  = (np.clip(actions_array[:,1], -1, 1) * self.R_range + self.R_avg)

        self.logger.record("episode/real_actions_Qp_mean", np.mean(Qp_array))
        self.logger.record("episode/real_actions_R_mean", np.mean(R_array))

        if self.n_calls % (self.log_freq * 10) == 0:
            self._log_action_histogram()

    def _log_action_histogram(self):
        """Creates and logs action distribution histograms as images to TensorBoard."""
        try:
            actions_array = np.array(self.action_buffer)
            n_dims = actions_array.shape[-1] if actions_array.ndim > 1 else 1
            fig, axes = plt.subplots(1, min(n_dims, 2), figsize=(12, 4))
            
            labels = ['Qp (Action 0)', 'R (Action 1)']
            for i in range(min(n_dims, 2)):
                axes[i].hist(actions_array[:, i], bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[i].set_title(f'Distribution: {labels[i]}')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            self.logger.record("actions/histogram", np.array(img), exclude=("stdout", "log", "json", "csv"))
            plt.close(fig)
        except Exception as e:
            if self.verbose > 0:
                print(f"Histogram logging error: {e}")


class DetailedActionLogger(BaseCallback):
    """
    More detailed version for in-depth analysis
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
        
        # Periodic detailed logging
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

def kep2car(a, e, i, Omega, omega, theta, mu):
    """
    Converts Keplerian elements to ECI Cartesian state.
    """
    cos, sin, sqrt = np.cos, np.sin, np.sqrt
    r_mag = a * (1 - e**2) / (1 + e * cos(theta))
    h = sqrt(mu * a * (1 - e**2))
    
    # Position and velocity in the Perifocal frame
    r_pf = np.array([r_mag * cos(theta), r_mag * sin(theta), 0])
    v_pf = np.array([-sin(theta), e + cos(theta), 0]) * (mu / h)
    
    # Rotation matrix from Perifocal to ECI
    Rot = np.array([
        [cos(Omega)*cos(omega)-sin(Omega)*sin(omega)*cos(i), -cos(Omega)*sin(omega)-sin(Omega)*cos(omega)*cos(i), sin(Omega)*sin(i)],
        [sin(Omega)*cos(omega)+cos(Omega)*sin(omega)*cos(i), -sin(Omega)*sin(omega)+cos(Omega)*cos(omega)*cos(i), -cos(Omega)*sin(i)],
        [sin(omega)*sin(i),                                  cos(omega)*sin(i),                                  cos(i)]
    ])
    
    return np.concatenate((Rot @ r_pf, Rot @ v_pf))

def rk8_step(state, dt, RHS, mu, r_sun, J2, Drag, control):
    """
    Performs a single integration step using the Runge-Kutta-Fehlberg 7(8) method.
    """
    # Butcher Tableau coefficients (RKF78)
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
        [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41],
        [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0],
        [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1]
    ]
    c_high = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]

    k = []
    for i in range(13):
        s = state.copy()
        for j in range(i):
            s += dt * b[i][j] * k[j]
        k.append(RHS(s, mu, r_sun, J2, Drag, control))

    # 8th order accurate combination
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
    v0 = np.random.normal(0, 0.1, 3) / 1000
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
        axis[(i+1)%3] = 1.0  
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
    """
    Calculates the approximate ECI position vector of the Sun.
    """
    # Approximate orbital angle
    theta = 2 * np.pi * (day_of_year / 365.25)
    # Ecliptic obliquity
    epsilon = np.radians(23.44)
    # ECI unit vector coordinates
    x = np.cos(theta)
    y = np.cos(epsilon) * np.sin(theta)
    z = np.sin(epsilon) * np.sin(theta)
    return np.array([x, y, z]) * 1.495978707e8  # 1 AU in km

def eci2lvlh(x_target):
    """
    Returns the rotation matrix from ECI to Hill (LVLH) frame.
    Convention:
      - i: Radial inward (satellite to Earth center)
      - j: Prograde (along velocity vector)
      - k: Normal (opposite to orbital angular momentum)
    """
    r, v = x_target[:3], x_target[3:]
    j = normalize(v)
    k = -normalize(np.cross(r, v))
    i = normalize(np.cross(j, k))

    return np.column_stack([i, j, k])