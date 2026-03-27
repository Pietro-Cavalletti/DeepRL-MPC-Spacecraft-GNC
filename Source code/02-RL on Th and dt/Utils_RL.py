# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Utility functions for orbital mechanics, coordinate transforms,
#              numerical integration (RK8), and stochastic state generation.
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
        self.episode_rewards = []  # Per raccogliere reward per episodio
        self.is_evaluating = False
    
    def _on_rollout_start(self) -> None:
        """Chiamato all'inizio di ogni valutazione"""
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
            
            # Raccoglie tutte le rewards step-by-step
            if rewards is not None:
                if isinstance(rewards, np.ndarray):
                    self.current_eval_rewards.extend(rewards.flatten())
                else:
                    self.current_eval_rewards.extend([rewards] if not isinstance(rewards, list) else rewards)
            
            # Raccoglie info quando un episodio finisce
            for i, (info, done) in enumerate(zip(infos, dones)):
                if done and isinstance(info, dict):
                    self.current_eval_infos.append(info)
                    
                    # Se c'è episode reward nell'info, la raccoglie
                    if 'episode' in info and 'r' in info['episode']:
                        self.episode_rewards.append(info['episode']['r'])
                    elif 'episode_reward' in info:
                        self.episode_rewards.append(info['episode_reward'])
        
        return super()._on_step()
    
    def _on_rollout_end(self):
        """Chiamato alla fine di ogni valutazione"""
        # PRIMA chiama il metodo della classe padre per calcolare eval/mean_reward
        super()._on_rollout_end()
        
        if self.is_evaluating:
            print(f"[EVAL] Evaluation completed at step {self.num_timesteps}")
            
            # Log delle performance metrics personalizzate
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
            
            # Log dettagliato delle rewards
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
            
            # Log delle rewards per episodio (se disponibili)
            if self.episode_rewards:
                mean_episode_reward = np.mean(self.episode_rewards)
                std_episode_reward = np.std(self.episode_rewards)
                
                self.logger.record("eval/mean_episode_reward", float(mean_episode_reward))
                self.logger.record("eval/std_episode_reward", float(std_episode_reward))
                self.logger.record("eval/num_episodes_completed", len(self.episode_rewards))
                
                print(f"[EVAL] Episode Mean Reward: {mean_episode_reward:.2f} ± {std_episode_reward:.2f}")
            
            # Verifica se eval/mean_reward è stato settato dalla classe padre
            if hasattr(self, 'last_mean_reward'):
                print(f"[EVAL] Parent class mean reward: {self.last_mean_reward:.2f}")
            
            # Forza il dump del logger
            self.logger.dump(step=self.num_timesteps)
        
        self.is_evaluating = False

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Override per catturare anche il valore di mean_reward della classe padre"""
        super()._log_success_callback(locals_, globals_)
        
        # Cattura il valore che la classe padre ha calcolato
        if hasattr(self, 'last_mean_reward'):
            self.logger.record("eval/mean_reward_parent_check", float(self.last_mean_reward))

class EpisodePerformanceLogger(BaseCallback):
    """
    Logga tutte le metriche `episode_perf/...` restituite da `info` a fine episodio.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        # Fattori di normalizzazione
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
                        # Normalizza il valore prima del logging
                        normalized_value = value / normalization_factors[key]
                        self.logger.record(key, normalized_value)
        return True

class  ActionLoggerCallback(BaseCallback):
    """
    Callback per loggare dettagli sulle azioni in TensorBoard
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_buffer = deque(maxlen=10000)  # Buffer per le ultime N azioni
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_count = 0

        self.dt_min = 0.6
        self.dt_max = 4.0
        self.dt_range=(self.dt_max-self.dt_min)/2
        self.dt_avg=(self.dt_min+self.dt_max)/2

        self.Th_min = 4.0
        self.Th_max = 30.0
        self.Th_range=(self.Th_max-self.Th_min)/2
        self.Th_avg=(self.Th_min+self.Th_max)/2

        
    def _on_step(self) -> bool:
        # Ottieni l'azione corrente
        if 'actions' in self.locals:
            actions = self.locals['actions']
            
            # Se è un array numpy, convertilo
            if isinstance(actions, np.ndarray):
                actions = actions.flatten()
            elif torch.is_tensor(actions):
                actions = actions.cpu().numpy().flatten()
            
            # Aggiungi al buffer
            self.action_buffer.append(np.array(actions))
            self.episode_actions.extend(actions)
        
        # Log delle rewards
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if isinstance(rewards, np.ndarray):
                self.episode_rewards.extend(rewards.flatten())
        
        # Log periodico
        if self.n_calls % self.log_freq == 0:
            self._log_action_statistics()
        
        # Log a fine episodio
        if 'dones' in self.locals and np.any(self.locals['dones']):
            self._log_episode_data()
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_count += 1

        #Debug entropy
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
        """Log statistiche delle azioni (supporta azioni multidimensionali)"""
        if len(self.action_buffer) == 0:
            return

        actions_array = np.array(self.action_buffer)

        # Verifica dimensioni corrette
        if actions_array.ndim == 1:
            if self.verbose:
                print(f"[WARNING] action_buffer ha shape 1D: {actions_array.shape}.")
            return  # Evita crash: non sappiamo distinguere dt e Th
        elif actions_array.shape[1] < 2:
            if self.verbose:
                print(f"[WARNING] Azioni con dimensione insufficiente: shape = {actions_array.shape}")
            return

        # Estrai le due componenti (supponendo azioni [dt, Th])
        dt_array = np.clip(actions_array[:,0], -1, 1) * self.dt_range + self.dt_avg
        Th_array = np.clip(actions_array[:,1], -1, 1) * self.Th_range + self.Th_avg

        # Log per dt
        self.logger.record("episode/real_actions_dt_mean", np.mean(dt_array))
        self.logger.record("episode/real_actions_dt_std", np.std(dt_array))
        self.logger.record("episode/real_actions_dt_min", np.min(dt_array))
        self.logger.record("episode/real_actions_dt_max", np.max(dt_array))

        # Log per Th
        self.logger.record("episode/real_actions_Th_mean", np.mean(Th_array))
        self.logger.record("episode/real_actions_Th_std", np.std(Th_array))
        self.logger.record("episode/real_actions_Th_min", np.min(Th_array))
        self.logger.record("episode/real_actions_Th_max", np.max(Th_array))

        # Logging immagine istogramma
        if self.n_calls % (self.log_freq * 10) == 0:
            self._log_action_histogram()

    
    def _log_episode_data(self):
        """Log dati dell'episodio appena completato"""
        if len(self.episode_actions) > 0:
            episode_actions = np.array(self.episode_actions)
            
            # Statistiche episodio
            self.logger.record(f"episode/actions_mean", np.mean(episode_actions))
            self.logger.record(f"episode/actions_std", np.std(episode_actions))
            self.logger.record(f"episode/action_changes", 
                             np.sum(np.abs(np.diff(episode_actions))))
            
            # Correlazione azione-reward se disponibile
            if len(self.episode_rewards) == len(self.episode_actions):
                correlation = np.corrcoef(episode_actions, self.episode_rewards)[0, 1]
                if not np.isnan(correlation):
                    self.logger.record(f"episode/action_reward_correlation", correlation)
    
    def _log_action_histogram(self):
        """Crea e logga istogramma delle azioni"""
        try:
            actions_array = np.array(self.action_buffer)
            
            # Crea figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if actions_array.ndim == 1:
                # Azioni 1D
                ax.hist(actions_array, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title('Distribuzione Azioni')
                ax.set_xlabel('Valore Azione')
                ax.set_ylabel('Frequenza')
            else:
                # Azioni multidimensionali - subplot per ogni dimensione
                n_dims = actions_array.shape[-1] if actions_array.ndim > 1 else 1
                fig, axes = plt.subplots(1, min(n_dims, 4), figsize=(15, 4))
                if n_dims == 1:
                    axes = [axes]
                
                actions_reshaped = actions_array.reshape(-1, n_dims)
                for i, ax in enumerate(axes[:n_dims]):
                    ax.hist(actions_reshaped[:, i], bins=30, alpha=0.7)
                    ax.set_title(f'Azione Dim {i}')
                    ax.set_xlabel('Valore')
                    ax.set_ylabel('Frequenza')
            
            # Salva come immagine in memoria
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Converti in array numpy per TensorBoard
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Log su TensorBoard
            self.logger.record("actions/histogram", 
                             img_array, 
                             exclude=("stdout", "log", "json", "csv"))
            
            plt.close(fig)
            buf.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Errore nel logging dell'istogramma: {e}")


class DetailedActionLogger(BaseCallback):
    """
    Versione più dettagliata per analisi approfondite
    """
    def __init__(self, log_freq=1000, save_raw_data=False, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_raw_data = save_raw_data
        self.raw_data = {'actions': [], 'rewards': [], 'observations': []} if save_raw_data else None
        
    def _on_step(self) -> bool:
        # Ottieni tutti i dati disponibili
        actions = self.locals.get('actions', None)
        rewards = self.locals.get('rewards', None)
        obs = self.locals.get('obs', None) if 'obs' in self.locals else None
        
        # Salva dati raw se richiesto
        if self.save_raw_data and actions is not None:
            self.raw_data['actions'].append(actions.copy() if isinstance(actions, np.ndarray) else actions)
            if rewards is not None:
                self.raw_data['rewards'].append(rewards.copy() if isinstance(rewards, np.ndarray) else rewards)
            if obs is not None:
                self.raw_data['observations'].append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        
        # Log dettagliato periodico
        if self.n_calls % self.log_freq == 0:
            self._detailed_logging()
        
        return True
    
    def _detailed_logging(self):
        """Logging dettagliato con metriche avanzate"""
        if not self.save_raw_data or len(self.raw_data['actions']) < 100:
            return
        
        recent_actions = np.array(self.raw_data['actions'][-1000:])  # Ultime 1000 azioni
        
        # Entropia delle azioni (misura della diversità)
        action_entropy = self._calculate_entropy(recent_actions)
        self.logger.record("actions/entropy", action_entropy)
        
        # Stabilità delle azioni (quanto cambiano tra step consecutivi)
        if len(recent_actions) > 1:
            action_stability = np.mean(np.abs(np.diff(recent_actions, axis=0)))
            self.logger.record("actions/stability", action_stability)
        
        # Pattern nelle azioni
        if recent_actions.ndim > 1:
            for dim in range(recent_actions.shape[1]):
                dim_actions = recent_actions[:, dim]
                
                # Autocorrelazione
                if len(dim_actions) > 1:
                    autocorr = np.corrcoef(dim_actions[:-1], dim_actions[1:])[0, 1]
                    if not np.isnan(autocorr):
                        self.logger.record(f"actions/dim_{dim}_autocorrelation", autocorr)
                
                # Range dinamico
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
    
    # Posizione e velocità nel sistema perifocale
    r_pf = np.array([r * cos(theta), r * sin(theta), 0])
    v_pf = np.array([-sin(theta), e + cos(theta), 0]) * (mu / h)
    
    # Matrice di rotazione
    R = np.array([
        [cos(Ω)*cos(ω)-sin(Ω)*sin(ω)*cos(i), -cos(Ω)*sin(ω)-sin(Ω)*cos(ω)*cos(i), sin(Ω)*sin(i)],
        [sin(Ω)*cos(ω)+cos(Ω)*sin(ω)*cos(i), -sin(Ω)*sin(ω)+cos(Ω)*cos(ω)*cos(i), -cos(Ω)*sin(i)],
        [sin(ω)*sin(i),                      cos(ω)*sin(i),                      cos(i)]
    ])
    
    return np.concatenate((R @ r_pf, R @ v_pf))

def rk8_step(state, dt, RHS, mu, r_sun, J2, Drag, control):
    # Coefficienti di Butcher RKF78
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

    # Combinazione per output di ordine 8
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
        axis[(i+1)%3] = 1.0  # usa un asse ortogonale fisso per semplicità
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
    # Angolo orbitale approssimato (in radianti)
    theta = 2 * np.pi * (day_of_year / 365.25)
    # Obliquità dell'eclittica (radianti)
    epsilon = np.radians(23.44)
    # Coordinate ECI (vettore unitario)
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