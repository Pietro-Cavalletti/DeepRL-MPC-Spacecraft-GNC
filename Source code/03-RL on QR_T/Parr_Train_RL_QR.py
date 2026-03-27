# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Training script for Autonomous Docking using Recurrent PPO (SB3).
#              Features: Multiprocessing, Core Affinity, Environment Hashing,
#              and VecNormalize for stable training on high-dimensional states.
# =============================================================================

import os
import time
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import warnings
import multiprocessing

from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    CheckpointCallback, CallbackList, BaseCallback)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

from Utils_RL_QR import ActionLoggerCallback, DetailedActionLogger, EpisodePerformanceLogger, EvalCallbackWithPerfLogging
from Docking_env_QR import DockingEnv

# === Path Constants ===
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints_QR")
LOG_DIR = os.path.join(script_dir, "logs_QR")
MODEL_DIR = os.path.join(script_dir, "models_QR")
OUTPUT_DIR = os.path.join(script_dir, "training_results_QR")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Wrapper to load the pre-trained dt/Th model ===
class DummyDtThEnvForLoading(gym.Env):
    """Minimal environment used only for loading the pre-trained VecNormalize stats."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
    def reset(self, seed=None, options=None): pass
    def step(self, action): pass

class DtThPredictionWrapper(gym.Wrapper):
    """Wrapper that uses a pre-trained RecurrentPPO policy to dynamically predict dt and Th."""
    def __init__(self, env):
        super().__init__(env)
        print("[Wrapper] Initializing Dt/Th Prediction Wrapper...")

        # Ensure the path to the pre-trained model is correct
        pretrained_model_dir = os.path.join(script_dir, "pretrained_model")
        MODEL_PATH = os.path.join(pretrained_model_dir, "ppo_docking_dt_Th_2025-08-02_17-40-56_400000_steps.zip")
        VECNORM_PATH = os.path.join(pretrained_model_dir, "ppo_docking_dt_Th_2025-08-02_17-40-56_vecnormalize_400000_steps.pkl")

        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
            raise FileNotFoundError(f"Pre-trained model or VecNormalize not found in {pretrained_model_dir}")

        from stable_baselines3.common.vec_env import DummyVecEnv
        temp_env_for_loading = DummyVecEnv([lambda: DummyDtThEnvForLoading()])

        # Load normalization statistics
        self.dt_Th_vec_env = VecNormalize.load(VECNORM_PATH, temp_env_for_loading)
        self.dt_Th_vec_env.training = False
        self.dt_Th_vec_env.norm_reward = False

        # Load the policy
        self.dt_Th_model = RecurrentPPO.load(MODEL_PATH)
        self.dt_Th_policy = self.dt_Th_model.policy
        self.dt_Th_policy.eval()
        print("[Wrapper] Pre-trained model for Dt/Th loaded successfully.")

        # Parameters to decode the dt/Th model action
        self.dt_range, self.dt_avg = 1.7, 2.3
        self.Th_range, self.Th_avg = 13, 17
        self._lstm_state = None
        self._first_step_in_episode = True
        self.prev_action_t = np.zeros(2, dtype=np.float32)

    def _reset_lstm(self):
        """Initializes or resets the LSTM hidden state for the pre-trained policy."""
        hidden_size = self.dt_Th_policy.lstm_actor.hidden_size
        device = self.dt_Th_policy.device
        self._lstm_state = (torch.zeros((2, 1, hidden_size), device=device), 
                            torch.zeros((2, 1, hidden_size), device=device))
        self._first_step_in_episode = True

    def reset(self, **kwargs):
        self._reset_lstm()
        self.prev_action_t = np.zeros(2, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        unwrapped_env = self.env
        # Construct the observation for the dt/Th model using the current environment state
        obs_for_dt_Th_model = np.concatenate([
                unwrapped_env.err_lvlh / unwrapped_env.delta_x_max,
                self.prev_action_t.flatten(),
                [np.linalg.norm(unwrapped_env.u_last) / unwrapped_env.u_max]
        ]).astype(np.float32)
        
        # Normalize observation using pre-trained stats
        normalized_obs = self.dt_Th_vec_env.normalize_obs(obs_for_dt_Th_model)
        
        with torch.no_grad():
            action_tensor, self._lstm_state = self.dt_Th_policy.predict(
                normalized_obs,
                state=self._lstm_state,
                episode_start=np.array([self._first_step_in_episode]),
                deterministic=True
            )
            
        self._first_step_in_episode = False
        self.prev_action_t = np.clip(np.array(action_tensor, dtype=np.float32), -1, 1)
        
        # Decode action into environment-ready dt and Th values
        dt = action_tensor[0] * self.dt_range + self.dt_avg
        Th = action_tensor[1] * self.Th_range + self.Th_avg
        
        return self.env.step(action, dt, Th)

# Utility function for CPU core pinning (Affinity)
def set_core_affinity(env_idx, total_envs):
    """Binds the process to a specific CPU core to optimize performance."""
    try:
        core_count = multiprocessing.cpu_count()
        core_id = env_idx % core_count
        os.sched_setaffinity(0, {core_id})
        print(f"[INFO] Env {env_idx}: affinity set to core {core_id}")
    except Exception as e:
        print(f"[WARN] Env {env_idx}: failed to set core affinity: {e}")


def main():
    # === Hyperparameters & Setup ===
    N_ENVS = 12
    GAMMA = 0.99
    TOTAL_TIMESTEPS = 5_000_000
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    MODEL_BASE_NAME = f"ppo_docking_QR_{timestamp}"

    # Factory function to create environments with core pinning
    def make_env(env_idx):
        def _init():
            set_core_affinity(env_idx, N_ENVS)
            env = DockingEnv()
            env = DtThPredictionWrapper(env)
            env = Monitor(env)
            return env
        return _init

    # === Environment Creation ===
    # Training environment
    env_fns = [make_env(i) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    # Evaluation environment
    # Use different cores to avoid interference with training processes
    eval_env_fns = [make_env(i + N_ENVS) for i in range(20-N_ENVS)] 
    eval_env = SubprocVecEnv(eval_env_fns)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, gamma=GAMMA)

    # === Callbacks ===
    def create_callbacks():
        # Stop training if a high reward is achieved
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500_000, verbose=1)
        
        # Periodically evaluate the model
        eval_callback = EvalCallbackWithPerfLogging(
            eval_env,
            callback_on_new_best=callback_on_best,
            eval_freq=max(1, int(50000 // N_ENVS)),
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            n_eval_episodes=N_ENVS,
            verbose=1
        )
        
        # Save model checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(50000 // N_ENVS)),
            save_path=CHECKPOINT_DIR,
            name_prefix=MODEL_BASE_NAME,
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=2)
            
        # Logging utilities
        perf_logger = EpisodePerformanceLogger(verbose=1)
        action_logger = ActionLoggerCallback(log_freq=1000, verbose=1)
        detailed_logger = DetailedActionLogger(log_freq=5000, save_raw_data=True, verbose=1)
        
        return CallbackList([eval_callback, checkpoint_callback, perf_logger, action_logger, detailed_logger])

    callback = create_callbacks()

    # === Model Initialization ===
    print(f"\n🆕 Starting new training run: {MODEL_BASE_NAME}")
    print(f"[INFO] Training with {N_ENVS} parallel environments.")
    print(f"[INFO] Total timesteps: {TOTAL_TIMESTEPS}")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        batch_size=1024,
        n_steps=9216 // N_ENVS,
        n_epochs=15,
        gamma=GAMMA,
        gae_lambda=0.98,
        clip_range=0.2,
        max_grad_norm=0.3,
        ent_coef=5e-4,
        learning_rate=lambda p: 5e-5 * p, # Linear learning rate decay
        target_kl=0.02,
        policy_kwargs=dict(n_lstm_layers=2, log_std_init=np.log(1)),
        tensorboard_log=os.path.join(script_dir, "tensorboard_QR")
    )

    # === Training Execution ===
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    finally:
        # Always save the final model even if interrupted (Ctrl+C)
        final_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_final.zip")
        final_vecnorm_path = os.path.join(OUTPUT_DIR, f"{MODEL_BASE_NAME}_vecnormalize_final.pkl")
        
        print("\n[Main] Saving final model and normalization stats...")
        model.save(final_model_path)
        env.save(final_vecnorm_path)
        
        print(f"💾 Final model saved to: {final_model_path}")
        print(f"💾 Final VecNormalize saved to: {final_vecnorm_path}")


if __name__ == "__main__":
    # Set multiprocessing start method for cross-platform stability (using 'fork' for Linux/affinity compatibility)
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass # Start method can only be set once

    main()