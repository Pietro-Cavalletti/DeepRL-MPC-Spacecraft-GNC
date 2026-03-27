# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Training script for Autonomous Docking using Recurrent PPO (SB3).
#              Features: Multiprocessing, Core Affinity, Environment Hashing,
#              and VecNormalize for stable training on high-dimensional states.
# =============================================================================

import os
import hashlib
from datetime import datetime
import numpy as np
import multiprocessing
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    CheckpointCallback, CallbackList, BaseCallback)
from Docking_env import DockingEnv
from Utils_RL import (ActionLoggerCallback, DetailedActionLogger, 
                      EpisodePerformanceLogger, EvalCallbackWithPerfLogging)

# --- CONFIGURATION ---
N_ENVS = 16         # Number of parallel CPU environments
GAMMA = 0.995       # Discount factor (high for long-horizon docking tasks)

def set_core_affinity(env_idx, total_envs):
    """
    Pins each environment to a specific CPU core to prevent overhead
    from OS context switching during heavy numerical integration.
    """
    try:
        core_count = multiprocessing.cpu_count()
        core_id = env_idx % core_count
        os.sched_setaffinity(0, {core_id})
        # print(f"[INFO] Env {env_idx}: set affinity to core {core_id}")
    except Exception as e:
        print(f"[WARN] Env {env_idx}: failed to set core affinity: {e}")

class RewardLoggerCallback(BaseCallback):
    """Logs mean reward to a text file for external monitoring."""
    def __init__(self, log_file="reward_log.txt", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file

    def _on_step(self) -> bool:
        if self.n_calls % 100000 == 0:
            if "rewards" in self.locals:
                mean_reward = np.mean(self.locals["rewards"])
                with open(self.log_file, "a") as f:
                    f.write(f"{self.num_timesteps},{mean_reward:.2f}\n")
        return True

def compute_file_hash(filepath):
    """Generates an MD5 hash of the environment file to detect logic changes."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def make_env(env_idx):
    """Helper function to initialize parallel environments."""
    def _init():
        set_core_affinity(env_idx, N_ENVS)
        env = DockingEnv()
        return Monitor(env) # Monitor tracks episode statistics (reward, length)
    return _init

def main():
    # --- FILENAME & PATH MANAGEMENT ---
    model_name = "ppo_docking_dt_Th"
    model_path = f"checkpoints/{model_name}.zip"
    vecnormalize_path = f"checkpoints/{model_name}_vecnormalize.pkl"
    reward_log_path = f"checkpoints/{model_name}_reward_log.txt"
    env_hash_path = "env_hash.txt"
    env_code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docking_env.py")
    
    current_env_hash = compute_file_hash(env_code_path)
    os.makedirs("checkpoints", exist_ok=True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- ENVIRONMENT SETUP ---
    # SubprocVecEnv runs each env in a separate process
    env_fns = [make_env(i) for i in range(N_ENVS)]
    base_env = SubprocVecEnv(env_fns)
    
    # VecNormalize is CRITICAL for PPO when observations/rewards have different scales
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    # Logic to check if we can resume training:
    # Requires: saved model, saved normalizer, and UNCHANGED environment code.
    can_resume = (
        os.path.exists(model_path) and
        os.path.exists(vecnormalize_path) and
        os.path.exists(env_hash_path) and
        open(env_hash_path).read() == current_env_hash
    )

    # --- EVALUATION ENVIRONMENT ---
    eval_env_fns = [make_env(i) for i in range(N_ENVS)]
    eval_env = SubprocVecEnv(eval_env_fns)
    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=GAMMA)
    
    eval_env.training = False # Important: don't update normalization stats during eval
    eval_env.norm_reward = False

    # Warmup: allows VecNormalize to get initial statistics before training starts
    if not can_resume:
        print("[INFO] Warmup: sampling random actions for VecNormalize statistics...")
        env.reset()
        for _ in range(100):
            actions = [env.action_space.sample() for _ in range(N_ENVS)]
            env.step(actions)

    # --- CALLBACKS CONFIGURATION ---
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name_prefix = f"{model_name}_{now}"

    def create_callbacks():
        # Stop training if agent is consistently successful
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500_000, verbose=1)
        
        # Periodic evaluation of the agent
        eval_callback = EvalCallbackWithPerfLogging(
            eval_env,
            callback_on_new_best=callback_on_best,
            eval_freq=max(1, int(50000 // N_ENVS)),
            best_model_save_path="./models/",
            log_path="./logs/",
            n_eval_episodes=N_ENVS,
            deterministic=True)

        # Regular model saves to prevent data loss
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(50000 // N_ENVS)),
            save_path="./checkpoints/",
            name_prefix=checkpoint_name_prefix,
            save_replay_buffer=True,
            save_vecnormalize=True)

        perf_logger = EpisodePerformanceLogger(verbose=1)
        action_logger = ActionLoggerCallback(log_freq=2000, verbose=1)
        detailed_logger = DetailedActionLogger(log_freq=5000, save_raw_data=True, verbose=1)

        return CallbackList([eval_callback, checkpoint_callback, perf_logger, action_logger, detailed_logger])

    callback = create_callbacks()

    # --- MODEL INITIALIZATION / LOADING ---
    if can_resume:
        print("✅ Environment unchanged. Resuming training from last checkpoint.")
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = True
        env.norm_reward = True
        model = RecurrentPPO.load(model_path, env=env)
    else:
        print("🆕 Starting fresh training (Environment changed or no model found).")
        # RecurrentPPO uses an LSTM layer to handle temporal dependencies (memory)
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            n_steps=512,            # Experience collected per environment before update
            batch_size=1024,        # Mini-batch size for SGD
            n_epochs=10,            # Number of passes over the collected experience
            # Learning rate scheduler: decays over time for better convergence
            learning_rate=lambda p: 1e-3 * p**2, 
            gamma=GAMMA,
            gae_lambda=0.98,
            clip_range=0.2,
            tensorboard_log="./tensorboard/",
            ent_coef=1e-4,          # Entropy coefficient (encourages exploration)
            target_kl=0.02,         # KL Divergence limit for policy updates
            policy_kwargs=dict(n_lstm_layers=2), # Depth of the memory network
            max_grad_norm=0.3,       # Gradient clipping to prevent exploding gradients
            verbose=1)

    # --- TRAINING LOOP ---
    try:
        model.learn(total_timesteps=2_000_000, callback=callback)
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user. Saving progress...")

    # --- FINAL SAVE ---
    model.save(model_path)
    env.save(vecnormalize_path)
    with open(env_hash_path, "w") as f:
        f.write(current_env_hash)

    print(f"💾 Training Complete. Artifacts saved in /checkpoints/")

if __name__ == "__main__":
    # "fork" is generally faster on Linux/Unix systems for SubprocVecEnv
    multiprocessing.set_start_method("fork", force=True)
    main()