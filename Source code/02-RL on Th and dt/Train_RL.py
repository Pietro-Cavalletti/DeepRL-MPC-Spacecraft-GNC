# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Training script for Satellite Docking using Recurrent PPO.
#              This version focuses on deep high-capacity policy networks
#              and robust environment consistency checks via MD5 hashing.
# =============================================================================

import os
import hashlib
from datetime import datetime
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    CheckpointCallback, CallbackList, BaseCallback
)
from Docking_env import DockingEnv
from Utils_RL import ActionLoggerCallback, DetailedActionLogger

# === CUSTOM CALLBACKS ===

class RewardLoggerCallback(BaseCallback): 
    """Custom callback to log mean rewards to a text file every 10k steps."""
    def __init__(self, log_file="reward_log.txt", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            if "rewards" in self.locals:
                mean_reward = np.mean(self.locals["rewards"])
                with open(self.log_file, "a") as f:
                    f.write(f"{self.num_timesteps},{mean_reward:.2f}\n")
        return True

# === UTILITIES ===

def compute_file_hash(filepath):
    """Computes MD5 hash to ensure the environment logic hasn't changed."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def make_env():
    """Environment factory wrapped with Monitor for statistics tracking."""
    def _init():
        env = DockingEnv()
        env = Monitor(env)
        return env
    return _init

# === PATHS & CONFIGURATION ===

MODEL_BASE_NAME = "ppo_docking_dt_Th"
model_path = f"checkpoints/{MODEL_BASE_NAME}.zip"
vecnormalize_path = f"checkpoints/{MODEL_BASE_NAME}_vecnormalize.pkl"
reward_log_path = f"checkpoints/{MODEL_BASE_NAME}_reward_log.txt"
env_hash_path = "env_hash.txt"

# Get absolute path for the environment file to check for changes
env_code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docking_env.py")
current_env_hash = compute_file_hash(env_code_path)

# Set working directory and ensure checkpoint folder exists
os.makedirs("checkpoints", exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === ENVIRONMENT INITIALIZATION ===

# Using DummyVecEnv for single-process training (change to SubprocVecEnv for parallel)
base_env = DummyVecEnv([make_env()])
# Initial VecNormalize setup
env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_reward=1000.0)

# Check if training can be resumed based on file presence and environment integrity
can_resume = (
    os.path.exists(model_path) and
    os.path.exists(vecnormalize_path) and
    os.path.exists(env_hash_path) and
    open(env_hash_path).read() == current_env_hash
)

# === EVALUATION ENVIRONMENT SETUP ===

# Create a separate environment for evaluation to keep normalization stats pure
if os.path.exists(vecnormalize_path):
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
else:
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

eval_env.training = False     # Do not update stats during evaluation
eval_env.norm_reward = False  # Use raw rewards for evaluation clarity

# === CALLBACK CHAIN SETUP ===

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_name_prefix = f"{MODEL_BASE_NAME}_{now}"

def create_callbacks():
    """Initializes the callback list for monitoring and saving progress."""
    # Training will stop if reward reaches this threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500_000, verbose=1)
    
    # Standard evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=25000,
        best_model_save_path="./models/",
        log_path="./logs/",
        n_eval_episodes=15,
        verbose=1,
        deterministic=True
    )
    
    # Checkpoint callback to save progress periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path="./checkpoints/",
        name_prefix=checkpoint_name_prefix, 
        save_replay_buffer=True,
        save_vecnormalize=True, 
        verbose=2
    )
    
    # Specialized loggers for monitoring MPC actions (dt and Th)
    action_logger = ActionLoggerCallback(log_freq=1000, verbose=1)
    detailed_logger = DetailedActionLogger(log_freq=5000, save_raw_data=True, verbose=1)
    
    return CallbackList([
        eval_callback, 
        checkpoint_callback, 
        action_logger,
        detailed_logger
    ])

callback_list = create_callbacks()
reward_logger = RewardLoggerCallback(log_file=reward_log_path)

# === MODEL SETUP & HYPERPARAMETERS ===

if can_resume:
    print("✅ Environment unchanged. Resuming training from existing checkpoint...")
    env = VecNormalize.load(vecnormalize_path, base_env)
    env.training = True
    env.norm_reward = True
    model = RecurrentPPO.load(model_path, env=env)
else:
    print("🆕 Starting from scratch (New environment hash or missing model).")
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env,
        n_steps=4096,            # Steps per update (large for stable docking trajectories)
        batch_size=512,          # Mini-batch size
        n_epochs=15,             # Number of optimization epochs per update
        learning_rate=5e-5,      # Stable learning rate for complex physics
        gamma=0.99,              # High discount factor for long-term planning
        gae_lambda=0.98,  
        clip_range=0.1,          # Conservative clipping for policy stability
        tensorboard_log="./tensorboard/",
        verbose=1,
        ent_coef=1e-5,           # Small entropy to prevent premature convergence
        policy_kwargs=dict(
            # Deeper Actor (pi) and Critic (vf) architectures
            net_arch=dict(pi=[256, 256, 128], vf=[128, 128]), 
            log_std_init=-0.7     # Start with slightly lower initial exploration noise
        )
    )

# === TRAINING EXECUTION ===

try:
    model.learn(total_timesteps=1_000_000, callback=callback_list)
except KeyboardInterrupt:
    print("⚠️ Training interrupted. Saving current progress...")

# Final saves
model.save(model_path)
env.save(vecnormalize_path)

# Update the hash file to track this specific version of the physics environment
with open(env_hash_path, "w") as f:
    f.write(current_env_hash)

print(f"💾 Model saved to: {model_path}")
print(f"💾 VecNormalize stats saved to: {vecnormalize_path}")
print(f"📊 Reward tracking saved to: {reward_log_path}")