import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FlattenObservation
import os
import pandas as pd
from test_env.test_mlp_breakout_env import setup_and_test_environment  # Import the modified setup

# Custom callback to log reward and episode length
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_file: str, verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards']
        self.current_episode_length += 1
        if self.locals['dones']:
            self.rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        if self.n_calls % self.check_freq == 0:
            df = pd.DataFrame({
                'timestep': [self.n_calls],
                'mean_reward_last_100': [np.mean(self.rewards[-100:]) if self.rewards else 0],
                'mean_episode_length_last_100': [np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0]
            })
            if os.path.exists(self.log_file):
                df.to_csv(self.log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.log_file, mode='w', header=True, index=False)
        return True

# Set up the environment
try:
    env_name = setup_and_test_environment()
except Exception as e:
    print(f"Environment setup failed: {e}")
    env = None
else:
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariWrapper(env, frame_skip=4, screen_size=84, terminal_on_life_loss=True)
    env = FlattenObservation(env)  # Add flattening for MLPPolicy

# Adjust hyperparameters
hyperparams = {
    'learning_rate': 4e-3,  # Adjusted for MLPPolicy
    'gamma': 0.99,
    'batch_size': 32,  # Increased for stability
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    'exploration_fraction': 0.5  # Adjusted for faster convergence
}

# Initialize DQN with MlpPolicy
if env is not None:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams['learning_rate'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        exploration_initial_eps=hyperparams['exploration_initial_eps'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        exploration_fraction=hyperparams['exploration_fraction'],
        buffer_size=100000,
        learning_starts=10000,
        target_update_interval=1000,
        train_freq=4,
        verbose=1
    )

    # Train the model
    experiment_value = input("Enter an experiment name (anything integers or strings): ")
    log_file = f"training_log_mlp_{experiment_value}.csv"
    callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file)
    model.learn(total_timesteps=50000, callback=callback)

    # Save the model
    model.save(f"mlp_dqn_{experiment_value}_model")

    # Close the environment
    env.close()