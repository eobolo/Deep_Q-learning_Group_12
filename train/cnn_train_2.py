import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from test_env.test_cnn_breakout_env_setup import setup_and_test_environment
import os
import pandas as pd

# Custom callback to log reward and episode length
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_file: str, verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.rewards = []
        self.episode_lengths = []
        self.episode_rewards = []
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
            # Log to file
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
# but before envrionment setup first test it out
try:
    env_name = setup_and_test_environment()
except Exception as e:
    pass
else: 
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariWrapper(env, frame_skip=4, screen_size=84, terminal_on_life_loss=True)

# Adjust hyperparameters
hyperparams = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'batch_size': 32,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    'exploration_fraction': 0.4
}

# Initialize DQN with CnnPolicy
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=hyperparams['learning_rate'],
    gamma=hyperparams['gamma'],
    batch_size=hyperparams['batch_size'],
    exploration_initial_eps=hyperparams['exploration_initial_eps'],
    exploration_final_eps=hyperparams['exploration_final_eps'],
    exploration_fraction=hyperparams['exploration_fraction'],
    buffer_size=10000,
    learning_starts=10000,
    target_update_interval=1000,
    train_freq=4,
    verbose=1
)

# Train the model
experiment_value = (input("Enter an experiment name (anything integers or strings): "))
log_file = "training_log_{0}.csv".format(experiment_value)
callback = TrainingLoggerCallback(check_freq=1000, log_file=log_file)
model.learn(total_timesteps=50000, callback=callback)

# Save the model
model.save("cnn_dqn_{0}_model".format(experiment_value))

# Close the environment
env.close()