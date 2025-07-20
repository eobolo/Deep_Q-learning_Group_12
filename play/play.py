import gymnasium as gym
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
from select_model import select_model_interactive
from gymnasium.wrappers import FlattenObservation
from test_env.test_cnn_breakout_env_setup import setup_and_test_environment


def load_and_play_env(model_path, num_episodes=5, render_delay=0.1):
    """
    Loads and renders environment for model to play
    """

    # Getting the environament name
    try:
        env_name = setup_and_test_environment()
    except Exception as e:
        print(f"Failed to setup environment")
        return
    
    env  = gym.make(env_name, render_mode="human")
    env = AtariWrapper(env, frame_skip=4, screen_size=84, terminal_on_life_loss=True)

    # Load the stored model from path
    try:
        model = DQN.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        env.close()
        return

    # Play episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done:
            # Use deterministic=True for greedy policy (highest Q-value)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Add small delay for better visualization
            time.sleep(render_delay)
            
            # Optional: Print action and reward info
            if episode_length % 100 == 0:
                print(f"Step {episode_length}: Action={action}, Reward={reward:.2f}, Total Reward={episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Average Reward per Step: {episode_reward/episode_length:.4f}")

    env.close()


def main():
    """Main function to run the player"""
    
    # Select model interactively
    model_path = select_model_interactive()
    
    if model_path is None:
        return

    try:
        num_episodes = int(input("Enter number of episodes to play (default 5): ") or 5)
    except ValueError:
        print("Invalid input, using default value of 5 episodes.")
        num_episodes = 5
    
    try:
        render_delay = float(input("Enter render delay in seconds (default 0.1): ") or 0.1)
    except ValueError:
        print("Invalid input, using default render delay of 0.1 seconds.")
        render_delay = 0.1
    
    print(f"\nLoading model from {model_path} and starting playback...")
    
    load_and_play_env(model_path, num_episodes, render_delay)

if __name__ == "__main__":
    main()
    
