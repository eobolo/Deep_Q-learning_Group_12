import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
import cv2
import numpy as np

def setup_and_test_environment():
    """
    The aim of this environment setup code provided is to prepare a 
    functional Atari environment Breakout-v4 for our Deep
    Q-Learning assignment, ensuring it is compatible with Stable Baselines3
    for training a DQN agent and supports rendering for visualization
    in play.py. The setup verifies that the environment works, preprocesses observations
    for CnnPolicy, and confirms rendering capabilities, laying the foundation for your Task 1
    training with CnnPolicy and initial hyperparameter tuning.
    """
    
    # Step 1: Create the Atari environment
    env_name = "Breakout-v4"
    try:
        env = gym.make(env_name, render_mode="rgb_array")
        print(f"Successfully created environment: {env_name}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return None

    # Step 2: Apply AtariWrapper for Stable Baselines3 compatibility
    env = AtariWrapper(env)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Step 3: Test rendering
    try:
        env.reset()
        frame = env.render()
        if frame is not None:
            print(f"Rendering successful. Frame shape: {frame.shape}")
        else:
            print("Rendering failed: No frame returned")
    except Exception as e:
        print(f"Rendering error: {e}")

    # Step 4: Test a few steps to ensure environment stability
    done = False
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            env.reset()
    print("Environment step test completed successfully")

    env.close()
    return env_name

if __name__ == "__main__":
    print("Setting up Atari environment...")
    env_name = setup_and_test_environment()
    if env_name:
        print(f"Environment {env_name} is set up and ready for use!")
    else:
        print("Environment setup failed. Check dependencies and ROMs.")