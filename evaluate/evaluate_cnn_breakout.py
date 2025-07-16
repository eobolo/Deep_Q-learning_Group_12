import pandas as pd
import matplotlib.pyplot as plt

training_log_file = "training_log_4.csv" # edit based on your log file name
log = pd.read_csv("logs/{}".format(training_log_file))


plt.figure(figsize=(10, 5))
plt.plot(log['timestep'], log['mean_reward_last_100'], label='Mean Reward (Last 100 Episodes) of {}'.format(training_log_file))
plt.xlabel('Timestep')
plt.ylabel('Mean Reward')
plt.title('Training Reward Trend')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(log['timestep'], log['mean_episode_length_last_100'], label='Mean Episode Length (Last 100 Episodes) of {}'.format(training_log_file))
plt.xlabel('Timestep')
plt.ylabel('Mean Episode Length')
plt.title('Training Episode Length Trend')
plt.legend()
plt.show()
