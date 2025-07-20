# Deep Q Learning - Training and Playing an RL Agent

## Project Overview

This project trains and evaluates a Deep Q-Network (DQN) agent to play an Atari game using Stable Baselines3 and Gymnasium. The agent is trained on the `Breakout-v4` environment and evaluated using a separate script. The project includes hyperparameter tuning and comparison of different policy architectures.

## Folder Structure

| Folder             | Purpose                                 | What to put here                              |
|--------------------|-----------------------------------------|-----------------------------------------------|
| deepQ/             | Python virtual environment              | Nothing manually                              |
| evaluate/          | Evaluation/analysis scripts             | Evaluation scripts, analysis code             |
| experiment_table/  | Experiment tracking (tables, logs)      | Spreadsheets, experiment logs                 |
| logs/              | Training logs                           | Training CSV log files                        |
| models/            | Saved models                            | Model files (.zip)                |
| test_env/          | Env setup/testing scripts               | Env setup, test, or utility scripts           |
| train/             | Training scripts                        | Model training scripts                        |
| training_graphs/   | Training plots/graphs                   | Output plots/figures from evaluation scripts  |

**Note:** The hyperparameter tuning table is also available as an Excel spreadsheet in [`experiment_table/hyperparameter_observation.xlsx`](experiment_table/hyperparameter_observation.xlsx).

## Assignment Instructions

- **Environment Selection:** Atari environment from Gymnasium (`Breakout-v4` used here).
- **Training Script:** `train/cnn_train.py`, `train/cnn_train_2.py`, `train/cnn_train_3.py`, `train/mlp_train.py` and so on help train the agent and save the policy network.
- **Playing Script:** Use the script `play.py` in `play/` which helps load the trained model and visualize agent performance.
- **Hyperparameter Tuning:** Multiple configurations tested and documented below.
- **Submission:** Repository contains all scripts, models, logs, and documentation.

## Hyperparameter Tuning Tables

### CNN

| DQN Policy | Environment | Gamma | Batch Size | Eps Initial | Eps Final | Learning Rate | Time Steps | Eps Decay | Buffer Size | Learning Starts | Target Update Interval | Observation |
|------------|-------------|-------|------------|--------------|------------|----------------|-------------|------------|---------------|------------------|--------------------------|-------------|
| CNN        | breakout-v4 | 0.85  | 32         | 1            | 0.02       | 0.0001         | 50000       | 0.1        | 10000         | 10000            | 1000                     | I believe the model is experiencing underfitting, as the mean reward never exceeds 0.7 and remains well below the expected 3–5 range, suggesting it hasn’t fully captured the complexities of Breakout-v4. A spike at 20k timesteps indicates a promising short-term strategy, likely due to low gamma. More training or better exploration might help. |
| CNN        | breakout-v4 | 0.99  | 32         | 1            | 0.02       | 0.0001         | 50000       | 0.1        | 10000         | 10000            | 1000                     | Increasing gamma from 0.85 to 0.99 led to better mean rewards (0.2 to 0.8). The agent showed a steady upward trend, indicating improved long-term planning. Episode lengths also improved gradually (10–15 timesteps), but further training could enhance results. |
| CNN        | breakout-v4 | 0.99  | 32         | 1            | 0.05       | 0.0001         | 50000       | 0.4        | 10000         | 10000            | 1000                     | With gamma=0.99 and slower epsilon decay (0.4), the model showed consistent reward growth from 0.2 to 1.0 without spikes. Episode length rose from 2 to 18, suggesting improved survivability. The agent likely benefits from extended exploration and could improve further with more training. |
| CNN        | breakout-v4 | 1     | 32         | 1            | 0.02       | 0.0001         | 1000000     | 0.1        | 100000        | 50000            | 1000                     | Over 1 million timesteps, the model achieved gradual reward improvement (0.5 to 2.5), showing strong learning behavior due to high gamma and long training. Episode lengths increased to 35, showing better survivability, though still far from optimal. |
| CNN | breakout-v4 | 1 | 32 | 1 | 0.02 | 0.0001 | 1,000,000 | 0.15 | 50,000  | 25,000 | 2,000 | Reducing buffer_size to 50,000, learning_starts to 25,000, and increasing target_update_interval to 2,000—the training shows a familiar three-phase pattern: initial struggle (0-150k timesteps) with rewards around 0.3-0.4, rapid improvement (150k-400k timesteps) climbing to ~2.2, and plateau (400k-1M timesteps) stabilizing around 2.0-2.3. The episode length mirrors this trend, growing from ~10 to 30-35 timesteps. Compared to the previous run, the performance appears nearly identical, suggesting these parameter adjustments had minimal impact on final learning outcomes, with the agent still achieving modest but consistent brick-breaking ability rather than expert-level mastery.|
| CNN | breakout-v4 | 1 | 32 | 1 | 0.03 | 0.0001 | 1,000,000 | 0.14 | 100,000 | 25,000 | 2,000 | Increasing eps_final to 0.03 and adjusting eps_decay to 0.14—the training shows notably different behavior compared to previous runs. The reward plateaus much lower, stabilizing around 1.0-1.3 instead of the previous 2.0-2.3 range, suggesting the increased final exploration (0.03 vs 0.02) may be preventing full exploitation of learned strategies. The episode length similarly caps around 18-22 timesteps compared to the previous 30-35, indicating reduced survival ability. The learning curve shows initial growth until ~500k timesteps but then maintains high variability without clear improvement, suggesting the agent struggles to converge to optimal policies when forced to maintain higher exploration throughout training.                                                                                                                                 |
| CNN | breakout-v4 | 1 | 32 | 1 | 0.03 | 0.0001 | 3,500,000 | 0.14 | 100,000 | 25,000 | 2,000 | With extended training to 3.5 million timesteps using the same configuration as the previous run, the agent shows significant improvement beyond the earlier plateau. The reward progresses through distinct phases: initial learning (0-500k timesteps) reaching ~2.0, continued growth (500k-1.5M timesteps) climbing to ~2.7, and a higher plateau (1.5M-3.5M timesteps) stabilizing around 2.4-2.8 with occasional spikes above 3.0. The episode length follows a similar trajectory, growing from ~10 to a plateau around 35-40 timesteps. This extended training demonstrates that the higher eps_final=0.03 configuration can achieve better performance than previous runs when given sufficient time, suggesting the increased exploration was beneficial for discovering more robust strategies, though it required significantly more timesteps to converge compared to lower exploration settings. |

### MLP

| DQN Policy | Environment | Gamma | Batch Size | Eps Initial | Eps Final | Learning Rate | Time Steps | Eps Decay | Buffer Size | Learning Starts | Target Update Interval | Observation |
|------------|-------------|-------|------------|--------------|------------|----------------|-------------|------------|---------------|------------------|--------------------------|-------------|
| MLP        | breakout-v4 | 0.99  | 64         | 1            | 0.05       | 0.0001         | 500000      | 0.1        | 10000         | 10000            | 1000                     | Rewards rose from 0.4 to 0.72 over 50k timesteps. Episode length improved to 15.26. Learning was stable with occasional fluctuations. Slow epsilon decay and low learning rate enabled effective long-term learning. |
| MLP        | breakout-v4 | 0.95  | 32         | 1            | 0.01       | 0.0005         | 500000      | 0.5        | 10000         | 10000            | 1000                     | Less stable reward growth compared to the previous run. Peaked at 0.51 and ended at 0.47, with shorter episode lengths (~9–13). Faster epsilon decay and high learning rate likely caused early exploitation and noisy updates. |
| MLP        | breakout-v4 | 1     | 64         | 1            | 0.02       | 0.0005         | 100000      | 0.5        | 10000         | 10000            | 1000                     | Reward improved moderately from 0.2 to 0.49, ending at 0.38. Episode length increased to 12.6. While more consistent than other MLP runs, performance was lower than the best run. Gamma=1 encouraged long-term planning but high learning rate limited progress. |
| MLP        | breakout-v4 | 0.99  | 32         | 1            | 0.05       | 0.0001         | 50000       | 0.5        | 100000        | 10000            | 1000                     | Reward started at ~0.29 and ended at 0.46, showing consistent, slow growth. Episode length rose from ~10.5 to 12.4. The smaller batch size and moderate learning rate helped smooth training. Lower peak performance but more stability than other MLP runs. |
| MLP        | breakout-v4 | 0.999  | 32         | 1            | 0.03       | 0.0001         | 1,000,000      | 0.14        | 100,000        | 10000            | 1000                     | Reward improved from ~0.2 to 0.5 with high variability, frequently spiking to 0.6-0.7 before dropping. Episode length remained stable around 10-13 timesteps with similar volatility. The extreme gamma=0.999 caused unstable learning dynamics that prevented convergence despite 1M timesteps of training. |
| MLP        | breakout-v4 | 1 | 32         | 1            | 0.05       | 0.0001         | 3,000,000      | 0.17        | 100,000        | 10000            | 1000                     | Reward improved from ~0.2 to 1.0-1.3 over 3M timesteps, plateauing around 0.8-1.3 after 500k steps. Episode length grew from ~8 to 18-22 timesteps. Extended training with gamma=1.0 achieved better performance than previous MLP runs but maintained high volatility throughout. |



**Note:** The hyperparameter tuning table is also available as an Excel spreadsheet in [`experiment_table/hyperparameter_observation.xlsx`](experiment_table/hyperparameter_observation.xlsx).

## Hyperparameter Tuning Discussion

The table above summarizes the impact of different hyperparameter configurations on agent performance. Increasing `gamma` from 0.85 to 0.99 improved the agent’s ability to focus on long-term rewards, resulting in higher mean rewards and more consistent learning. Adjusting `eps_decay` and training duration also led to smoother reward trends and better paddle control. The best results were achieved with extended training and higher buffer sizes, though further improvements are possible with additional tuning and exploration strategies.

## Group Collaboration and Individual Contributions

- **Task 1: EMMANUEL OBOLO;** Environment setup, initial training with CnnPolicy, baseline hyperparameter tuning, and README/documentation.
- **Task 2: MERVEILLE KANGABIRE;** Training with MlpPolicy, further hyperparameter tuning.
- **Task 3: SHOBI OLA-ADISA;** Play script creation, final fine-tuning, video recording, and policy comparison.

Each member contributed to different aspects of the project, with workload distributed across environment setup, training, evaluation, and documentation. The hyperparameter table and experiment logs reflect collaborative input and ongoing updates.

## Creating a virtual environment inorder for my python dependencies and modules not to clash
```bash
python -m venv deepQ
```

## activate the virtual environment
```bash
deepQ\Scripts\activate.bat
```

## install the dependencies using the requirements.txt file
```bash
pip install -r requirements.txt
```

## Accepting Rom licences
some systems require us to accpet rom licences so we would do that before
so incase it would be needed we would eventually already have it.
```bash
pip install autorom
AutoROM --accept-license
```
4. **Review results:**  
   - Training logs are in [`logs/`](logs)
   - Saved models are in [`models/`](models)
   - Play Atari Game using [`play/play.py`](play)
   - Hyperparameter observations are in [`experiment_table/hyperparameter_observation.xlsx`](experiment_table/hyperparameter_observation.xlsx)

## Policy Comparison

CNNs  outperform MLPs in this Breakout experiment, with 2.4-2.8 average rewards (peak 3.0) compared to MLPs' 0.8-1.3 and 35-40 timesteps versus 18-22, CNNs perform better at playing the breakout game 

This gap in performance most likely results from different architectural designs of the two policies: CNNs use convolutional layers, which have shared weights that can identify patterns anywhere on the screen and naturally detect spatial patterns like the trajectory of the ball, and paddle positions. This ability is especially important when playing Atari allowing the CNN to outperform the MLP policy. Learningspatial relationships may be more difficult with MLPs since they handle each of the 7,056 pixels individually.

MLPs continue to be volatile even after 3M timesteps, whereas CNNs show distinct learning phases and steady convergence. Although the precise mechanisms underlying CNNs' superior performance cannot be accurately decided, these results imply that they are more appropriate for visual tasks like this.

## Video Demonstration

### CNN Demo
![CNN Demo](./demos/CNN.gif)

### MLP Demo
![MLP Demo](./demos/MLP.gif)