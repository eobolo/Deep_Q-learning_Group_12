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

## Hyperparameter Tuning Table

| Policy | Env         | Gamma | Batch | Eps Init | Eps Final | LR     | Buffer | Eps Decay | Learning Starts | Target Update | Timesteps | Observed Behavior |
|--------|-------------|-------|-------|----------|-----------|--------|--------|-----------|-----------------|---------------|-----------|-------------------|
| CNN    | breakout-v4 | 0.85  | 32    | 1        | 0.02      | 0.00   | 50000  | 0.10      | 10000           | 10000         | 1000      | Model underfits, mean reward never exceeds 0.7, brief spike at 20k steps, gradual upward trend, episode length 10-15. More training or exploration needed. |
| CNN    | breakout-v4 | 0.99  | 32    | 1        | 0.02      | 0.00   | 50000  | 0.10      | 10000           | 10000         | 1000      | Higher gamma improves mean reward (0.2–0.8), upward trend, episode length 10-15, better long-term strategy. |
| CNN    | breakout-v4 | 0.99  | 32    | 1        | 0.05      | 0.00   | 50000  | 0.40      | 10000           | 10000         | 1000      | Gradual reward increase (0.2–1.0), smooth learning, episode length 2–18, agent learning basic paddle control. |
| CNN    | breakout-v4 | 1     | 32    | 1        | 0.02      | 0.0001 | 1000000| 0.10      | 50000           | 1000          | 1000000   | Mean reward improves (0.5–2.5), steady upward trend, episode length 10–35, agent learning to break bricks, benefits from longer training. |

**Note:** The hyperparameter tuning table is also available as an Excel spreadsheet in [`experiment_table/hyperparameter_observation.xlsx`](experiment_table/hyperparameter_observation.xlsx).

## Hyperparameter Tuning Discussion

The table above summarizes the impact of different hyperparameter configurations on agent performance. Increasing `gamma` from 0.85 to 0.99 improved the agent’s ability to focus on long-term rewards, resulting in higher mean rewards and more consistent learning. Adjusting `eps_decay` and training duration also led to smoother reward trends and better paddle control. The best results were achieved with extended training and higher buffer sizes, though further improvements are possible with additional tuning and exploration strategies.

## Group Collaboration and Individual Contributions

- **Task 1: EMMANUEL OBOLO** Environment setup, initial training with CnnPolicy, baseline hyperparameter tuning, and README/documentation.
- **Task 2: MERVEILLE KANGABIRE** Training with MlpPolicy, further hyperparameter tuning.
- **Task 3: SHOBI OLADISA** Play script creation, final fine-tuning, video recording, and policy comparison.

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

## Video Demonstration

*A video showing the agent playing in the Atari environment (using play.py) will be added here.*
