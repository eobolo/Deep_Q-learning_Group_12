# DeepQ Learning Atari

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

## Folder Structure

| Folder             | Purpose                                 | What to put there                              |
|--------------------|-----------------------------------------|-----------------------------------------------|
| deepQ/             | Python virtual environment              | Nothing manually                              |
| evaluate/          | Evaluation/analysis scripts             | Evaluation scripts, analysis code             |
| experiment_table/  | Experiment tracking (tables, logs)      | Spreadsheets, experiment logs                 |
| logs/              | Training logs                           | Training CSV log files                        |
| models/            | Saved models                            | Model files (.zip, .pth, etc.)                |
| test_env/          | Env setup/testing scripts               | Env setup, test, or utility scripts           |
| train/             | Training scripts                        | Model training scripts                        |
| training_graphs/   | Training plots/graphs                   | Output plots/figures from evaluation scripts  |

