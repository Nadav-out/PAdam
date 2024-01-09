import pandas as pd
import wandb
import pickle
from tqdm import tqdm

def main():
    api = wandb.Api()
    project_name = "rmt-ml/PAdam"

    # Fetch runs from the project
    runs = api.runs(project_name)

    all_runs_history = []
    all_runs_metadata = []
    for run in tqdm(runs):
        # Fetch the history (time-series metrics) for each run
        history_df = run.history()
        history_df['run_id'] = run.id
        all_runs_history.append(history_df)

        # Fetch the configuration (metadata) for each run
        metadata_df = pd.DataFrame([run.config], index=[run.id])
        metadata_df['run_id'] = run.id  # Explicitly add the 'run_id' column
        all_runs_metadata.append(metadata_df)


    # Concatenate all runs into single DataFrames
    combined_history_df = pd.concat(all_runs_history, ignore_index=True)
    combined_metadata_df = pd.concat(all_runs_metadata, ignore_index=True)

    # Save to Pickle
    combined_history_df.to_pickle("./results/CIFAR10/all_runs_history.pkl")
    combined_metadata_df.to_pickle("./results/CIFAR10/all_runs_metadata.pkl")

    print("Data saved successfully.")

if __name__ == "__main__":
    main()
