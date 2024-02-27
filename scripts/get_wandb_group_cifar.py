import pandas as pd
import wandb
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from tqdm import tqdm

def fetch_run_data(run):
    try:
        history_df = run.history()
        history_df['run_id'] = run.id

        metadata_df = pd.DataFrame([run.config], index=[run.id])
        metadata_df['run_id'] = run.id

        return history_df, metadata_df
    except Exception as e:
        logging.error(f"Error fetching data for run {run.id}: {e}")
        return None, None

def main():
    api = wandb.Api(timeout=50)
    project_name = "rmt-ml/PAdam"

    all_runs_history = []
    all_runs_metadata = []

    print(f'Fetching runs...')
    # Filter runs by a specific group
    group="L0_2" #"OTO_2"
    runs = api.runs(project_name, {"$and": [{"group": group}]})

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(fetch_run_data, runs), total=len(runs), desc=f"Fetching Runs"))

    for history_df, metadata_df in results:
        if history_df is not None and metadata_df is not None:
            all_runs_history.append(history_df)
            all_runs_metadata.append(metadata_df)

    combined_history_df = pd.concat(all_runs_history, ignore_index=True)
    combined_metadata_df = pd.concat(all_runs_metadata, ignore_index=True)

    # Make sure save directory exists
    out_dir = "./results/CIFAR10"
    os.makedirs(out_dir, exist_ok=True)

    # Save to Pickle
    combined_history_df.to_pickle(os.path.join(out_dir, f"{group}_history.pkl"))
    combined_metadata_df.to_pickle(os.path.join(out_dir, f"{group}_metadata.pkl"))

    print("Data saved successfully.")

if __name__ == "__main__":
    main()
