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
    runs = api.runs(project_name)

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
    combined_history_df.to_pickle(os.path.join(out_dir, "all_runs_history.pkl"))
    combined_metadata_df.to_pickle(os.path.join(out_dir, "all_runs_metadata.pkl"))

    print("Data saved successfully.")

if __name__ == "__main__":
    main()


# import pandas as pd
# import wandb
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# import logging
# import os
# from tqdm import tqdm
# from datetime import datetime
# import pytz

# def fetch_run_data(run):
#     try:
#         history_df = run.history()
#         history_df['run_id'] = run.id

#         metadata_df = pd.DataFrame([run.config], index=[run.id])
#         metadata_df['run_id'] = run.id

#         return history_df, metadata_df
#     except Exception as e:
#         logging.error(f"Error fetching data for run {run.id}: {e}")
#         return None, None

# def main():
#     api = wandb.Api(timeout=50)
#     project_name = "rmt-ml/PAdam"

#     all_runs_history = []
#     all_runs_metadata = []

#     # Define the user, timestamp, and groups
#     user = "noamlevi"
#     timestamp = datetime.strptime("2024-01-14 14:26:05.138598", "%Y-%m-%d %H:%M:%S.%f")
#     timestamp = timestamp.replace(tzinfo=pytz.UTC)  # Convert to UTC
#     timestamp_unix = timestamp.timestamp()  # Convert to Unix time

#     groups = ['pAdam 0.1', 'pAdam 0.2', 'pAdam 0.3', 'pAdam 0.4', 'pAdam 0.5',
#               'pAdam 0.6', 'pAdam 0.7', 'pAdam 0.8', 'pAdam 0.9', 'pAdam 1.0', 
#               'pAdam 1.2', 'pAdam 1.5', 'pAdam 1.9', 'AdamW', 'AdamW 2']

#     print(f'Fetching runs...')
#     runs = api.runs(project_name)

#     # Filter the runs based on the user, timestamp, and group
#     filtered_runs = [run for run in runs if run.user == user and run.created_at.timestamp() > timestamp_unix and run.group in groups]

#     print(f'Total runs: {len(runs)}')
#     print(f'Filtered runs: {len(filtered_runs)}')
    
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         results = list(tqdm(executor.map(fetch_run_data, filtered_runs), total=len(filtered_runs), desc=f"Fetching Runs"))

#     for history_df, metadata_df in results:
#         if history_df is not None and metadata_df is not None:
#             all_runs_history.append(history_df)
#             all_runs_metadata.append(metadata_df)

#     combined_history_df = pd.concat(all_runs_history, ignore_index=True)
#     combined_metadata_df = pd.concat(all_runs_metadata, ignore_index=True)

#     # Make sure save directory exists
#     out_dir = "./results/CIFAR10"
#     os.makedirs(out_dir, exist_ok=True)

#     # Save to Pickle
#     combined_history_df.to_pickle(os.path.join(out_dir, "all_runs_history.pkl"))
#     combined_metadata_df.to_pickle(os.path.join(out_dir, "all_runs_metadata.pkl"))

#     print("Data saved successfully.")

# if __name__ == "__main__":
#     main()