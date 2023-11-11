import os
import torch
import torchvision.transforms as transforms
import pickle
from torchvision.datasets import FashionMNIST

def prepare_data(script_dir):
    data_dir = os.path.join(script_dir, '../data')


    # Check if data directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Data directory created at: {data_dir}")

    # Check if the stats file exists
    stats_file = os.path.join(data_dir, 'fashion_mnist_stats.pkl')
    if not os.path.exists(stats_file):
        print("Stats file not found. Preparing data...")

        # Download data without any transforms
        print("Downloading FashionMNIST data...")
        train_raw = FashionMNIST(data_dir, train=True, download=True)
        test_raw = FashionMNIST(data_dir, train=False, download=True)
        print("Download complete.")

        # Convert train and test data to tensors
        print("Processing data...")
        train_data = torch.stack([transforms.ToTensor()(image) for image, _ in train_raw])
        test_data = torch.stack([transforms.ToTensor()(image) for image, _ in test_raw])

        # Calculate mean and std
        print("Calculating mean and standard deviation...")
        train_mean = torch.mean(train_data)
        train_std = torch.std(train_data)
        test_mean = torch.mean(test_data)
        test_std = torch.std(test_data)

        # Save the mean and std dev values
        with open(stats_file, 'wb') as f:
            pickle.dump({'train_mean': train_mean, 'train_std': train_std, 'test_mean': test_mean, 'test_std': test_std}, f)
        print(f"Statistics saved in '{stats_file}'")
        print("Data preparation completed.")
    else:
        print(f"Stats file already exists at: {stats_file}.\nNo need to prepare data again.")

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print("Starting data preparation...")
    prepare_data(script_dir)
    

if __name__ == "__main__":
    main()
