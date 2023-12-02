import os
import pickle
import torchvision
import torchvision.transforms as tt
from functions import calculate_mean_std
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR10', help='directory to store data')
    data_dir = parser.parse_args().data_dir

    # set up output directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    trainset_raw = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tt.ToTensor())
    testset_raw = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tt.ToTensor())

    # Check if mean/std file exists, calculate if not
    mean_std_file = os.path.join(data_dir, 'cifar10_mean_std.pkl')
    if not os.path.exists(mean_std_file):  
        mean, std = calculate_mean_std(trainset_raw)
        with open(mean_std_file, 'wb') as f:
            pickle.dump((mean, std), f)
        print("Mean and Std Dev calculated and saved.\n")
    else:
        print("Mean and Std Dev already exists.\n")

if __name__ == '__main__':
    main()
