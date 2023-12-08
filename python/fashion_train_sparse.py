import argparse
import os
import subprocess
import torch
import torchvision.transforms as transforms
import pickle
from torchvision.datasets import FashionMNIST
from torch.utils import data as dataloader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import copy
import time
import numpy as np

from models import *
from Optimizers import *
from functions import *

def get_args():
    parser = argparse.ArgumentParser(description='Training script for Fashion MNIST.')

    # Device and Model Parameters
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training (default: cuda:3, can be cuda, cpu, or mps)')
    
    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--seed', type=int, default=2349, help='Seed for reproducibility (default: 2349)')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training (default: 400)')
    parser.add_argument('--small_weight_threshold', type=float, default=1e-11, help='Threshold below which we say a weight is small (default: 1e-11)')

    # Optimizer Parameters
    parser.add_argument('--lr_1', type=float, default=3e-3, help='Learning rate for first model (default: 3e-3)')
    parser.add_argument('--lr_2', type=float, default=3e-3, help='Learning rate for second model (default: 3e-3)')
    parser.add_argument('--lambda_p1', type=float, default=3e-3, help='Lambda parameter for Adam + LP optimizer (default: 3e-3)')
    parser.add_argument('--lambda_p2', type=float, default=3e-3, help='Lambda parameter for PAdam optimizer (default: 3e-3)')
    parser.add_argument('--p_norm', type=float, default=0.8, help='p-norm for PAdam optimizer (default: 0.8)')

    # Scheduler Parameter
    parser.add_argument('--scheduler_exponent', type=float, default=0.6, help='Exponent for calculating the LR scheduler decay rate (default: 0.6)')

    # Output Folder Parameter
    default_save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='Directory to save trained models and metrics')

    return parser.parse_args()




def main():
    args = get_args()
    # Print all command-line arguments
    print('\n')
    print("Running with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print('\n')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or 'mps' in args.device else "cpu")
    print(f"Using device: {device}")
    print('\n')

    # Set seed for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Define the path of the stats file relative to this script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '../data')
    stats_file = os.path.join(data_dir, 'fashion_mnist_stats.pkl')

    # Check for stats file, run fashion_prep.py if not found
    if not os.path.exists(stats_file):
        print("Statistics file not found. Running 'fashion_prep.py'.")
        subprocess.run(['python', os.path.join(script_dir, 'fashion_prep.py')], check=True)



    # Load stats file
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        train_mean = stats['train_mean']
        train_std = stats['train_std']
        test_mean = stats['test_mean']
        test_std = stats['test_std']

    # Define transforms using calculated mean and std
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(train_mean,), std=(train_std,))
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random", inplace=False)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(test_mean,), std=(test_std,))
    ])

    # Create new datasets using the normalization calculated
    train = FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
    test = FashionMNIST(data_dir, train=False, download=True, transform=test_transform)

    # Create DataLoader
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=args.batch_size)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    
    
    # Initialize and duplicate models
    Model_1 = FashionCNN()
    Model_2 = copy.deepcopy(Model_1)

    # Move models to the appropriate device
    Model_1.to(device)
    Model_2.to(device)

    # Print total number of parameters
    total_params = count_parameters(Model_1)
    print(f"Total trainable parameters: {total_params}\n")

    # Set up optimizers
    optimizer_1 = optim.Adam(Model_1.parameters(), lr=args.lr_1, weight_decay=0)
    optimizer_2 = PAdam(Model_2.parameters(), lr=args.lr_2, lambda_p=args.lambda_p2, p_norm=args.p_norm)

    # Set up schedulers
    decay_rate = 10 ** (-args.scheduler_exponent / args.epochs)
    scheduler_1 = ExponentialLR(optimizer_1, decay_rate)
    scheduler_2 = ExponentialLR(optimizer_2, decay_rate)
    
    # Initialize dictionaries of things we want to keep
    params_epochs = {
        'train_1': [], 'train_2': [],
        'test_1': [], 'test_2': [],
        'accuracy_1': [], 'accuracy_2': [],
        'small_weights_1': [], 'small_weights_2': []
    }

    parameters_per_layer = np.array([p.nelement() for p in Model_2.parameters() if p.requires_grad])
    parameters_per_layer_weight = parameters_per_layer / np.sum(parameters_per_layer)

    # Function to calculate the fraction of small weights for a given parameter tensor
    def fraction_small_weights(param, threshold):
        small_weights = torch.sum(torch.abs(param.data) < threshold).item()
        total_weights = param.nelement()
        return small_weights / total_weights

    # Record the start time
    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        Model_1.train()
        Model_2.train()

        train_loss_epoch_1 = 0
        train_loss_epoch_2 = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            y_pred_1 = Model_1(data)
            y_pred_2 = Model_2(data)

            loss_1 = F.cross_entropy(y_pred_1, target)
            loss_2 = F.cross_entropy(y_pred_2, target)
            train_loss_epoch_1 += loss_1.item()
            train_loss_epoch_2 += loss_2.item()

            # Calculate and add L_p^p regularization for Model_1
            reg_loss_1 = torch.tensor(0., requires_grad=True).to(device)
            for param in Model_1.parameters():
                if param.requires_grad:
                    reg_loss_1 += torch.sum(torch.abs(param) ** args.p_norm)
            loss_1 += args.lambda_p1 * reg_loss_1



            loss_1.backward()
            optimizer_1.step()
            loss_2.backward()
            optimizer_2.step()

        # Calculate the fraction of small weights for each layer
        small_weights_layers_1 = [fraction_small_weights(param, args.small_weight_threshold) for param in Model_1.parameters() if param.requires_grad]
        small_weights_layers_2 = [fraction_small_weights(param, args.small_weight_threshold) for param in Model_2.parameters() if param.requires_grad]

        # Calculate the total fraction of small weights for each model
        total_frac_model_1 = parameters_per_layer_weight @ np.array(small_weights_layers_1)
        total_frac_model_2 = parameters_per_layer_weight @ np.array(small_weights_layers_2)

        # Append the fractions of small weights for each layer along with the total fraction
        params_epochs['small_weights_1'].append(small_weights_layers_1 + [total_frac_model_1])
        params_epochs['small_weights_2'].append(small_weights_layers_2 + [total_frac_model_2])

        # Calculate and store the average training loss for this epoch
        params_epochs['train_1'].append(train_loss_epoch_1 / len(train_loader))
        params_epochs['train_2'].append(train_loss_epoch_2 / len(train_loader))

        # Evaluation phase
        test_loss_1, correct_1, test_loss_2, correct_2 = 0, 0, 0, 0
        Model_1.eval()
        Model_2.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output_1 = Model_1(data)
                test_loss_1 += F.cross_entropy(output_1, target).item()
                pred_1 = output_1.argmax(dim=1, keepdim=True)
                correct_1 += pred_1.eq(target.view_as(pred_1)).sum().item()

                output_2 = Model_2(data)
                test_loss_2 += F.cross_entropy(output_2, target).item()
                pred_2 = output_2.argmax(dim=1, keepdim=True)
                correct_2 += pred_2.eq(target.view_as(pred_2)).sum().item()

        # Calculate and store the average test loss and accuracy
        params_epochs['test_1'].append(test_loss_1 / len(test_loader))
        params_epochs['test_2'].append(test_loss_2 / len(test_loader))
        params_epochs['accuracy_1'].append(100. * correct_1 / len(test_loader.dataset))
        params_epochs['accuracy_2'].append(100. * correct_2 / len(test_loader.dataset))

        # Update learning rate
        scheduler_1.step()
        scheduler_2.step()

        # Calculate and format runtime and expected time
        elapsed_time = time.time() - start_time
        expected_time = elapsed_time * args.epochs / (epoch + 1)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        expected_str = time.strftime("%H:%M:%S", time.gmtime(expected_time))

        # Print status
        status_message = f"Epoch: {epoch+1}/{args.epochs}\tTrain Loss: {params_epochs['train_1'][-1]:.4f} | {params_epochs['train_2'][-1]:.4f}\tTest Loss: {params_epochs['test_1'][-1]:.4f} | {params_epochs['test_2'][-1]:.4f}\tAccuracy: {params_epochs['accuracy_1'][-1]:.2f}% | {params_epochs['accuracy_2'][-1]:.2f}%\tSmall Weights: {100*total_frac_model_2:.2f}%\tElapsed Time: {elapsed_str}\tExpected Time: {expected_str}"
        print(f"\r{status_message:<150}", end='')

    print()



    # Save the trained models
    torch.save(Model_1.state_dict(), os.path.join(args.save_dir, 'fashion_Model_1.pth'))
    torch.save(Model_2.state_dict(), os.path.join(args.save_dir, 'fashion_Model_2.pth'))

    # Save metrics
    with open(os.path.join(args.save_dir, 'fashion_metrics.pkl'), 'wb') as f:
        pickle.dump(params_epochs, f)

if __name__ == "__main__":
    main()
