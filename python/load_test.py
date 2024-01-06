import pickle

import torch
import torchvision
import torchvision.transforms as tt

import os

from functions import *
from models import *

import tqdm

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '../data/CIFAR10')
    mean_std_file=os.path.join(data_dir, "cifar10_mean_std.pkl")
    
    with open(mean_std_file, 'rb') as f:
        mean, std = pickle.load(f)
    
    transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])


    # Load the CIFAR-10 dataset with transforms above
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    

    device='mps'
    device_type = False
    
    loader_args = dict(num_workers=4, pin_memory=True) if device_type == 'cuda' else dict()
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, **loader_args)


    # move to device
    testloader = DeviceDataLoader(testloader, device)

    model = to_device(resnet18(10), device)

    model.eval()
    all_probabilities = []


    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(testloader):
            # Single forward pass
            logits = model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probabilities.append(probs.cpu())
            print(len(probs))


    
    print(len(all_probabilities))

if __name__ == '__main__':
    main()