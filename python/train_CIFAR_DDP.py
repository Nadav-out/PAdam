import os
import subprocess
import torch
import torchvision

import torchvision.transforms as tt
import pickle
from torchvision.datasets import FashionMNIST
from torch.utils import data as dataloader
import copy
import time
import numpy as np
import math

from models import *
from Optimizers import *
from functions import *


# -----------------------------------------------------------------------------
# DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# -----------------------------------------------------------------------------

# default config values

#I/O
data_dir='./data'
out_dir='./CIFAR_results'
save_checkpoints=True

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'PAdam'
wandb_run_name = 'ResNet18' + str(time.time())

# dataset
num_workers = 4

# training
batch_size = 400
epochs = 100

# optimizer
optimizer_name = 'AdamW'
max_lr = 1e-3
lambda_p = 1e-3
p_norm = 0.8
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_epochs = int(0.8*epochs) # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

small_weights_threshold = 1e-13 # weights smaller than this will be considered "small"


# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# Initialize DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def parse_config():
    config = {}
    # Extracting global configuration keys
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    
    # Execute config file to update globals
    exec(open('./python/config.py').read(), globals())
    
    # Store updated configurations in a dictionary
    for k in config_keys:
        config[k] = globals()[k]
    return config


def main(rank, world_size, config):
    print(f"Rank {rank} is using CUDA device: cuda:{rank}")
    torch.cuda.set_device(rank)  # Explicitly set the device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    try:
        setup(rank, world_size)
        
        # Apply configurations to the global namespace of this process
        globals().update(config)

        

        # set up I/O directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_save_path = os.path.join(out_dir, 'best_model.pth')
        stats_save_path = os.path.join(out_dir, 'training_stats.pkl')

        # Set a different random seed for each process
        torch.manual_seed(1337 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in device else 'cpu'

        # Check if mean/std file exists, calculate if not
        mean_std_file = os.path.join(data_dir, 'cifar10_mean_std.pkl')
        if not os.path.exists(mean_std_file):
            # Load CIFAR-10 without normalization
            trainset_raw = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tt.ToTensor())
            mean, std = calculate_mean_std(trainset_raw)
            with open(mean_std_file, 'wb') as f:
                pickle.dump((mean, std), f)
            print("Mean and Std Dev calculated and saved.\n")
        else:
            with open(mean_std_file, 'rb') as f:
                mean, std = pickle.load(f)
            print("Mean and Std Dev loaded from file.\n")


        transform_train = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 

                                tt.RandomRotation(degrees=(-10, 10)),
                                tt.RandomHorizontalFlip(), 
                                #tt.RandomPerspective(distortion_scale=0.14),
                                # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                                # tt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                                tt.ToTensor(), 
                                tt.Normalize(mean,std,inplace=True)])
        transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])

        # Load datasets with transforms
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # Initialize DataLoaders with DistributedSampler for DDP
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=3, pin_memory=True)

        test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=2*batch_size, sampler=test_sampler, num_workers=3, pin_memory=True)
        

        # Model Initialization for DDP
        # model = ResNet18(3, 10).to(rank)
        model = SimpleNet().to(rank)
        print(f"Initializing DDP on rank {rank}...")
        model = DDP(model, device_ids=[rank])
        print("DDP initialized successfully.")

    
        print(f"Number of trainable parameters: {count_parameters(model):,}")


        # initialize a GradScaler. If enabled=False scaler is a no-op
        if device_type == 'cuda':
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(optimizer_name, lambda_p, max_lr, p_norm, (beta1, beta2), device_type)

        if compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        criterion = torch.nn.CrossEntropyLoss()


        lr_decay_iters=len(trainloader)*lr_decay_epochs

        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return max_lr * it / warmup_iters+1e-6
            # 2) if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (max_lr - min_lr)
        
        best_accuracy = 0.0
        iteration_count = 0
        train_losses = []
        val_losses = []
        accuracies = []
        lrs = []
        small_weight_fractions = []
        start_time = time.time()    
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data in trainloader:
                    # determine and set the learning rate for this iteration
                lr = get_lr(iteration_count) if decay_lr else max_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                iteration_count += 1
                inputs, labels = data[0].to(rank), data[1].to(rank)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(trainloader)
            train_losses.append(avg_train_loss)

            # Validation loop
            if rank==0:
                model.eval()
                correct = 0
                total = 0
                running_val_loss = 0.0

                with torch.no_grad():
                    for data in testloader:
                        images, labels = data[0].to(rank), data[1].to(rank)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        



                avg_val_loss = running_val_loss / len(testloader)
                val_losses.append(avg_val_loss)

                accuracy = 100 * correct / total
                accuracies.append(accuracy)


                small_weight_fractions.append([fraction_small_weights(param, small_weights_threshold) for param in model.parameters() if param.requires_grad])
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy,
                    }, model_save_path)
                    print(f"Reached accuracy {best_accuracy:.2f}% on epoch {epoch+1}. Model saved to {model_save_path}.")

                    # Calculate and format runtime and expected time
                    elapsed_time = time.time() - start_time
                    expected_time = elapsed_time * epochs / (epoch + 1)
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                    expected_str = time.strftime("%H:%M:%S", time.gmtime(expected_time))

                    # Track and store current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    lrs.append(current_lr)

                    status_message = f"Epoch: {epoch+1}/{epochs}\tTrain Loss: {avg_train_loss:.4f}\tTest Loss: {avg_val_loss:.4f}\tAccuracy: {accuracy:.2f}%\tCurrent LR: {current_lr:.5f}\tElapsed Time: {elapsed_str}\tExpected Time: {expected_str}"
                    print(f"\r{status_message}",end='')

        print()


        with open(stats_save_path, 'wb') as f:
            pickle.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies,
                'lrs': lrs
            }, f)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()




    

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    config = parse_config()
    torch.multiprocessing.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
























