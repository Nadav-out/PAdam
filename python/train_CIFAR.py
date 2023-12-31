import os
from requests import get
import torch
import torchvision

import torchvision.transforms as tt
import pickle

import time
import numpy as np
import math
import pandas as pd

from models import *
from Optimizers import *
from functions import *
import subprocess

script_dir = os.path.dirname(os.path.realpath(__file__))

# -----------------------------------------------------------------------------
# default config values

#I/O
relative_paths=True
data_dir = '../data/CIFAR10'
out_dir= '../results/CIFAR10'
save_checkpoints = False
save_model = False

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
optimizer_name = 'PAdam'
max_lr = 1e-3
lambda_p = 1e-3
p_norm = 0.8
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
lr1 = True # whether to use the lr1 function or the lr2 function
warmup_iters = 500 # how many steps to warm up for
lr_decay_frac=1.0 # fraction of the max_lr to drop to at lr_decay_epochs
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
small_weights_threshold = 1e-13 # weights smaller than this will be considered "small"


# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_path = os.path.join(script_dir, 'config.py')
exec(open(config_path).read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------



def main():

    

    # If relative paths are used:
    if relative_paths:
        global data_dir, out_dir
        data_dir = os.path.join(script_dir, data_dir)
        out_dir = os.path.join(script_dir, out_dir)
        
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'



    mean_std_file=os.path.join(data_dir, "cifar10_mean_std.pkl")
    data_paths_check = [
        data_dir,
        os.path.join(data_dir, "cifar-10-batches-py"),
        mean_std_file
    ]
    for path in data_paths_check:
        if not os.path.exists(path):
            print("CIFAR-10 data not found. Running 'CIFAR_10_prep.py'.")
            subprocess.run(['python', os.path.join(script_dir, 'CIFAR_10_prep.py'),f'--data_dir={data_dir}'], check=True)



    with open(mean_std_file, 'rb') as f:
        mean, std = pickle.load(f)




    transform_train = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 

                            tt.RandomRotation(degrees=(-10, 10)),
                            # tt.RandomHorizontalFlip(), 
                            tt.RandomHorizontalFlip(0.5),
                            tt.RandomCrop(32, padding=4),
                            #tt.RandomPerspective(distortion_scale=0.14),
                            # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                            # tt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                            tt.ToTensor(), 
                            tt.Normalize(mean,std,inplace=True)])
    transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])


    # Load the CIFAR-10 dataset with transforms above
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
    

    loader_args = dict(num_workers=num_workers, pin_memory=True) if device_type == 'cuda' else dict()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **loader_args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2*batch_size, shuffle=False, **loader_args)


    # move to device
    trainloader = DeviceDataLoader(trainloader, device)
    testloader = DeviceDataLoader(testloader, device)


    # initialize model
    # model = to_device(ResNet18(3, 10), device)
    model = to_device(resnet18(10), device)




    # initialize a GradScaler. If enabled=False scaler is a no-op
    # if device_type == 'cuda':
    #     scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    if optimizer_name=='Manual':
        optimizer = model.configure_optimizers('AdamW', 0, max_lr, 0, (beta1, beta2), device_type)
    else:
        optimizer = model.configure_optimizers(optimizer_name, lambda_p, max_lr, p_norm, (beta1, beta2), device_type)

    if compile and device_type == 'cuda':
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    
    
    lr_decay_epochs = int(lr_decay_frac*epochs) 
    lr_decay_iters=len(trainloader)*lr_decay_epochs # should be ~= max_iters per Chinchilla
    # learning rate decay scheduler (cosine with warmup)
    def get_lr1(it):
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
    
    def cosine_annealing(it, T_max, max_lr, min_lr):
        # Adjust the iteration count and T_max for the cosine function
        cosine_value = math.cos(math.pi * it / T_max)
        lr = min_lr + (max_lr - min_lr) * (1 + cosine_value) / 2
        return lr

    # Global variables
    q = epochs // 15
    epoch_steps = [epochs - q * x for x in [0, 8, 12, 14]]
    epoch_steps.append(0)  # Append 0 to represent the start of the training
    epoch_steps = epoch_steps[::-1]  # Reverse to get the correct order
    it_steps = [len(trainloader) * e for e in epoch_steps]  # Convert epochs to iterations

    rate=80*len(trainloader)
    def get_lr2(it):
        # for i in range(len(it_steps) - 1):
        #     if it < it_steps[i + 1]:
        #         return cosine_annealing(it - it_steps[i], it_steps[i + 1] - it_steps[i], max_lr, min_lr)
        # return min_lr
        return max_lr*0.1**(it//rate)

    if lr1:
        get_lr = get_lr1
    else:   
        get_lr = get_lr2


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    accuracy_save_path = os.path.join(out_dir, 'best_accuracy_model.pth')
    loss_save_path = os.path.join(out_dir, 'best_loss_model.pth')
    stats_save_path = os.path.join(out_dir, 'training_stats.pkl')



    best_accuracy = 0.0
    best_val_loss = np.inf
    iteration_count = 0
    train_losses = []
    val_losses = []
    accuracies = []
    lrs = [] 

    # logging
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

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
            inputs, labels = data



            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

         
            if optimizer_name == 'Manual' and epoch>0:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is None:
                            continue

                        # Apply the general Lp^p regularization
                        if lambda_p != 0:
                            lp_grad = (param.data.abs()**(p_norm - 2)) * param.data
                            param.grad.data.add_(lp_grad, p_norm * lambda_p)  


            loss.backward()
            # scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            # scaler.step(optimizer)
            # scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            


        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
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


        # update small weights count
        cur_sparsity = model.append_small_weight_vec(small_weights_threshold, epoch)


        
        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "validation/loss": avg_val_loss,
                "validation/accuracy": accuracy,
                "lr": lr,
                "sparsity": cur_sparsity,
                "decayed_weights_hist": wandb.Histogram(np_histogram=model.decayed_weights_histogram()),
                "validation/best_accuracy": best_accuracy,
                "validation/best_val_loss": best_val_loss,
            })

        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, accuracy_save_path)
                print(f"\n\nReached accuracy {best_accuracy:.2f}% on epoch {epoch+1}. Model saved to {accuracy_save_path}.")
            else:
                print(f"\n\nReached accuracy {best_accuracy:.2f}% on epoch {epoch+1}.")
            print(f'Sparsity: {cur_sparsity:.5f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, loss_save_path)
                print(f"\n\nReached validation loss {best_val_loss:.4f} on epoch {epoch+1}. Model saved to {loss_save_path}.")
            else:
                print(f"\n\nReached validation loss {best_val_loss:.4f} on epoch {epoch+1}.")
            print(f'Sparsity: {cur_sparsity:.5f}')
            

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


    if save_model:
        with open(stats_save_path, 'wb') as f:
            pickle.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies,
                'lrs': lrs,
                'small_weights': model.sparsity_df
            }, f)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
