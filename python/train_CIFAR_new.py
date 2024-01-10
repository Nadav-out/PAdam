import argparse
import os


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

from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.live import Live
from rich.table import Table




def get_args():
    parser = argparse.ArgumentParser(description="Training script for CIFAR10")

    # Add progress_bar and verbose arguments
    parser.add_argument('--progress_bar', action='store_true', help='Enable rich live layout progress bar')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')


    # I/O
    parser.add_argument('--absolute_paths', action='store_true', help="Whether to use absolute paths")
    parser.add_argument('--data_dir', type=str, default='../data/CIFAR10', help="Directory for data")
    parser.add_argument('--out_dir', type=str, default='../results/CIFAR10', help="Output directory")
    parser.add_argument('--save_checkpoints', action='store_true', help="Save checkpoints during training")
    parser.add_argument('--save_model', action='store_true', help="Save the final model")

    # WandB Logging
    parser.add_argument('--wandb_log', action='store_true', help="Enable logging to Weights & Biases")
    parser.add_argument('--wandb_project', type=str, default='PAdam', help="Weights & Biases project name")
    parser.add_argument('--wandb_run_name', type=str, default='ResNet18' + str(time.time()), help="Weights & Biases run name")
    parser.add_argument('--wandb_group', type=str, default=None, help="Weights & Biases run group")

    # Dataset
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")

    # Training
    parser.add_argument('--batch_size', type=int, default=400, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")

    # Optimizer
    parser.add_argument('--optimizer_name', type=str, default='PAdam', help="Name of the optimizer")
    parser.add_argument('--max_lr', type=float, default=1e-3, help="Maximum learning rate")
    parser.add_argument('--lambda_p', type=float, default=1e-3, help="Lambda parameter value")
    parser.add_argument('--p_norm', type=float, default=0.8, help="P-norm value")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument('--grad_clip', type=float, default=0.0, help="Gradient clipping value")

    # Learning Rate Decay Settings
    parser.add_argument('--non_decay_lr', action='store_true', help="Disable learning rate decay")
    parser.add_argument('--warmup_epochs', type=float, default=2, help="Number of warmup epochs, can be fractional")
    parser.add_argument('--lr_decay_frac', type=float, default=1.0, help="Fraction of max_lr to decay to")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument('--small_weights_threshold', type=float, default=1e-13, help="Threshold for considering weights as small")

    # System
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training (e.g., 'cuda', 'cpu', 'mps')")
    parser.add_argument('--compile', action='store_true', help="Use PyTorch 2.0 to compile the model for faster training")

    return parser.parse_args()

def cosine_lambda(epoch, epochs, max_lr, min_lr, warmup_epochs, lr_decay_frac):
    # linear warmup for warmup_epochs steps
    if epoch <= warmup_epochs:
        coef=(max_lr-min_lr) / warmup_epochs
        lr = coef * epoch+min_lr
    # Constant learning rate after finished decaying
    elif epoch > epochs * lr_decay_frac: # Usually lr_decay_frac~1
        lr = min_lr
    # Cosine annealing from max_lr to min_lr in between
    else:
        decay_ratio = (epoch - warmup_epochs) / (epochs * lr_decay_frac - warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        lr = min_lr + coeff * (max_lr - min_lr)
    return lr/max_lr
        
def train_one_epoch(model, trainloader, optimizer, criterion, scheduler, grad_clip):
    model.train()
    running_loss = 0.0

    for data in trainloader:
        inputs, labels = data

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        running_loss += loss.item()


        loss.backward()
        # clip the gradient
        if grad_clip != 0.0:            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # lr step
        scheduler.step()


    avg_train_loss = running_loss / len(trainloader)
    return avg_train_loss




def validate(model, testloader, criterion):
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    probs_array = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            logits = model(images)
            loss = criterion(logits, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs_array.append(probs.cpu())
            

    avg_val_loss = running_val_loss / len(testloader)
    accuracy = 100 * correct / total

    return avg_val_loss, accuracy, probs_array

def update_display(progress, layout, avg_train_loss, avg_val_loss, accuracy, current_lr, cur_sparsity, best_val_loss_str, best_accuracy_str):
    
    # Update the progress
    layout["progress"].update(progress)

    # Update the current status
    status = format_status(avg_train_loss, avg_val_loss, accuracy, current_lr, cur_sparsity)
    layout["status"].update(status)

    # Update the best results in a table
    results_table = Table.grid(padding=(0, 2))
    results_table.add_column("Metric", justify="left")
    results_table.add_column("Value", justify="left")
    
    results_table.add_row(best_val_loss_str)
    results_table.add_row(best_accuracy_str)
    layout["best_results"].update(results_table)

    
    
    


    



def main():
    # Collect arguments 
    args = get_args()
    args.min_lr = args.max_lr / 10

    # script_dir can be used as before
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Convert args to a dictionary for easy access and logging
    config = vars(args)
    if args.verbose:
        print("\n")
        for arg in config:
            print(f"{arg}: {getattr(args, arg)}")
        print("\n")


    
    # If relative paths are used, update data_dir and out_dir
    if not args.absolute_paths:
        args.data_dir = os.path.join(script_dir, args.data_dir)
        args.out_dir = os.path.join(script_dir, args.out_dir)
        
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'



    mean_std_file=os.path.join(args.data_dir, "cifar10_mean_std.pkl")
    data_paths_check = [
        args.data_dir,
        os.path.join(args.data_dir, "cifar-10-batches-py"),
        mean_std_file
    ]
    for path in data_paths_check:
        if not os.path.exists(path):
            print("CIFAR-10 data not found. Running 'CIFAR_10_prep.py'.")
            subprocess.run(['python', os.path.join(script_dir, 'CIFAR_10_prep.py'),f'--data_dir={args.data_dir}'], check=True)



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
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)
    

    loader_args = dict(num_workers=args.num_workers, pin_memory=True) if device_type == 'cuda' else dict()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **loader_args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **loader_args)


    # move to device
    trainloader = DeviceDataLoader(trainloader, args.device)
    testloader = DeviceDataLoader(testloader, args.device)


    # initialize model
    model = to_device(resnet18(10), args.device)

    # optimizer
    optimizer = model.configure_optimizers(args.optimizer_name, args.lambda_p, args.max_lr, args.p_norm, (args.beta1, args.beta2), device_type)    
    
    # scheduler
    if args.non_decay_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: 1)
        # num_iters=len(trainloader)*args.epochs
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_iters//2,2*num_iters//3,3*num_iters//4,4*num_iters//5], gamma=0.5)
        # # rate=(args.min_lr/args.max_lr)**(1/num_iters)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=rate)
    
    else: 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: cosine_lambda(iter/len(trainloader), args.epochs, args.max_lr, args.min_lr, args.warmup_epochs, args.lr_decay_frac))
    

    if args.compile and device_type == 'cuda':
        print("Compiling the model...\n")
        model = torch.compile(model) # requires PyTorch 2.0

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')



    if args.save_checkpoints or args.save_model:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        accuracy_save_path = os.path.join(args.out_dir, 'best_accuracy_model.pth')
        loss_save_path = os.path.join(args.out_dir, 'best_loss_model.pth')
        stats_save_path = os.path.join(args.out_dir, 'training_stats.pkl')


    
    # logging
    if args.wandb_log:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    
    console = Console()
    if args.verbose:
        console.print("\nTraining...\n")

    if args.progress_bar:
        # Set up the progress bar
        progress = Progress(
            TextColumn("Epoch:", justify="left"),
            MofNCompleteColumn(),
            BarColumn(),
            TextColumn("•"),
            TextColumn("Time Elapsed:", justify="left"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("Time Remaining:", justify="left"),
            TimeRemainingColumn(),
            expand=False
        )
        task_id = progress.add_task("Training", total=args.epochs)

        # Set up the layout
        layout = Layout()
        layout.split(
            Layout(name="progress", size=1),
            Layout(name="status", size=2),
            Layout(name="best_results", size=2)
        )
        layout["progress"].update(progress)

        # Start the Live context
        live = Live(layout, console=console, auto_refresh=False)
        live.start()
        update_display(progress, layout, np.inf, np.inf, 0.0, scheduler.get_last_lr()[0], 0.0, "", "")
        live.refresh()

    
    

    # initialize metrices
    best_accuracy = 0.0
    best_val_loss = np.inf
    train_losses = []
    val_losses = []
    accuracies = []
    lrs = [] 




    



    for epoch in range(args.epochs):
        # Get learning rate for this epoch
        lr=scheduler.get_last_lr()[0]
        # Log current lr
        lrs.append(lr)

        # Train one epoch
        avg_train_loss = train_one_epoch(model, trainloader, optimizer, criterion, scheduler, args.grad_clip)
        # Log current train loss
        train_losses.append(avg_train_loss)


        # Validate one epoch
        avg_val_loss, accuracy, probs = validate(model, testloader, criterion)
        # Log validation loss and accurecy
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)

        # update small weights count
        cur_sparsity = model.append_small_weight_vec(args.small_weights_threshold, epoch)


        # wandb log
        if args.wandb_log:
            wandb.log({
                "epoch": epoch+1,
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
            best_accuracy_str = f"Best Validation Accuracy: {best_accuracy:.2f}%, achived at epoch {epoch+1} with {100*cur_sparsity:.1f}% sparsity"
            if args.verbose:
                console.print(best_accuracy_str)

            if args.save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'probs': probs,
                    'accuracy': accuracy,
                }, accuracy_save_path)
                if args.verbose:
                    console.print(f"Saved best accuracy model to {accuracy_save_path}")
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_str = f"Best Validation Loss: {best_val_loss:.4f}, achived at epoch {epoch+1} with {100*cur_sparsity:.1f}% sparsity"
            if args.verbose:
                console.print(best_val_loss_str)

            if args.save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'probs': probs,
                    'val_loss': avg_val_loss,
                }, loss_save_path)
                if args.verbose:
                    console.print(f"Saved best loss model to {loss_save_path}")
        
        
        

        # Update/print stats
        if args.verbose:
            console.print(f"Epoch: {epoch+1}/{args.epochs}\tTrain Loss: {avg_train_loss:.4f}\tTest Loss: {avg_val_loss:.4f}\tAccuracy: {accuracy:.2f}%\tCurrent LR: {lr:.5f}")
        if args.progress_bar:
            progress.update(task_id, advance=1)
            update_display(progress, layout, avg_train_loss, avg_val_loss, accuracy, lr, cur_sparsity, best_val_loss_str, best_accuracy_str)
            live.refresh()
    
    if args.progress_bar:
        live.stop()



    
    if args.verbose:
        console.print("Training completed.")
        console.print(best_accuracy_str)
        console.print(best_val_loss_str)



    if args.save_model:
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
