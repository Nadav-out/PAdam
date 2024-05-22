# pWD and pAdam

An example use case for the pAdam optimizer, for training ResNet18 on CIFAR10.





### Requirements

Ensure you have the following dependencies installed:
- PyTorch
- torchvision
- NumPy
- pickle
- pandas
- rich

### Running the Training Script

1. Clone the repository:
   ```bash
   git clone https://github.com/Nadav-out/PAdam
   ```
2. Navigate to the project directory:
    ```bash
    cd PAdam
    ```
3. Run the training script:
    ```bash
    python python/train_CIFAR10.py [optional arguments]
    ```


### Optional Arguments

- `--progress_bar`: Enable a rich live layout progress bar for visual feedback during training.
- `--verbose`: Enable verbose output for detailed logging information.
- `--absolute_paths`: Use absolute paths for data and output directories.
- `--data_dir <path>`: Set the directory for data (default: `../data/CIFAR10`).
- `--out_dir <path>`: Specify the output directory for results (default: `../results/CIFAR10`).
- `--save_checkpoints`: Save checkpoints during training to allow resuming.
- `--save_summary`: Save a comprehensive summary of the training session, including training and validation losses, accuracies, learning rates, and model sparsity details.
- `--num_workers <number>`: Number of workers for data loading (default: `4`).
- `--batch_size <size>`: Batch size for training (default: `400`).
- `--epochs <number>`: Number of training epochs (default: `100`).
- `--optimizer_name <name>`: Name of the optimizer to use (`AdamW` or `PAdam`, default: `PAdam`).
- `--max_lr <rate>`: Maximum learning rate (default: `1e-3`).
- `--lambda_p <value>`: Lambda parameter value for PAdam optimizer (default: `1e-3`).
- `--p_norm <value>`: P-norm value for PAdam optimizer (default: `0.8`).
- `--beta1 <value>`: Beta1 for Adam optimizer (default: `0.9`).
- `--beta2 <value>`: Beta2 for Adam optimizer (default: `0.999`).
- `--grad_clip <value>`: Gradient clipping value (default: `0.0`).
- `--warmup_epochs <number>`: Number of warmup epochs, can be fractional (default: `2`).
- `--lr_decay_frac <fraction>`: Fraction of max_lr to decay to (default: `1.0`).
- `--min_lr <rate>`: Minimum learning rate (default: `1e-5`).
- `--small_weights_threshold <value>`: Threshold for considering weights as small (default: `1e-13`).
- `--device <device>`: Device to use for training (`cuda` or `cpu`, default: `cuda`).
- `--compile`: (requires PyTorch >2.0) compiles the model for faster training.
- `--wandb_log`: Enable logging to Weights & Biases for experiment tracking.
- `--wandb_project <name>`: Specify the Weights & Biases project name (default: `PAdam`).
- `--wandb_run_name <name>`: Set the Weights & Biases run name (default includes `ResNet18` with a timestamp).
- `--wandb_group <name>`: Group related runs in Weights & Biases.


For detailed explanations and additional parameters, refer to the argument parser setup in the training script.

