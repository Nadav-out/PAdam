# PAdam

PAdam is an extension of the Adam optimizer that allows for any p-norm regularization.

## Blog Post

For more details on the concepts and motivations behind PAdam, please refer to my [blog post](https://nadav-out.github.io/posts/PAdam/).



## Current Implementation

- The current version focuses on the simplest form of the PAdam optimizer, which I refer to as the 'adiabatic' PAdam optimizer.
- It is currently applied here only to the `FashionMNIST` dataset for demonstration purposes.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:
- PyTorch
- torchvision
- NumPy
- Matplotlib
- seaborn

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
    python ./python/fashion_train.py [optional arguments]
    ```
### Optional Arguments

You can customize the training by specifying the following arguments:
- `--device`: Set the device for training (default: `cuda:0`, on a Mac with apple silicone, use `mps`).
- `--batch_size`: Batch size for training (default: `256`).
- `--seed`: Seed for reproducibility (default: `2349`).
- `--epochs`: Number of training epochs (default: `400`).
- `--lr_1`, `--lr_2`: Learning rates for the models (default: `3e-3`).
- `--weight_decay`: Weight decay for AdamW optimizer (default: `1e-1`).
- `--lambda_p`: Lambda parameter for PAdam optimizer (default: `3e-3`).
- `--p_norm`: p-norm for PAdam optimizer (default: `0.8`).
- `--scheduler_exponent`: Exponent for LR scheduler decay rate (default: `0.6`).
- `--save_dir`: Directory to save trained models and metrics (default: `../data`).
