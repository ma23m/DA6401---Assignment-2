import argparse  # For parsing command-line arguments
import wandb  # For experiment tracking and hyperparameter sweeps
from config import get_sweep_config  # Function to get sweep configuration
from train_utils import train  # Import the training function

# Function to define and parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 with hyperparameter sweep")

    # Argument for setting the WandB project name
    parser.add_argument('--project_name', type=str, default='DL_A2', help='WandB project name')
    
    # Argument for naming the sweep
    parser.add_argument('--sweep_name', type=str, default='resnet50_hyperparam_sweep', help='WandB sweep name')
    
    # Argument to specify dataset root directory
    parser.add_argument('--data_path', type=str, default='/kaggle/input/inaturalist-12/inaturalist_12K', help='Root path to dataset')
    
    # Argument to set how many runs the sweep will perform
    parser.add_argument('--sweep_count', type=int, default=50, help='Number of sweep runs')
    
    # Flag to indicate whether to use GPU if available
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')

    # Return parsed arguments
    return parser.parse_args()

# Run only if the script is executed directly
if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    sweep_config = get_sweep_config(args.sweep_name)  # Get sweep config based on sweep name
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)  # Initialize a sweep on wandb

    # Start the sweep agent which will run the training function multiple times
    wandb.agent(sweep_id, function=lambda: train(args), count=args.sweep_count)


# Example command to run the script:
# python train.py --project_name "DL_A2" --sweep_name "resnet50_exp2_sweep" --data_path "C:\Users\HP\Downloads\nature_12K" --sweep_count 1 --use_cuda
