import wandb  # import W&B for experiment tracking
import argparse  # import argparse for command-line argument parsing
from train import train_model  # import the train_model function from train.py

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Run a W&B hyperparameter sweep")  # create argument parser with description
parser.add_argument('--project', type=str, default='DL_A2', help='W&B project name')  # project name argument
parser.add_argument('--sweep_name', type=str, default='scratch_hyperparam_sweep-1', help='Name of the sweep')  # sweep name argument
parser.add_argument('--sweep_count', type=int, default=100, help='Number of sweep runs to execute')  # sweep runs count
args = parser.parse_args()  # parse provided command-line arguments

# Define sweep configuration dictionary
sweep_config = {
    'name': args.sweep_name,  # assign sweep name from args
    'method': 'bayes',  # use Bayesian optimization method
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},  # metric to optimize
    'parameters': {  # specify hyperparameters and their possible values
        'num_filters': {  # convolution filter sizes
            'values': [  # list of possible filter size lists
                [16, 32, 64, 128, 256],
                [64, 64, 64, 64, 64],
                [256, 128, 64, 32, 16],
                [32, 32, 32, 32, 32],
                [16, 32, 64, 32, 16]
            ]
        },
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},  # activation functions to try
        'batch_norm': {'values': [True, False]},  # whether to use batch normalization
        'dropout_rate': {'values': [0.2, 0.3]},  # dropout rates to try
        'data_augmentation': {'values': [True, False]},  # toggle data augmentation
        'filter_organization': {'values': ['same', 'double', 'half']},  # filter organization strategies
        'kernel_size': {  # convolution kernel sizes
            'values': [  # list of kernel size lists
                [3, 3, 3, 3, 3],
                [3, 3, 5, 3, 3],
                [3, 5, 3, 5, 3],
                [5, 5, 5, 5, 5],
                [5, 7, 7, 3, 5]
            ]
        },
        'dense_neurons': {  # fully connected layer sizes
            'values': [  # list of neuron count lists
                [512], [256], [128],
                [64, 128, 256],
                [512, 256, 64]
            ]
        },
        'epochs': {'values': [5, 7, 10, 12, 15, 17, 20]}  # number of training epochs
    }
}

# Initialize and run the sweep
if __name__ == '__main__':  # only run when script executed directly
    sweep_id = wandb.sweep(sweep_config, project=args.project)  # create a new W&B sweep
    wandb.agent(sweep_id, function=train_model, count=args.sweep_count)  # launch agent to execute sweep runs

# Example usage:
# python sweep_runner.py --project DL_A2 --sweep_name MySweep --sweep_count 1  # run with custom args
