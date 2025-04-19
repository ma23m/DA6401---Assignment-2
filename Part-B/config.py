def get_sweep_config(sweep_name):  # function to build a W&B sweep config
    return {  # return the configuration dictionary
        'name': sweep_name,  # name of the sweep
        'method': 'bayes',  # use Bayesian optimization for hyperparameter search
        'metric': {  # define metric 
            'name': 'val_acc',  # validation accuracy metric name
            'goal': 'maximize'  # direction to optimize the metric
        },
        'parameters': {  # specify hyperparameters and their possible values
            'batch_size': {'values': [16, 32, 64, 128]},  # different batch sizes to try
            'freeze_up_to': {'values': [0, 3, 5, 7]},  # number of layers to freeze
            'epochs': {'values': [5, 7, 10, 12]},  # possible epoch counts
            'learning_rate': {'values': [0.1, 0.01, 0.001]}  # different learning rates
        }
    }  # end of config dictionary
