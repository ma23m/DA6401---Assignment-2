import wandb  # import the Weights & Biases library for experiment tracking

def login():  # define a function to handle user authentication
    key = input('Enter your WandB API key: ')  # ask user to input their API key
    wandb.login(key=key)  # log in to W&B using the provided key

if __name__ == "__main__":  # check if script is run directly
    login()  # execute the login function
