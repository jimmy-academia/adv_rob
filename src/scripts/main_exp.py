import sys
sys.path.append('.')
from utils import run_command

from printer import display_images_in_grid


def run_experiments():
    models = ['lenet', 'efficientnet', 'mobilenet', 'resnet4']
    train_envs = ['AST', 'AT'] 
    datasets = ['mnist', 'cifar10']

    train_the_models(models, train_env, datasets)
    evaluate_the_models(models, train_env, datasets)

def train_the_models(models, train_env, datasets):

    attack_type = 'none'

    log_dir = "experiment_logs"

    # Loop over all combinations of configurations
    for dataset in datasets:
        for model in models:
            for train_env in train_envs:

                print(''' 
                    do logging => print to console
                    check what to record in experiment_logs?
                    if exists do not overwrite
                    evalaute and print result later
                    ''')
                input('>> stop <<')

                cmd = [
                    "python", "main.py",
                    "--model", model,
                    "--train_env", train_env,
                    "--dataset", dataset,
                    "--attack_type", attack_type,
                    "--ckpt", log_dir
                ]

                # Run the command
                run_command(cmd, shell=False)


def evaluate_the_models(models, train_env, datasets):

    attack_type = 'none'

    log_dir = "experiment_logs"

    # Loop over all combinations of configurations
    for dataset in datasets:
        for model in models:
            for train_env in train_envs:

                print(''' 
                    load the model
                    ''')
                input('>> stop <<')

                cmd = [
                    "python", "main.py",
                    "--model", model,
                    "--train_env", train_env,
                    "--dataset", dataset,
                    "--attack_type", attack_type,
                    "--ckpt", log_dir
                ]

                # Run the command
                run_command(cmd, shell=False)

    # evaluate and print result

def print_experiments():
    print('okay')
    pass

if __name__ == '__main__':
    run_experiments()