import sys
sys.path.append('.')
from utils import run_command

from printer import display_images_in_grid


def run_experiments():

    models = ['mobilenet', 'resnetcifar', 'resnetcifar_apt']
    train_envs = ['AT', 'AST']
    datasets = ['cifar10']
    attack_types = ['aa', 'pgd']

    log_dir = "experiment_logs"

    # Loop over all combinations of configurations
    for model in models:
        for train_env in train_envs:
            for dataset in datasets:
                for attack_type in attack_types:
                    # Build the command for the experiment
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

def print_experiments():
    print('okay')
    pass

if __name__ == '__main__':
    run_experiments()