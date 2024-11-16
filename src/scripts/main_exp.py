import subprocess
# Define the different configurations
models = ['mobilenet', 'resnetcifar', 'resnetcifar_apt']
train_envs = ['AT', 'AFA', 'ZLQH']
datasets = ['cifar10']
attack_types = ['aa', 'pgd']

# Directory to save logs and results
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
                run_command(cmd)


def run_command(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")