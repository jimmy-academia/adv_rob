import sys
sys.path.append('.')
from utils import run_command

TASK = 'main_exp'

def run_experiments():
    models = ['lenet', 'efficientnet', 'mobilenet', 'resnet4']
    train_envs = ['AST', 'AT'] 
    datasets = ['mnist', 'cifar10']

    train_the_models(models, train_envs, datasets)
    evaluate_the_models(models, train_envs, datasets)

def train_the_models(models, train_envs, datasets):

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
                    "--eval_interval" 10**10,
                    "--save_interval" 1,
                    "--task", TASK
                ]

                # Run the command
                run_command(cmd, shell=False)


def evaluate_the_models(models, train_envs, datasets):

    # Loop over all combinations of configurations
    for dataset in datasets:
        specs = SimpleNamespace()
        specs.dataset = dataset
        specs.image_size = 32 if dataset != 'imagenet' else 256
        specs.test_time = 'none'

        __, test_loader = get_dataloader(specs)
        for model in models:
            for train_env in train_envs:
                record_path = Path(f'ckpt/{TASK}/{train_env}_{model}_{dataset}')

                # load last epoch model
                weight_path = record_path.with_suffix('.pth')
                arg_path = record_path.with_suffix('.json')
                args = loadj(arg_path).get('arguments')
                model = get_model(args)
                model.load_state_dict(torch.load(weight_path, weigths_only=True))

                for attack_type in ['fgsm', 'pgd', 'aa']:
                    args.attack_type = attack_type
                    results = conduct_attack(args, model, test_loader)
                    test_correct, adv_correct, total = results
                    if attack_type == 'pgd':
                        input('adv_correct is three numbers')

def print_experiments():
    print('okay')
    pass

if __name__ == '__main__':
    run_experiments()