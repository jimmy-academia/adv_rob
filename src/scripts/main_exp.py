import sys
sys.path.append('.')
import torch
import logging
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

from utils import set_verbose, run_command, loadj, dumpj, convert_args, model_to_kb
from attacks import conduct_attack
from networks import get_model
from datasets import get_dataloader

from debug import *

TASK = 'main_exp'
output_dir = Path('ckpt/output/')
output_dir.mkdir(parents=True, exist_ok=True)
Record_path = output_dir/f'{TASK}_record.json'

model_list = ['lenet', 'mobilenet', 'resnet4', 'efficientnet']
train_env_list = ['AT', 'AST'] 
dataset_list = ['mnist', 'cifar10'] #'cifar100'

def run_experiments():
    set_verbose(1)
    train_the_models()
    evaluate_the_models()

def train_the_models():

    # Loop over all combinations of configurations
    for dataset in dataset_list:
        for model_name in model_list:
            for train_env in train_env_list:
                cmd = [
                    "python", "main.py",
                    "--model", model_name,
                    "--train_env", train_env,
                    "--dataset", dataset,
                    "--eval_interval", str(-1),
                    "--save_interval", str(1),
                    "--task", TASK
                ]

                result_path = f'ckpt/{TASK}/{train_env}_{model_name}_{dataset}.pth'
                if Path(result_path).exists():
                    logging.info(f'{result_path} exists_______|')
                else:
                    # Run the command
                    run_command(cmd, shell=False)

def _early_stopping(val_loss_list, patience = 10):
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    for i, val_loss in enumerate(val_loss_list):
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = i+1
        else:
            counter += 1
            if counter >= patience:
                break

    return best_epoch

def evaluate_the_models():

    Record = defaultdict(list)
    # Loop over all combinations of configurations
    for dataset in dataset_list:
        specs = SimpleNamespace()
        specs.dataset = dataset
        specs.batch_size = 128
        specs.image_size = 32 if dataset != 'imagenet' else 256
        specs.test_time = 'none'
        specs.percent_val_split = 10

        __, __, test_loader = get_dataloader(specs)
        for model_name in model_list:
            for train_env in train_env_list:

                _instance = f'{train_env}_{model_name}_{dataset}'

                logging.info(f'evaluating {_instance}')
                result_path = Path(f'ckpt/{TASK}/{_instance}')
                instance_info = loadj(result_path.with_suffix('.json'))

                args = instance_info.get('arguments')
                args = SimpleNamespace(**convert_args(args))
                # __, Num, whatB = instance_info.get('param_items')
                # Record[_instance].append(f'{Num}{whatB}')


                ## AST use early stopping
                training_records = instance_info.get('training_records')
                _epoch = args.num_epochs
                if train_env == 'AST':
                    _epoch = _early_stopping(training_records.get('val_loss'))
                    logging.info(f'early stoping at epoch {_epoch}')

                
                _suffix = '' if _epoch == args.num_epochs else f'.{_epoch}'
                weight_path = result_path.with_suffix('.pth'+_suffix)

                model = get_model(args)
                model.load_state_dict(torch.load(weight_path, weights_only=True))
                model.eval()


                Record[_instance].append(f'{model_to_kb(model)}KB')
                Record[_instance].append(f'epoch:{_epoch}')

                done_test = False # do test once for each instance
                attack_results = []
                for attack_type in ['fgsm', 'pgd', 'aa']:
                    args.attack_type = attack_type
                    test_correct, adv_correct, total = conduct_attack(args, model, test_loader, multi=True, do_test = not done_test)

                    if not done_test:
                        done_test=True
                        Record[_instance].append(test_correct/total)
                    if attack_type in ['pgd', 'aa_list']:
                        attack_results += [a/total for a in adv_correct]
                    else:
                        attack_results.append(adv_correct/total)

                Record[_instance] += attack_results
                Record[_instance].append(min(attack_results))
                Record[_instance].append(training_records.get('runtime')[_epoch-1])

                dumpj(Record, Record_path)


#############################################
####### USAGE: python printer.py main #######
#############################################

Model_Name_List = ["LeNet", "EfficientNet", "MobileNet", "ResNet_tiny"]

def print_experiments():
    # Loop over all combinations of configurations
    Record = loadj(Record_path)
    for dataset in dataset_list:
        eps = "0.3" if dataset == 'mnist' else "8/255"
        print("\\midrule")
        print("\\multirow{10}{*}{\\shortstack{\\textbf{"+dataset.upper()+"}\\\\($\\epsilon="+eps+"$)}} ")

        for model_name in model_list:
            for train_env in train_env_list:
                _instance = f'{train_env}_{model_name}_{dataset}'
                inst_record = Record.get(_instance)
                inst_record = [str(x) for x in inst_record]

                model_label = Model_Name_List[model_list.index(model_name)]
                plusipt = " + IPT" if train_env == 'AST' else ""
                latex_line = f"& {model_label}{plusipt} ({inst_record[0]}) & {train_env} & " + " & ".join(inst_record[1:]) + " \\\\"
                print(latex_line)
            print("\\cmidrule(lr){2-11}")


if __name__ == '__main__':
    run_experiments()