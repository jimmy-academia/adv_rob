import sys
sys.path.append('.')
import torch
import logging
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

from utils import set_verbose, run_command, loadj, dumpj, convert_args
from attacks import conduct_attack
from networks import get_model
from datasets import get_dataloader

from debug import *

TASK = 'main_exp'
output_dir = Path('ckpt/output/')
output_dir.mkdir(parents=True, exist_ok=True)
Record_path = output_dir/f'{TASK}_epoch_record.json'

model_list = ['lenet'] #, 'efficientnet', 'mobilenet', 'resnet4'
train_env_list = ['AT', 'AST'] 
dataset_list = ['mnist'] #, 'cifar10'

def run_experiments():
    set_verbose(1)
    evaluate_the_models()

def evaluate_the_models():

    Record = defaultdict(list)
    # Loop over all combinations of configurations
    for dataset in dataset_list:
        specs = SimpleNamespace()
        specs.dataset = dataset
        specs.batch_size = 128
        specs.image_size = 32 if dataset != 'imagenet' else 256
        specs.test_time = 'none'

        __, test_loader = get_dataloader(specs)
        for model_name in model_list:
            for train_env in train_env_list:

                instance_label = f'{train_env}_{model_name}_{dataset}'

                logging.info(f'evaluating {instance_label}')
                result_path = Path(f'ckpt/{TASK}/{instance_label}')
                instance_info = loadj(result_path.with_suffix('.json'))
                args = instance_info.get('arguments')
                args = SimpleNamespace(**convert_args(args))
                model = get_model(args)

                for epoch in range(1, 50):
                    weight_path = result_path.with_suffix(f'.pth.{epoch}')
                    model.load_state_dict(torch.load(weight_path, weights_only=True))
                    for attack_type in ['fgsm', 'pgd']:
                        args.attack_type = attack_type
                        results = conduct_attack(args, model, test_loader, multi_pgd=True, do_test = False)
                        test_correct, adv_correct, total = results


                        if attack_type == 'pgd':
                            for iter_, a in zip(['10', '20', '50'], adv_correct):
                                Record[instance_label + '_pgd'+iter_].append(a/total)
                        else:
                            Record[instance_label+'_fgsm'].append(adv_correct/total)

                dumpj(Record, Record_path)

if __name__ == '__main__':
    run_experiments()