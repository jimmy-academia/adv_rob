import copy
import datetime
from pathlib import Path

from utils import *
from networks import get_model
from datasets import get_dataloader
from train_env import get_trainer

import argparse

def set_arguments():
    parser = argparse.ArgumentParser(description='Run experiments')
    
    # environment
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--device', type=int, default=0, help='gpu device id, use -1 for cpu')
    
    # logging decisions
    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--task', type=str, default='serial', help='store in task file or serial attempts')

    # model decisions
    parser.add_argument('--model', type=str, default='resnet4')
    # detail model decisions (iptnet)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=128)
    parser.add_argument('--tok_ablation', type=str, default='zlt', choices=['opt', 'pzt', 'zlt', 'zlqt'])
    parser.add_argument('--direct', action='store_true', help='no token; directly predict high order noise')
    parser.add_argument('--joint_train', action='store_true', help='no dual steps, train embedding and predictor together')
    parser.add_argument('--lambda1', type=float, default=0.3)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.03)

    # Dataset 
    parser.add_argument('--dataset', type=str, default='cifar10') 
    parser.add_argument('--percent_val_split', type=int, default=10) 
    # Adv_training, attack_type
    parser.add_argument('--train_env', type=str, default='AST')
    parser.add_argument('--attack_type', type=str, default='pgd')

    # detail train/attack decisions
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eps', type=int, default=None)
    parser.add_argument('--attack_iters', type=int, default=50, help='iter for eval')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--step_size', type=float, default=10, help='lr schedule epoch steps')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr schedule reduce rate')
    
    # periodic check and save
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=-1)

    # prevent overwrite for script/tasks when not set
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    return args

def post_process_args(args):
    # environment
    set_seeds(args.seed)
    set_verbose(args.verbose)
    if args.device > torch.cuda.device_count():
        input(f'Warning: args.device {args.device} > device count {torch.cuda.device_count()}. Set to default index 0. Press anything to continue.')
        args.device = 0
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    # save path decisions
    # if python main.py => ckpt/serial/AT_resnet4_mnist_1207-12_3
    # if python script/main_exp.py => ckpt/main_exp/AT_resnet4_mnist
    args.ckpt = Path(args.ckpt)/args.task
    args.ckpt.mkdir(parents=True, exist_ok=True)
    record_path = f'{args.train_env}_{args.model}_{args.dataset}'

    if args.task == 'serial':
        record_path += f'_{datetime.datetime.now().strftime("%b%d-%H")}'
        count = len(list(args.ckpt.glob(f"{record_path}*")))
        record_path += f'_{count+1}' if count >= 1 else ''

    args.record_path = args.ckpt/record_path

    ## adjust model name
    if args.train_env == 'AST':
        args.model += '_ipt'

    ## model dimensions
    args.channels = 3
    args.image_size = 32 if args.dataset != 'imagenet' else 256
    args.num_classes = 10 if args.dataset not in ['imagenet', 'cifar100'] else (100 if args.dataset == 'cifar100' else 1000)

    args.lr = float(args.lr)

    ## training/attacking
    bs_list = {'mnist': 512, 'imagenet': 128}
    if args.batch_size is None:
        args.batch_size = bs_list[args.dataset] if args.dataset in bs_list else 256
    
    if args.eps is None:
        args.eps = 0.3 if args.dataset == 'mnist' else 8/255
    
    return args

def main():

    args = set_arguments()
    args = post_process_args(args)
    
    logging.info(f'== running exp: {args.train_env} on {args.model} for {args.dataset} ==> {args.record_path}')

    # if args.return_args:
        # return args

    if args.train_env == 'AST':
        logging.info(f'[IPT info]: {args.patch_size}x{args.patch_size} patch, T={args.vocab_size}; Tok_ablation={args.tok_ablation}, direct={args.direct}, joint train={args.joint_train}')
    
    model = get_model(args)
    train_loader, valid_loader, test_loader = get_dataloader(args)
    trainer = get_trainer(args, model, train_loader, valid_loader, test_loader)
    
    trainer_settings = '''
    trainer.train() will 
    1. based on argument settings:
        - save model after every args.save_interval epochs
        - check every eval_interval (detailed eval_records)

    2. always
        - save final epoch
        # - save the full train_record
    '''
    logging.debug(trainer_settings)
    trainer.train()

    string_args = {key: str(value) for key, value in vars(args).items()}
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Num, whatB = params_to_memory(num_parameters)
    param_msg = f'model param count: {num_parameters} ≈ {Num}{whatB}'
    dumpj({'arguments':string_args, 'param_msg': param_msg, 'param_items':[num_parameters, Num, whatB], 'training_records':trainer.training_records, ' eval_records':trainer.eval_records}, args.record_path.with_suffix('.json'))
    logging.info(f'......saved model, train_log to {args.record_path}.pth, .json')

if __name__ == '__main__':

    if os.getenv("IS_SUBPROCESS") != "1":
        msg = """
            Next ==> 

            A. converge current results
                a. logging and printing result!
                b. code (check) script/main_exp
                c. code script/ablation_exp.py
                d. code script/sensitivity_exp.py
                e. run all code

            B. increase realm = test time training
        """
        print(msg)
        input('>stop<')

    main()
