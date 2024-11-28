import copy
import datetime
from pathlib import Path

from utils import *
from networks import get_model
from datasets import get_dataloader
from train_env import get_trainer

from debug import *

import argparse

# fix attack evaluation methods; adjust model and tranining
def set_arguments():
    parser = argparse.ArgumentParser(description='Run experiments')
    # environment
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='gpu device id, use -1 for cpu')
    
    # logging decisions
    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--task', type=str, default='serial', help='store in task file or serial attempts')
    parser.add_argument('--record_path_suffix', type=str, default='')

    # main decisions
    # --model choices=['mobilenet', 'mobilenet_apt', 'tttbasic', 'resnetcifar', 'resnetcifar_apt', 'resnetcifar_afa']
    parser.add_argument('--model', type=str, default='resnetcifar_zlqh')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--train_env', type=str, default='ZLQH', choices=['AT', 'AST', 'ALT', 'AFA', 'ZLQH', 'ZLQH_dir', 'TTT', 'TTAdv'])
    parser.add_argument('--attack_type', type=str, default='aa', choices=['aa', 'pgd'])

    # test time settings
    parser.add_argument('--test_time', type=str, default='none', choices=['none', 'standard', 'online'])
    parser.add_argument('--test_domain', type=str, default='corrupt', choices=['corrupt'])
    parser.add_argument('--corrupt_level', type=int, default=5)
    parser.add_argument('--corrupt_type', type=str, default='gaussian_noise', choices=common_corruptions+['all'])
    parser.add_argument('--test_time_iter', type=int, default=1) # 1,3,10

    # detail train/attack decisions
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eps', type=int, default=None)
    parser.add_argument('--attack_iters', type=int, default=50, help='iter for eval')
    parser.add_argument('--eval_interval', type=int, default=1, help='eval during train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--step_size', type=float, default=10, help='lr schedule epoch steps')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr schedule reduce rate')
    
    # detail model decisions (iptnet)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=128)
    parser.add_argument('--direct', action='store_true', help='directly predict high order noise')
    
    args = parser.parse_args()
    return args

def post_process_args(args):
    # environment
    set_seeds(args.seed)
    if args.device > torch.cuda.device_count():
        input(f'Warning: args.device {args.device} > device count {torch.cuda.device_count()}. Set to default index 0. Press anything to continue.')
        args.device = 0
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    # logging decisions
    args.ckpt = Path(args.ckpt)/args.task
    args.ckpt.mkdir(parents=True, exist_ok=True)
    record_path = f'{args.train_env}_{args.model}_{args.dataset}' \
        + args.record_path_suffix
    ## do more adjustments to record_path below if needed
    if args.task == 'serial':
        record_path += f'_{datetime.datetime.now().strftime("%b%d-%H")}'
        count = len(list(args.ckpt.glob(f"{record_path}*")))
        record_path += f'_{count+1}' if count >= 1 else ''

    args.record_path = args.ckpt/record_path

    ## attacks
    if args.eps is None:
        args.eps = 0.3 if args.dataset == 'mnist' else 8/255

    ## model dimensions
    args.channels = 3
    args.image_size = 32 if args.dataset != 'imagenet' else 256
    args.num_classes = 10 if args.dataset not in ['imagenet', 'cifar100'] else (100 if args.dataset == 'cifar100' else 1000)

    args.lr = float(args.lr)
    return args

def main():

    args = set_arguments()
    args = post_process_args(args)
    string_args = {key: str(value) for key, value in vars(args).items()}

    print()
    print(f'=== running experiment: {args.train_env} {args.model} {args.dataset} ===')
    print(string_args)
    print(f' >>> to save results to {args.record_path}')

    # Initialize components based on config
    model = get_model(args)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Num, whatB = params_to_memory(num_parameters)
    param_msg = f'model param count: {num_parameters} ≈ {Num}{whatB}'
    print(f'>>> {param_msg}')
    # additional information for iptnet
    if 'ipt' in args.model or 'apt' in args.model:
        print(f'>>>> ipt config: vocab_size = {args.vocab_size}, patch_size={args.patch_size}')

    train_loader, test_loader = get_dataloader(args)
    trainer = get_trainer(args, model, train_loader, test_loader)
    # Run experiment
    trainer.train()
    trainer.eval()
    dumpj({'param info': param_msg, 'training_records':trainer.training_records, ' eval_records':trainer.eval_records, 'arguments':string_args}, args.record_path.with_suffix('.json'))
    torch.save(trainer.model.state_dict(), args.record_path.with_suffix('.pth'))
    
if __name__ == '__main__':

    if os.getenv("IS_SUBPROCESS") != "1":
        msg = """
            working on:
            script/main_exp.py for main experiments, 
            script/abalation.py for ablation experiments
            future .... script/qualitative.py ....

            ===

            Next ==> 

            A. converge current results
            a. consolidate ipt, afa, .... into ipt with argument variants
            b. logging and printing result!
            c. code sensitivity tests

            B. increase realm = test time training
        """
        print(msg)
        input('>stop<')

    main()