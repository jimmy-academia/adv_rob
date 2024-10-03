import datetime
from pathlib import Path

from utils import *
from networks import get_model
from datasets import get_dataloader
from train_env import get_trainer

# fix attack evaluation methods; adjust model and tranining
def set_arguments():
    parser = argparse.ArgumentParser(description='Run experiments')
    # base
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='gpu device id')
    
    # main decisions
    parser.add_argument('--model', type=str, default='tttbasic', choices=['mobilenet', 'mobileapt', 'tttbasic'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--train_env', type=str, default='TTT', choices=['AT', 'AST', 'TTT'])
    parser.add_argument('--test_time', type=str, default='corrupt', choices=['none', 'corrupt'])
    
    # detail train/attack decisions
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eps', type=int, default=None)
    parser.add_argument('--attack_iters', type=int, default=50, help='iter for eval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval during train')
    parser.add_argument('--corrupt_level', type=int, default=5)
    parser.add_argument('--corrupt_type', type=str, default='gaussian_noise', choices=common_corruptions+['all'])

    # detail model decisions (iptnet)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=1024)
    
    # train record path or notes 
    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--record_path', type=str, default=None)
    
    
    args = parser.parse_args()

    # post parsing adjustments
    set_seeds(args.seed)
    args.device = torch.device(f"cuda:{args.device}" if int(args.device) >= 0 and torch.cuda.is_available() else "cpu")
    args.ckpt = Path(args.ckpt)
    args.ckpt.mkdir(exist_ok=True)
    if args.record_path is None:
        mmdd = datetime.datetime.now().strftime("%m%d")
        record_path = f'{args.train_env}-{args.model}-{args.dataset}-{mmdd}'
        count = len(list(args.ckpt.glob(f"{record_path}*")))
        suffix = f'_{count+1}' if count >= 1 else ''
        args.record_path = args.ckpt/(record_path+suffix)
    else:
        args.record_path = args.ckpt/args.record_path

    ## attacks
    if args.eps is None:
        args.eps = 0.3 if args.dataset == 'mnist' else 8/255

    ## model dimensions
    args.channels = 3
    args.image_size = 32 if args.dataset != 'imagenet' else 256
    args.num_classes = 10 if args.dataset not in ['imagenet', 'cifar100'] else (100 if args.dataset == 'cifar100' else 1000)

    return args

def main():
    args = set_arguments()

    print(f'=== running experiment: {args.train_env} {args.model} {args.dataset} ===')

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
    dumpj({'param info': param_msg, 'training_records':trainer.training_records, ' eval_records':trainer.eval_records, 'arguments':vars(args)}, args.record_path)
    torch.save(trainer.model.state_dict(), self.record_path.with_suffix('.pth'))
    
if __name__ == '__main__':
    main()