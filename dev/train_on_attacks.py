import sys
sys.path.append('.')

import torch
import yaml
from config import config_arguments
from utils import set_seeds, writef

from ipt.data import get_dataset, get_dataloader, tokenize_dataset
from ipt.networks import retrieve_ipt_model
from ipt.train.two_stage import train_tokenizer, train_classifier, test_attack

from ipt.attacks import pgd_attack, square_attack

def main():
    with open('dev/train_on_attacks.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args = config_arguments(config)
    set_seeds(args.seed)

    if args.ckpt_dir.exists():
        input(f'{args.ckpt_dir} already exists. Press any key to continue')
    else:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    for attack_type, _attack in zip(['square', 'pgd'], [square_attack, pgd_attack]):

        args.attack_type = attack_type
        args.bpda = attack_type == 'pgd'

        tok_weight_path = args.ckpt_dir/f'ipt_{attack_type}_{args.dataset}_{args.patch_size}_{args.vocab_size}.pth'
        weight_path = args.ckpt_dir/f'iptat_{attack_type}_{args.dataset}_{args.patch_size}_{args.vocab_size}.pth'
        message_path = args.ckpt_dir/f'log_{attack_type}_{args.dataset}_{args.patch_size}_{args.vocab_size}.txt'
        iptresnet = retrieve_ipt_model(args)

        train_set, test_set = get_dataset(args.dataset)
        train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

        if tok_weight_path.exists():
            iptresnet.tokenizer.load_state_dict(torch.load(tok_weight_path))
        else:
            iptresnet = train_tokenizer(args, iptresnet, train_loader, _attack)
            torch.save(iptresnet.tokenizer.cpu().state_dict(), tok_weight_path)
            
        tok_train_set = tokenize_dataset(train_loader, iptresnet, args.device)
        tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)
        message = train_classifier(args, iptresnet, tok_train_loader, test_loader, _attack)
        torch.save(iptresnet.cpu().state_dict(), weight_path)
        writef(message_path, message)

if __name__ == '__main__':  
    main()

