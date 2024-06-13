import sys
sys.path.append('.')

import torch
import yaml
from config import config_arguments
from utils import set_seeds, awritef

from ipt.data import get_dataset, get_dataloader, tokenize_dataset
from ipt.networks import retrieve_ipt_model, Decoder
from ipt.train.iter_auto import autoenc_adv_training
# from ipt.train.two_stage import train_classifier

# from ipt.attacks import square_attack

def main():
    with open('dev/train_auto_encode.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args = config_arguments(config)
    set_seeds(args.seed)

    if args.ckpt_dir.exists():
        input(f'{args.ckpt_dir} already exists. Press any key to continue')
    else:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    iptresnet = retrieve_ipt_model(args)
    decoder = Decoder(args)

    args.bpda = False
    tau = args.config['train']['tau']
    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

    for epoch in range(args.config['train']['toktrain_epochs']):
        print(f'======= epoch {epoch} =======')
        weight_path = args.ckpt_dir/f'iptat_{args.dataset}_{args.patch_size}_{args.vocab_size}_{epoch}.pth'
        message_path = args.ckpt_dir/f'log_{args.dataset}_{args.patch_size}_{args.vocab_size}.txt'
        
        autoenc_adv_training(args, iptresnet, decoder, train_loader, test_loader, tau)
        tau *= 2
        torch.save(iptresnet.cpu().state_dict(), weight_path)

        # tok_train_set = tokenize_dataset(train_loader, iptresnet, args.device)
        # tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)
        # message = train_classifier(args, iptresnet, tok_train_loader, test_loader, square_attack)
        # print(message)
        # awritef(message_path, message)
        # torch.save(iptresnet.cpu().state_dict(), weight_path)

if __name__ == '__main__':  
    main()

