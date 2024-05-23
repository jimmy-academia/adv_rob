import torch
from parse import set_arguments
from utils import set_seeds
from data import get_dataset, get_dataloader, tokenize_dataset
from networks import IPTResnet
from train import advtrain_tokenizer, train_classifier, test_attack

def main():



    print('''
        does patch robustness correlate to adversarial robustness?
        what is the required parameter for good classification?
        ...
    ''')
    input()

    
    args = set_arguments()
    set_seeds(args.seed)
    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

    iptresnet = IPTResnet(args).to(args.device)

    if not args.tokenizer_path.exists():
        advtrain_tokenizer(args, iptresnet.tokenizer, train_loader, test_loader)
        torch.save(iptresnet.tokenizer.state_dict(), args.tokenizer_path)
    else:
        iptresnet.tokenizer.load_state_dict(torch.load(args.tokenizer_path))

    tok_train_set = tokenize_dataset(train_loader, iptresnet.tokenizer, args.patch_size, args.device)
    tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)
    
    train_classifier(args, iptresnet, tok_train_loader, test_loader)
    torch.save(iptresnet.state_dict(), args.classifier_path)

if __name__ == '__main__':
    main()