import torch
from arguments import set_arguments
from utils import set_seeds
from data import get_dataset, get_dataloader, tokenize_dataset
from networks import Tokenizer, TokClassifier
from train import train_tokenizer, test_tokenizer, train_classifier, attack_classifer


def main():
    args = set_arguments()
    set_seeds(args.seed)
    print('gogo mnist')

    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, 128)

    tokenizer = Tokenizer(args.patch_size**2, args.vocabulary_size, args.num_hidden_layer).to(args.device)
    if not args.tokenizer_path.exists():
        train_tokenizer(args, tokenizer, train_loader, test_loader)
    
    tokenizer.load_state_dict(torch.load(args.tokenizer_path))
    correct, total = test_tokenizer(args, tokenizer, test_loader)
    print(f'tokenizer attack test accuracy: {correct/total:.4f}')
        
    #train model with fixed tokenizer
    classifer = TokClassifier(args).to(args.device)
    if not args.classifier_path.exists():
        tok_train_set = tokenize_dataset(train_loader, tokenizer, args.patch_size, args.device)
        tok_test_set = tokenize_dataset(test_loader, tokenizer, args.patch_size, args.device)
        tok_train_loader, tok_test_loader = get_dataloader(tok_train_set, tok_test_set)
        train_classifier(args, classifer, tok_train_loader, tok_test_loader, test_loader)

    # final adversarial evaluation
    classifer.load_state_dict(torch.load(args.classifier_path))
    classifer.tokenizer.load_state_dict(torch.load(args.tokenizer_path))
    classifer.to(args.device)
    attack_classifer(args, classifer, test_loader)
    

    

if __name__ == '__main__':
    main()
