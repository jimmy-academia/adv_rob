import torch
from tqdm import tqdm
from train_env.at import AdversarialTrainer
from train_env.ast import AdversarialSimilarityTrainer
from train_env.ttt import TestTimeTrainer, TestTimeAdvTrainer

def get_trainer(args, model, train_loader, test_loader):
    if args.train_env == 'AT':
        return AdversarialTrainer(args, model, train_loader, test_loader)
    elif args.train_env == 'AST':
        return AdversarialSimilarityTrainer(args, model, train_loader, test_loader)
    elif args.train_env == 'TTT':
        return TestTimeTrainer(args, model, train_loader, test_loader)
    elif args.train_env == 'TTAdv':
        return TestTimeAdvTrainer(args, model, train_loader, test_loader)
    else:
        raise NotImplementedError(f"train_env {args.train_env} not implemented")

