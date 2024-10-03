import torch
from tqdm import tqdm
from train_env.advtraining import AdversarialTrainer
from train_env.simtraining import AdversarialSimilarityTrainer
from train_env.testtimetraining import TestTimeTrainer

def get_trainer(args, model, train_loader, test_loader):
    if args.train_env == 'AT':
        return AdversarialTrainer(args, model, train_loader, test_loader)
    if args.train_env == 'AST':
        return AdversarialSimilarityTrainer(args, model, train_loader, test_loader)
    if args.train_env == 'TTT':
        return TestTimeTrainer(args, model, train_loader, test_loader)
