import torch
from tqdm import tqdm
from train_env.advtraining import AdversarialTrainer

def get_trainer(args, model, train_loader, test_loader):
    if args.train_env == 'AT':
        return AdversarialTrainer(args, model, train_loader, test_loader)

