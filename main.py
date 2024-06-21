import torch
import yaml
from config import config_arguments
from utils import set_seeds, check

from ipt.data import get_dataset, get_dataloader
from ipt.networks import build_ipt_model
from ipt.train import avg_patch_training, adv_patch_training, stable_training, train_classifier, test_attack, adversarial_training
from ipt.attacks import square_attack, pgd_attack

from tqdm import tqdm

def main():
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args = config_arguments(config)
    set_seeds(args.seed)
    iptresnet = build_ipt_model(args)

    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)


    avg_patch_training(args, iptresnet, train_loader)
    train_classifier(args, iptresnet, train_loader)
    
    for iter_ in range(30):
        # stable_training(args, iptresnet, train_loader)
        adversarial_training(args, iptresnet, train_loader)
        # adv_patch_training(args, iptresnet, train_loader)
        # correct, adv_correct, total = test_attack(args, iptresnet, test_loader, pgd_attack, True)
        correct, adv_correct, total = test_attack(args, iptresnet, test_loader, square_attack, True)
        message = f'test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        print(message, iptresnet.tau)



if __name__ == '__main__':  
    main()

