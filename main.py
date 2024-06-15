import torch
import yaml
from config import config_arguments
from utils import set_seeds, check

from ipt.data import get_dataset, get_dataloader
from ipt.networks import build_ipt_model
from ipt.train import test_attack, avg_patch_training, target_adv_training, train_classifier
# , patch_overlap_exploration
from ipt.attacks import square_attack

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
    iptresnet.visualize_tok_image(test_set[0][0].to(args.device))
    print('===test?===')
    prep_set = prep_dataset(args, iptresnet, train_loader)
    prep_loader = get_dataloader(prep_set, batch_size=args.batch_size)

    error_map = target_adv_training(args, iptresnet, prep_loader)
    train_classifier(args, iptresnet, train_loader, error_map)
    correct, adv_correct, total = test_attack(args, iptresnet, test_loader, square_attack, True)
    message = f'test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
    print(message)


def prep_dataset(args, iptresnet, train_loader):
    prep_set = []
    for images, __ in tqdm(train_loader, ncols=80, desc='preparing token_prediction', leave=False):
        images = images.to(args.device)
        pred = torch.argmax(iptresnet.tokenize_image(images), dim=2)
        prep_set.extend([(images[i].cpu(), pred[i].cpu()) for i in range(images.size(0))])
    return prep_set



if __name__ == '__main__':  
    main()


    # print(f'======= epoch {epoch} =======')
    # patch_overlap_exploration(args, iptresnet, train_loader)
    # sub_loader = prep_sub_loader(args, train_loader)
    # sub_patch_training(args, iptresnet, sub_loader)
    
    # adv_patch_training(args, iptresnet, prep_loader)
    ####!!!
    # train_classifier(args, iptresnet, train_loader)
