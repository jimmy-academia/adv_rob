'''
check the recon situation
'''
import random
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from attacks.default import pgd_attack, auto_attack
from networks.iptnet import APTNet
from networks.test_time import ResNetCifar
from networks.model_list import Dummy

from main import post_process_args
from datasets import get_dataloader, rev_norm_transform, cifar10_class_names

from printer import display_images_in_grid
from utils import *

def main():

    print('### Phase 2, check attack')

    args = default_args()
    args = post_process_args(args)
    args.vocab_size = 8
    args.patch_size = 1

    ckptdir = Path('ckpt')/'insp'
    ckptdir.mkdir(exist_ok=True)
    result_train_path = ckptdir/'train_result.jpg'
    result_attack_path = ckptdir/'attack_result.jpg'
    reverse_transform = rev_norm_transform(args.dataset) 

    # === 

    model = Dummy(APTNet(args), ResNetCifar(depth=26, classes=args.num_classes))

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Num, whatB = params_to_memory(num_parameters)
    param_msg = f'model param count: {num_parameters} ≈ {Num}{whatB}'
    print(f'>>> {param_msg}')
    # additional information for iptnet
    if 'ipt' in args.model or 'apt' in args.model:
        print(f'>>>> ipt config: vocab_size={args.vocab_size}, patch_size={args.patch_size}')
    
    print()
    print(model)
    print()
    print()
    train_loader, test_loader = get_dataloader(args)

    model.train()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.iptnet.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    ref_images, ref_labels = next(iter(train_loader))
    indices = random.sample(range(len(ref_images)), 5)
    
    ref_images = [ref_images[i] for i in indices]
    Plot_labels = [cifar10_class_names[ref_labels[i]] for i in indices]
    Plot_images = [[img.clone() for img in ref_images]]
    ref_images = torch.stack(ref_images).to(args.device)


    for epoch in range(20):

        # reconstruction pretrain
        pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model.iptnet, images, True, attack_iters=7)
            output = model.iptnet(adv_images)
            mseloss = nn.MSELoss()(output, images)

            # Add the regularization loss for distinct embedding weights
            weight = model.iptnet.embedding.weight
            norm_weight = nn.functional.normalize(weight, p=2, dim=1)
            cosine_similarity = torch.matmul(norm_weight, norm_weight.t())
            identity_mask = torch.eye(cosine_similarity.size(0), device=cosine_similarity.device)
            regloss = (cosine_similarity * (1 - identity_mask)).sum() / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))

            loss = mseloss + 0.01 * regloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=mseloss.item())

        recon_images = model.iptnet(ref_images)

        # adversarial similarity training
        pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')

        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels, False, attack_iters=7)
            output = model.iptnet(adv_images)

            mseloss = torch.nn.MSELoss()(output, images)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

            cl_output = model.classifier(output.detach())
            loss = torch.nn.CrossEntropyLoss()(cl_output, labels)
            opt_class.zero_grad()
            loss.backward()
            opt_class.step()

        recon_images = model.iptnet(ref_images)
        Plot_images.append([rimg.cpu() for rimg in recon_images])
            
        display_images_in_grid(result_train_path, Plot_images, Plot_labels, reverse_transform, verbose=1) 
        
        ## attack, select corect test images

        images, labels = next(iter(test_loader))
        images, labels = images.to(args.device), labels.to(args.device)
        output = model(images)
        correct = output.argmax(dim=1) == labels
        correct_ind = torch.nonzero(correct).squeeze()[:5]
        test_images, test_labels = images[correct_ind], labels[correct_ind]
        adv_Plot_images = [[img.cpu() for img in test_images]]

        adv_images = auto_attack(args, test_images, model, test_labels)
        adv_recons = model.iptnet(adv_images)
        adv_Plot_images.append([img.cpu() for img in adv_images])

        diff_imgs = []
        for advimg, img in zip(test_images, adv_images):
            diff = advimg - img 
            diff_imgs.append(diff.cpu())
        adv_Plot_images.append(diff_imgs)
        adv_Plot_images.append([img.cpu() for img in adv_recons])

        display_images_in_grid(result_attack_path, adv_Plot_images, None, reverse_transform, verbose=1) 


if __name__ == '__main__':
    main()


