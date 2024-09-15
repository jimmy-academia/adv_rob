import copy
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import check

from collections import defaultdict
from ipt.attacks import patch_square_attack, pgd_attack
from ipt.data import get_dataloader


def stable_training(args, iptresnet, train_loader):
    iptresnet.train()
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])
    # prep_set = []
    # for images, __ in pbar:
    #     images = images.to(args.device)
    #     patches = iptresnet.patcher(images, True)
    #     targets = iptresnet.patch_embed(patches)
    #     prep_set.extend([(patches[i].cpu(), targets[i].cpu()) for i in range(patches.size(0))])
        
    # prep_loader = get_dataloader(prep_set, args.batch_size)
    # pbar = tqdm(prep_loader, ncols=88, desc='stable')

    # num_patches = 0
    embed_grad = torch.zeros_like(iptresnet.embedding.weight)
    pbar = tqdm(train_loader, ncols=88, desc='similarity training')
    for images, __ in pbar:
        images = images.to(args.device)
        patches = iptresnet.patcher(images, True)
        adv_patches = pgd_attack(args, patches, iptresnet.patch_embed, patches, True)
        mseloss = nn.MSELoss()(iptresnet.patch_embed(adv_patches), patches)

        optimizer.zero_grad()
        mseloss.backward()
        embed_grad += iptresnet.embedding.weight.grad.detach()
        # num_patches += len(patches)
        optimizer.step()
        pbar.set_postfix(l=f'{float(mseloss):.4f}')

        new_embeddings = iptresnet.embedding.weight.detach() - embed_grad/len(patches)
        iptresnet.embedding.weight = nn.Parameter(new_embeddings)

def avg_patch_training(args, iptresnet, train_loader, kill=None):
    iptresnet.train()
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])
    anchors = iptresnet.embedding.weight.detach()

    pbar = tqdm(train_loader, ncols=88, desc='avg_patch')

    iter_ = 0
    for images, labels in pbar:
        if kill is not None:
            iter_ += 1
            if iter_ >= kill:
                break
        images, labels = images.to(args.device), labels.to(args.device)
        
        patches = iptresnet.patcher(images, True)
        min_dist = None
        for i, anchor in enumerate(anchors):
            _dist = torch.amax(torch.abs(patches - anchor), dim=1)
            # _dist = torch.linalg.norm(patches - anchor, dim=1)
            if min_dist is None:
                min_dist = _dist
                tok_label = torch.tensor([i]*len(patches), device=args.device, dtype=torch.long)
            else:
                mask = _dist < min_dist
                min_dist[mask] = _dist[mask]
                tok_label[mask] = i
        dist_loss = nn.CrossEntropyLoss()(iptresnet.tokenizer(patches), tok_label)
            
        optimizer.zero_grad()
        dist_loss.backward()
        optimizer.step()

        new_anchor_sum = torch.zeros_like(anchors)
        count = torch.zeros(args.vocab_size, device=args.device, dtype=torch.long)  # Shape: (vocab_size,)
        new_anchor_sum = new_anchor_sum.scatter_add(0, tok_label.unsqueeze(1).expand(-1, patches.size(1)), patches)
        count = count.scatter_add(0, tok_label, torch.ones_like(tok_label, dtype=torch.long))
        nz_idx = count > 0
        anchors[nz_idx] = new_anchor_sum[nz_idx] / count[nz_idx].unsqueeze(1)
        acc = (iptresnet.tokenizer(patches).argmax(1) == tok_label).float().mean().item()*100
        pbar.set_postfix(tok_pred_acc=acc)
            
    iptresnet.embedding.weight = nn.Parameter(anchors)
    iptresnet.visualize_tok_image(images[0])

def prep_dataset(args, iptresnet, train_loader):
    prep_set = []
    for images, __ in tqdm(train_loader, ncols=80, desc='preparing token_prediction', leave=False):
        images = images.to(args.device)
        pred = torch.argmax(iptresnet.tokenize_image(images), dim=2)
        prep_set.extend([(images[i].cpu(), pred[i].cpu()) for i in range(images.size(0))])
    return prep_set


def adv_patch_training(args, iptresnet, train_loader, attack_type='pgd'):
    iptresnet.train()
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])

    prep_set = prep_dataset(args, iptresnet, train_loader)
    prep_loader = get_dataloader(prep_set, batch_size = args.batch_size)
    pbar = tqdm(prep_loader, ncols=88, desc=f'{attack_type[0]}.adv_patch')

    # adv_set = []
    for images, pred in pbar:
        images, pred = images.to(args.device), pred.to(args.device)
        
        # args_big = copy.deepcopy(args)
        # args_big.eps = args.eps * 2

        if attack_type=='pgd':
            patches = iptresnet.patcher(images, True)
            adv_images = pgd_attack(args, patches, iptresnet.tokenizer, pred.view(-1))
            adv_images = iptresnet.patcher.inverse(adv_images.view(images.size(0), -1, args.patch_numel))
        elif attack_type== 'square':
            adv_images = patch_square_attack(args, images, iptresnet.tokenize_image, pred)
        else:
            raise ValueError(f'attack_type {attack_type} not defined')

        adv_prob = iptresnet.tokenize_image(adv_images)
        adv_loss = nn.CrossEntropyLoss()(adv_prob.view(-1, adv_prob.size(-1)), pred.view(-1))

        optimizer.zero_grad()
        adv_loss.backward()
        optimizer.step()

        adv_acc = (adv_prob.max(2)[1] == pred).all(1)
        adv_acc = float(adv_acc.sum()/len(adv_acc))
        adv_mean = (adv_prob.max(2)[1] == pred).sum(1)/pred.size(1)
        adv_mean = f'{float(adv_mean.sum()/len(adv_mean)):.2f}'
        pbar.set_postfix({'acc': f'{adv_acc:.2f}', 'macc': adv_mean, 'l': adv_loss.item()})




def do_adversarial_similarity_training(args, model, train_loader, test_loader, adv_attacks, atk_names):
    # assume: model.iptnet vs model.classifier
    Record = defaultdict(list)

    optimizer = torch.optim.Adam(model.iptnet.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    elapsed_time = 0
    for epoch in range(args.num_epochs):
        start = time.time()    
        pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            adv_images = pgd_attack(args, images, model.iptnet, images, True)
            output = model.iptnet(adv_images)
            mseloss = nn.MSELoss()(output, images)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()
            pbar.set_postfix(loss=mseloss.item())

        pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels, False)
            output = model.iptnet(adv_images)

            mseloss = nn.MSELoss()(output, images)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

            cl_output = model.classifier(output.detach())
            cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
            opt_class.zero_grad()
            cl_loss.backward()
            opt_class.step()

            adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})

        elapsed_time += time.time() - start
        Record['epoch'].append(epoch)
        Record['elapsed_time'].append(elapsed_time)
        Record = run_tests(args, model, test_loader, Record, adv_attacks, atk_names)
        
    return Record

def run_tests(args, model, test_loader, Record, adv_attacks, atk_names):

    for attack, atk_name in zip(adv_attacks, atk_names):
        correct, adv_correct, total = test_attack(args, model, test_loader, attack)
        print(f'[{atk_name}] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...')
        Record[f'{atk_name}_test_acc'].append(correct/total)
        Record[f'{atk_name}_adv_acc'].append(adv_correct/total)

    return Record


def test_attack(args, model, test_loader, adv_perturb):
    total = correct = adv_correct = 0
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        correct += float((pred.argmax(dim=1) == labels).sum())

        adv_images = adv_perturb(args, images, model, labels)
        adv_pred = model(adv_images)
        adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
        total += len(labels)
    return correct, adv_correct, total

# def test_attack(args, iptresnet, test_loader, adv_perturb, fast=False):
#     total = correct = adv_correct = 0
#     iptresnet.eval()
#     pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
#     for images, labels in pbar:
#         images = images.to(args.device)
#         labels = labels.to(args.device)
#         pred = iptresnet(images)
#         correct += (pred.argmax(dim=1) == labels).sum()

#         adv_images = adv_perturb(args, images, iptresnet, labels)
#         adv_pred = iptresnet(adv_images)
#         adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
#         total += len(labels)


#         tok_pred = torch.argmax(iptresnet.tokenize_image(images), dim=2)
#         adv_mean = (iptresnet.tokenize_image(adv_images).max(2)[1] == tok_pred).sum(1)/tok_pred.size(1)
#         adv_mean = f'{float(adv_mean.sum()/len(adv_mean)):.2f}'

#         if fast:
#             print()
#             print('macc:', adv_mean)
#             print()
#             break
#         else:
#             pbar.set_postfix({'macc': adv_mean})

#     iptresnet.visualize_tok_image(images[0])
#     iptresnet.visualize_tok_image(adv_images[0])

#     return correct, adv_correct, total


    ## 2D scatter by linear indices and bincount
    # linear_ind = pred[index_mask][fooled_idx] * args.vocab_size + adv_pred[fooled_idx]
    # error_map += torch.bincount(linear_ind, minlength=args.vocab_size**2).view(args.vocab_size, args.vocab_size)
