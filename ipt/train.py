import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import check

from collections import defaultdict
from ipt.attacks import patch_square_attack, pgd_attack

def target_adv_training(args, iptresnet, train_loader):
    iptresnet.train()
    iptresnet.to(args.device)

    # [incorrect][correct]
    weight_map = torch.ones(args.vocab_size, args.vocab_size, device=args.device)
    error_map = torch.zeros(args.vocab_size, args.vocab_size, device=args.device).long() 
    # error_map -= torch.eye(args.vocab_size, device=args.device).long()

    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])
    pbar = tqdm(train_loader, ncols=88, desc='target_adv_patch')
    for iter_, (images, pred) in enumerate(pbar):
        images, pred = images.to(args.device), pred.to(args.device)
        patches = iptresnet.patcher(images)
        optimizer.zero_grad()
        # collect target attack towards each vocab
        all_adv_patches = []
        adv_pred = []
        new_pred = []
        for v in range(args.vocab_size):
            index_mask = pred != v
            if index_mask.sum() == 0:
                continue
            adv_patches = pgd_attack(args, patches[index_mask], iptresnet.tokenizer, v)
            all_adv_patches.append(adv_patches) # no support for full image perturbation yet
            adv_pred.append(iptresnet.tokenizer(adv_patches).argmax(1))
            new_pred.append(pred[index_mask])

        all_adv_patches = torch.cat(all_adv_patches, 0)
        adv_pred = torch.cat(adv_pred, 0)
        new_pred = torch.cat(new_pred, 0)

        # for each incorrect vocab, calculate the weighted loss
        for v in range(args.vocab_size):
            index_mask = adv_pred == v
            if index_mask.sum() == 0:
                continue
            loss = nn.CrossEntropyLoss(weight=weight_map[v])(iptresnet.tokenizer(all_adv_patches[index_mask]), new_pred[index_mask])
            loss.backward()
            fooled_idx = new_pred[index_mask] != v
            error_map[v].index_add_(0, new_pred[index_mask][fooled_idx], torch.ones_like(new_pred[index_mask][fooled_idx]))

        optimizer.step()
        break
        # masked_error_map = error_map * (error_map > 0)
        # max_vals, max_indices = torch.max(masked_error_map, dim=1)
        # rows = torch.arange(error_map.size(0), device=error_map.device)
        # pair_idx = torch.stack((rows[max_vals != 0], max_indices[max_vals != 0]), dim=1)

        # sorted_ind = error_map.sort()[1]
        # ranks = torch.arange(args.vocab_size, device=args.device).expand_as(sorted_ind)
        # ranks = torch.zeros_like(sorted_ind).scatter_(1, sorted_ind, ranks).scatter_(1, sorted_ind, ranks)
        # weight_map = torch.ones(args.vocab_size, args.vocab_size, device=args.device) * 2
        # weight_map[pair_idx[:, 0], pair_idx[:, 1]] = 1

        # pbar.set_postfix({'loss': f'{float(loss):.2f}', 'acc': f'{float((adv_pred == new_pred).float().mean()):.2f}'})
        # if (iter_+1) % 10 == 0:
        #     tqdm.write(f'loss: {float(loss):.2f} acc: {float((adv_pred == new_pred).float().mean()):.2f}')
    return error_map

def sub_patch_training(args, iptresnet, sub_loader):
    iptresnet.train()
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])
    for images, toklabels in sub_loader:
        images, toklabels = images.to(args.device), toklabels.to(args.device)
        patches = iptresnet.patcher(images, True)
        loss = nn.CrossEntropyLoss()(iptresnet.tokenizer(patches), toklabels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def fit_silos(silos, patch, eps):
    patch = patch.unsqueeze(0)
    joined = False
    for i in range(len(silos)):
        group = silos[i]
        if any ((group - patch).abs().max(1)[0] == 0):
            joined = True
            break
        if any((group - patch).abs().max(1)[0] < eps):
            silos[i] = torch.cat([group, patch], 0)
            joined = True
            break
    if not joined:
        silos.append(patch)
    return silos

def print_len_list(silos, pointer, image_counter):
    len_list = [len(s) for s in silos]
    len_list[pointer:] = ['|'] + len_list[pointer:]
    div = max(pointer*2, 20)
    len_list = len_list[:div] + [sum(len_list[div:])] + [f'image count:{image_counter} len:{len(silos)}']
    tqdm.write(f'{len_list}')

def patch_overlap_exploration(args, iptresnet, train_loader):
    iptresnet.patcher.to(args.device)
    silos = []
    image_counter = 0
    for images, labels in train_loader:
        image_counter += len(images)
        print('---------------')
        images, labels = images.to(args.device), labels.to(args.device)
        patches = iptresnet.patcher(images, True)
        for iter_, patch in enumerate(tqdm(patches, ncols=89, desc='patch overlap analysis...')):
            silos = fit_silos(silos, patch, args.eps)
            if (iter_+1) % (len(patches)//2) == 0 or iter_ == len(patches)-1:
                tqdm.write('depth-first re-analyse silos')
                pointer = 1
                silos.sort(key=lambda x: len(x), reverse=True)
                print_len_list(silos, pointer, image_counter)
                while pointer < len(silos):
                    if pointer > 5 and len(silos[pointer]) <= 10:
                        break
                    _length = len(silos[pointer-1])
                    sub_patches = torch.cat(silos[pointer:], 0)
                    silos = silos[:pointer]
                    for patch in sub_patches:
                        silos = fit_silos(silos, patch, args.eps)
                    silos.sort(key=lambda x: len(x), reverse=True)
                    print_len_list(silos, pointer, image_counter)
                    
                    if len(silos[pointer-1]) == _length:
                        pointer += 1
                    
                    
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

def adv_patch_training(args, iptresnet, prep_loader, attack_type='pgd'):
    iptresnet.train()
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=args.config['train']['lr'])

    pbar = tqdm(prep_loader, ncols=88, desc=f'{attack_type[0]}.adv_patch')

    # adv_set = []
    for images, pred in pbar:
        images, pred = images.to(args.device), pred.to(args.device)
        
        args_big = copy.deepcopy(args)
        args_big.eps = args.eps * 2

        if attack_type=='pgd':
            patches = iptresnet.patcher(images, True)
            adv_images = pgd_attack(args_big, patches, iptresnet.tokenizer, pred.view(-1))
            adv_images = iptresnet.patcher.inverse(adv_images.view(images.size(0), -1, args.patch_numel))
        elif attack_type== 'square':
            adv_images = patch_square_attack(args_big, images, iptresnet.tokenize_image, pred)
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

def train_classifier(args, iptresnet, train_loader, error_map=None):
    iptresnet.train()
    optimizer = torch.optim.Adam(iptresnet.classifier.parameters(), lr=0.001)
    train_pbar = tqdm(range(args.config['train']['train_epochs']), ncols=90, desc='train classifier')

    if error_map is not None:
        error_map = error_map/error_map.sum()
        indices = (error_map > 0).nonzero()
        probs = error_map[indices[:, 0], indices[:, 1]]
        
    for i in train_pbar:
        cor = tot = 0
        for images, labels in train_loader:

            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            if error_map is None:
                output = iptresnet(images)
            else:
                tok_pred = iptresnet.tokenize_image(images, True).argmax(1)
                rand_choice = torch.multinomial(probs, tok_pred.size(0), replacement=True)
                flip_ind = tok_pred == indices[rand_choice, 1]
                tok_pred[flip_ind] = indices[rand_choice[flip_ind], 0]

                tok_pred = tok_pred.view(images.size(0), -1)
                x = iptresnet.embedding(tok_pred)
                x = iptresnet.patcher.inverse(x)
                output = iptresnet.classifier(x)                

            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
            cor += (output.argmax(dim=1) == labels).sum()
            tot += len(labels)
            accuracy = float(cor/tot)
            train_pbar.set_postfix(l=f'{float(loss):.2f}', acc=f'{accuracy:.3f}')

def test_attack(args, iptresnet, test_loader, adv_perturb, fast=False):
    total = correct = adv_correct = psadv_correct = 0
    iptresnet.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = iptresnet(images)
        correct += (pred.argmax(dim=1) == labels).sum()

        adv_images = adv_perturb(args, images, iptresnet, labels)
        adv_pred = iptresnet(adv_images)
        adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
        total += len(labels)


        tok_pred = torch.argmax(iptresnet.tokenize_image(images), dim=2)
        adv_mean = (iptresnet.tokenize_image(adv_images).max(2)[1] == tok_pred).sum(1)/tok_pred.size(1)
        adv_mean = f'{float(adv_mean.sum()/len(adv_mean)):.2f}'

        if fast:
            print()
            print('macc:', adv_mean)
            print()
            break
        else:
            pbar.set_postfix({'macc': adv_mean})

    iptresnet.visualize_tok_image(images[0])
    iptresnet.visualize_tok_image(adv_images[0])

    return correct, adv_correct, total


    ## 2D scatter by linear indices and bincount
    # linear_ind = pred[index_mask][fooled_idx] * args.vocab_size + adv_pred[fooled_idx]
    # error_map += torch.bincount(linear_ind, minlength=args.vocab_size**2).view(args.vocab_size, args.vocab_size)
