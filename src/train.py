import torch
import torch.nn as nn
from tqdm import tqdm

from attack import adv_perturb
from utils import check

from data import get_dataloader, tokenize_dataset

def incremental_testing(args, iptresnet, train_loader, test_loader):
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=0.001)
    pbar = tqdm(range(args.toktrain_epochs), ncols=90, desc='advtr. tokr.')
    anchors = torch.randn(args.vocab_size, args.patch_size).to(args.device)

    threshold_epoch = 1
    init_kick = True

    for epoch in pbar:
        for images, __ in tqdm(train_loader, ncols=70, desc='load image...token', leave=False):
            images = images.to(args.device)
            patches = images.view(-1, args.patch_size)
    
            # use l_infty distance to anchor as ground truth label
            # label = torch.argmax(torch.mm(patches, anchor.t()), dim=1)
            ## iterate over vocab_size

            min_dist = None
            for i, anchor in enumerate(anchors):
                # l_infty_dist = torch.amax(torch.abs(patches - anchor), dim=1)
                _dist = torch.linalg.norm(patches - anchor, dim=1)
                if min_dist is None:
                    min_dist = _dist
                    label = torch.tensor([i]*len(patches), device=args.device, dtype=torch.long)
                else:
                    mask = _dist < min_dist
                    min_dist[mask] = _dist[mask]
                    label[mask] = i
            sim_loss = nn.CrossEntropyLoss()(iptresnet.tokenizer(patches), label)
            
            if init_kick:
            # if epoch < threshold_epoch:
                for __ in range(10):
                # for __ in range(4):  ##mnist works
                    optimizer.zero_grad()
                    sim_loss = nn.CrossEntropyLoss()(iptresnet.tokenizer(patches), label)
                    sim_loss.backward()
                    optimizer.step()

            acc = float((torch.argmax(iptresnet.tokenizer(patches), dim=1) == label).float().mean()*100)

            # use original prediction as ground truth label
            pred = torch.argmax(iptresnet.tokenizer(patches), dim=1)
            adv_patches = adv_perturb(patches, iptresnet.tokenizer, pred, args.eps, args.attack_iters)
            adv_prob = iptresnet.tokenizer(adv_patches)

            adv_loss = nn.CrossEntropyLoss()(adv_prob, pred)

            if init_kick:
            # if epoch < threshold_epoch:
                optimizer.zero_grad()
                adv_loss.backward()
                optimizer.step()

            adv_pred = torch.argmax(adv_prob, dim=1)
            adv_acc = int((adv_pred == pred).sum()/pred.numel()*100)

            if not init_kick:
            # if epoch >= threshold_epoch:
                loss = (100 - acc)/10 * sim_loss + (100 - adv_acc)/10 * adv_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update anchor center of each cluster
            new_anchor_sum = torch.zeros_like(anchors)
            count = torch.zeros(args.vocab_size, device=args.device, dtype=torch.long)  # Shape: (vocab_size,)
            new_anchor_sum = new_anchor_sum.scatter_add(0, label.unsqueeze(1).expand(-1, patches.size(1)), patches)
            count = count.scatter_add(0, label, torch.ones_like(label, dtype=torch.long))
            nonzero_counts = count > 0
            delta = args.delta * new_anchor_sum[nonzero_counts] / count[nonzero_counts].unsqueeze(1)
            anchors[nonzero_counts] = (anchors[nonzero_counts] + delta)/(1+ args.delta)
            delta_sum = float(delta.abs().sum())
            # {(adv_pred == pred).sum()}/{pred.numel()}=
            
            accuracy_message = f'sim:{acc:.1f}% adv:{adv_acc}% d:{delta_sum:.2f}, l:{float(sim_loss):.1f},{float(adv_loss):.1f}'
            pbar.set_postfix(r=accuracy_message)

            if acc > 1: 
                init_kick = False

        iptresnet.embedding.weight = nn.Parameter(anchors)
        for param in iptresnet.embedding.parameters():
            param.requires_grad = False

        div = args.toktrain_epochs//20 if args.toktrain_epochs > 20 else 1 
        if (epoch+1) % div == 0 or epoch == args.toktrain_epochs-1:
            correct = total = 0
            for images, __ in tqdm(test_loader, ncols=70, desc='test iptresnet.tokenizer', leave=False):
                images = images.to(args.device)
                patches = images.view(-1, args.patch_size)
                pred = torch.argmax(iptresnet.tokenizer(patches), dim=1)
            
                adv_patches = adv_perturb(patches, iptresnet.tokenizer, pred, args.eps, args.attack_iters)
                adv_pred = torch.argmax(iptresnet.tokenizer(adv_patches), dim=1)

                correct += (adv_pred == pred).sum()
                total += pred.numel()
            print(f'epoch: {epoch}| attacked iptresnet.tokenizer accuracy: {correct/total:.4f}')


            tok_train_set = tokenize_dataset(train_loader, iptresnet.tokenizer, args.patch_size, args.device)
            tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)

            train_classifier(args, iptresnet, tok_train_loader, test_loader)
    

def train_classifier(args, iptresnet, tok_train_loader, test_loader):
    optimizer = torch.optim.Adam(iptresnet.parameters(), lr=0.001)
    train_pbar = tqdm(range(args.train_epochs), ncols=90, desc='train classifier')
    for epoch in train_pbar:
        cor = tot = 0
        for tokens, labels in tok_train_loader:
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            output = iptresnet.from_tokens(tokens)
            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
            cor += (output.argmax(dim=1) == labels).sum()
            tot += len(labels)
            accuracy = float(cor/tot)
            train_pbar.set_postfix(l=f'{float(loss):.2f}', acc=f'{accuracy:.3f}')
        div = args.train_epochs//20 if args.train_epochs > 20 else 1 
        if (epoch+1) % div == 0 or epoch == args.train_epochs-1:
            correct, adv_correct, total = test_attack(args, iptresnet, test_loader)
            print(f'train acc: {accuracy:.2f} test accuracy: {correct/total:.4f}, adv accuracy: {adv_correct/total:.4f}')



def test_attack(args, iptresnet, test_loader):
    total = correct = adv_correct = 0
    for images, labels in tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False):
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = iptresnet.inference(images)
        correct += (pred.argmax(dim=1) == labels).sum()

        adv_images = adv_perturb(images, iptresnet, labels, args.eps, args.attack_iters)

        adv_pred = iptresnet.inference(adv_images)
        adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
        total += len(labels)
    return correct, adv_correct, total


