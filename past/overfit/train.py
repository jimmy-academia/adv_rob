import torch
import torch.nn as nn
from tqdm import tqdm

from attack import adv_perturb

def advtrain_tokenizer(args, tokenizer, train_loader, test_loader):
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.001)
    pbar = tqdm(range(args.toktrain_epochs), ncols=90, desc='advtrain tokenizer')
    for epoch in pbar:
        for images, __ in tqdm(train_loader, ncols=70, desc='load image...token', leave=False):
            optimizer.zero_grad()
            images = images.to(args.device)
            patches = images.view(-1, images.size(1)*args.patch_size)
            pred = torch.argmax(tokenizer(patches), dim=1)
            adv_patches = adv_perturb(patches, tokenizer, pred, args.eps, args.attack_iters)
            adv_prob = tokenizer(adv_patches)

            loss = nn.CrossEntropyLoss()(adv_prob, pred)
            loss.backward()
            optimizer.step()
            adv_pred = torch.argmax(adv_prob, dim=1)
            accuracy_message = f'{(adv_pred == pred).sum()}/{pred.numel()} = {(adv_pred == pred).sum()/pred.numel():.4f}'
            pbar.set_postfix(r=accuracy_message)

        # if (epoch+1) % (args.toktrain_epochs//20) == 0 or epoch == args.toktrain_epochs-1:
        if True:
            correct = total = 0
            for images, __ in tqdm(test_loader, ncols=70, desc='test tokenizer', leave=False):
                images = images.to(args.device)
                patches = images.view(-1, images.size(1)*args.patch_size)
                pred = torch.argmax(tokenizer(patches), dim=1)
            
                adv_patches = adv_perturb(patches, tokenizer, pred, args.eps, args.attack_iters)
                adv_pred = torch.argmax(tokenizer(adv_patches), dim=1)

                correct += (adv_pred == pred).sum()
                total += pred.numel()
            print(f'epoch: {epoch}| attacked tokenizer accuracy: {correct/total:.4f}')

def train_classifier(args, iptresnet, tok_train_loader, test_loader):
    optimizer = torch.optim.Adam(iptresnet.parameters(), lr=0.001)
    train_pbar = tqdm(range(args.train_epochs), ncols=90, desc='train classifier')
    for epoch in train_pbar:
        for tokens, labels in tok_train_loader:
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            output = iptresnet.from_tokens(tokens)
            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = float((output.argmax(dim=1) == labels).sum()/len(labels))
            train_pbar.set_postfix(l=f'{float(loss):.2f}', acc=f'{accuracy:.3f}')
        if (epoch+1) % (args.train_epochs//20) == 0 or epoch == args.train_epochs-1:
            test_attack(args, iptresnet, test_loader)

def test_attack(args, iptresnet, test_loader):
    total = correct = adv_correct = 0
    for images, labels in tqdm(test_loader, ncols=90, desc='test_attack', unit='batch'):
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = iptresnet.inference(images)
        correct += (pred.argmax(dim=1) == labels).sum()

        adv_images = adv_perturb(images, iptresnet, labels, args.eps, args.attack_iters)

        adv_pred = iptresnet.inference(adv_images)
        adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
        total += len(labels)
    
    print(f'test accuracy: {correct/total:.4f}, adv accuracy: {adv_correct/total:.4f}')


