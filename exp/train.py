import torch
import torch.nn as nn

from data import split_patch

from collections import defaultdict
from tqdm import tqdm
from utils import check


def train_tokenizer(args, tokenizer, train_loader, test_loader):
    optimizer = torch.optim.SGD(tokenizer.parameters(), lr=0.01)
    pbar = tqdm(range(args.toktrain_epochs), ncols=70, desc='training', unit='epochs')
    for epoch in pbar:
        for images, __ in tqdm(train_loader, ncols=90, desc='train', unit='batch', leave=False):
            images = images.to(args.device)
            patches = split_patch(images, args.patch_size)
            adv_pred, pred = train_tokenizer_batch(args, patches, tokenizer, optimizer)
            pbar.set_postfix(r=f'{(adv_pred == pred).sum()/len(pred):.2f}/{pred.numel()//len(pred)}')

        if (epoch+1) % (args.toktrain_epochs//100) == 0:
            correct, total = test_tokenizer(args, tokenizer, test_loader)
            print(f'test accuracy: {correct/total:.2f}')
            
    # Save the tokenizer
    tokenizer.to('cpu')
    torch.save(tokenizer.state_dict(), args.tokenizer_path)

def test_tokenizer(args, tokenizer, test_loader):
    counter = defaultdict(lambda:0)
    counter['test_count'] = 0
    counter['test_correct'] = 0
    for images, __ in tqdm(test_loader, ncols=90, desc='test', unit='batch', leave=False):
        images = images.to(args.device)
        patches = split_patch(images, args.patch_size)
        adv_pred, pred = test_tokenizer_batch(args, patches, tokenizer)
        counter['test_correct'] += (adv_pred == pred).sum()
        counter['test_count'] += pred.numel()
    return counter['test_correct'], counter['test_count']

def train_tokenizer_batch(args, patches, tokenizer, optimizer):
    for iter in range(args.tok_batch_iters):
        pred = torch.softmax(tokenizer(patches), dim=1).argmax(dim=1)
        # Adversarial attack
        adv_patches = patches.clone().detach()
        attack_iters = max(10, int(args.attack_iters * (iter/args.tok_batch_iters)))
        adv_patches = adv_perturb(adv_patches, tokenizer, pred, eps = args.eps, num_iters = attack_iters)
        adv_softmax = torch.softmax(tokenizer(adv_patches), dim=1)
        adv_stab_loss = nn.CrossEntropyLoss()(adv_softmax, pred)

        optimizer.zero_grad()
        adv_stab_loss.backward()
        optimizer.step()
        adv_pred = adv_softmax.argmax(dim=1)
        return adv_pred, pred

def test_tokenizer_batch(args, patches, tokenizer):
    pred = torch.softmax(tokenizer(patches), dim=1).argmax(dim=1)
    adv_patches = patches.clone().detach()
    adv_patches = adv_perturb(adv_patches, tokenizer, pred, eps = args.eps, num_iters=args.attack_iters)
    adv_pred = torch.softmax(tokenizer(adv_patches), dim=1).argmax(dim=1)
    return adv_pred, pred

def adv_perturb(pixels, tokenizer, pred, eps, num_iters, inf=False):
    pixels_clone = pixels.clone().detach()
    for __ in range(num_iters):
        perturbed = pixels_clone.clone().detach().requires_grad_(True)
        if not inf:
            output = tokenizer(perturbed)
        else:
            output = tokenizer.inference_image(perturbed)
        torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, pred)
        loss.backward()
        grad = perturbed.grad.data
        perturbed = perturbed + eps * torch.sign(grad)
        perturbed = perturbed.clamp(pixels - eps, pixels + eps)
        perturbed = perturbed.clamp(0, 1) # for images
        pixels_clone = perturbed.detach()
    return pixels_clone

def train_classifier(args, classifer, tok_train_loader, tok_test_loader, test_loader):
    optimizer = torch.optim.Adam(classifer.parameters(), lr=0.01)
    train_pbar = tqdm(range(args.train_epochs), ncols=90, desc='training', unit='epochs')
    for epoch in train_pbar:
        for tokens, labels in tok_train_loader:
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            output = classifer(tokens)
            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            trainacc = 0
            count = 0
            for tokens, labels in tok_train_loader:
                tokens, labels = tokens.to(args.device), labels.to(args.device)
                output = classifer(tokens)
                trainacc += sum(output.argmax(dim=1).eq(labels).float())
                count += len(labels)
            trainacc = trainacc / count

            testacc = 0
            count = 0
            for tokens, labels in tok_test_loader:
                tokens, labels = tokens.to(args.device), labels.to(args.device)
                output = classifer(tokens)
                testacc += sum(output.argmax(dim=1).eq(labels).float())
                count += len(labels)
            testacc = testacc / count
            # print()
            # print(f'epoch {epoch}: train acc {trainacc}, test acc {testacc}')
            train_pbar.set_postfix(r=f'train: {trainacc:.2f}/test: {testacc:.2f}')
        if (epoch+1) % (args.train_epochs//20) == 0:
            attack_classifer(args, classifer, test_loader)
        torch.save(classifer.state_dict(), args.classifier_path)


def attack_classifer(args, classifer, test_loader):
    attackcount = 0
    count = 0
    for images, labels in tqdm(test_loader, ncols=90, desc='attacking', unit='images', leave=False):
        images, labels = images.to(args.device), labels.to(args.device)
        adv_images = adv_perturb(images, classifer, labels, eps = args.eps, num_iters = args.attack_iters, inf=True)
        output = classifer.inference_image(adv_images)
        attackcount += sum(output.argmax(dim=1).eq(labels).float())
        count += len(labels)

    attackacc = attackcount / count
    print('Attack accuracy:', attackacc.item())
