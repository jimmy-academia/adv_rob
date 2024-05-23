import torch
import torch.nn as nn

from data import tokenize_dataset, get_dataloader
from attack import adv_perturb

from utils import check
from tqdm import tqdm

def train(args, iptresnet, train_loader, test_loader):
    prepare_tokenembedder(args, iptresnet.tokenembedder, train_loader, test_loader)
    perpare_classifier(args, iptresnet, train_loader, test_loader)

def prepare_tokenembedder(args, tokenembedder, train_loader, test_loader):
    optimizer = torch.optim.SGD(tokenembedder.tokenizer.parameters(), lr=0.01)
    tokenembedder.set_stage('token')
    pbar = tqdm(range(args.toktrain_epochs), ncols=70, desc='tr...tok')
    for epoch in pbar:
        for images, __ in tqdm(train_loader, ncols=70, desc='toktrain', leave=False):
            optimizer.zero_grad()
            images = images.to(args.device)
            pred = tokenembedder(images)

            tokenembedder.attackable = True
            adv_images = adv_perturb(images, tokenembedder, pred, args.eps, args.attack_iters)
            adv_prob = tokenembedder(adv_images)
            tokenembedder.attackable = False

            loss = nn.CrossEntropyLoss()(adv_prob, pred)
            loss.backward()
            optimizer.step()
            adv_pred = torch.argmax(adv_prob, dim=1)
            accuracy_message = f'{(adv_pred == pred).sum()}/{pred.numel()} = {(adv_pred == pred).sum()/pred.numel():.4f}'
            pbar.set_postfix(r=accuracy_message)

            break #<<<=======
        # if (epoch+1) % (args.toktrain_epochs//20) == 0 or epoch == args.toktrain_epochs-1:
        if True:
            correct, total = test_tokenizer(args, tokenembedder, test_loader)
            print(f'epoch: {epoch}| attacked tokenizer accuracy: {correct/total:.4f}')
            # torch.save(tokenembedder.state_dict(), args.tokenembedder_path)

def test_tokenizer(args, tokenembedder, test_loader):
    tokenembedder.set_stage('token')
    correct = total = 0
    for images, __ in test_loader:
        images = images.to(args.device)
        pred = tokenembedder(images)

        tokenembedder.attackable = True
        adv_images = adv_perturb(images, tokenembedder, pred, args.eps, args.attack_iters)
        tokenembedder.attackable = False

        adv_pred = tokenembedder(adv_images)
        correct += (adv_pred == pred).sum()
        total += pred.numel()
    return correct, total

def perpare_classifier(args, iptresnet, train_loader, test_loader):

    optimizer = torch.optim.Adam(iptresnet.parameters(), lr=0.001)
    tok_train_set = tokenize_dataset(train_loader, iptresnet.tokenembedder.tokenizer, args.patch_size, args.device)
    tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)

    train_pbar = tqdm(range(args.train_epochs), ncols=90, desc='training', unit='epochs')
    for epoch in train_pbar:
        iptresnet.tokenembedder.set_stage('embedding')
        for tokens, labels in tok_train_loader:
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            output = iptresnet(tokens)
            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()

        if True:
        # if (epoch+1) % (args.train_epochs//20) == 0 or epoch == args.train_epochs-1:
            test_attack(args, iptresnet, test_loader)
            # torch.save(tokresnet.state_dict(), args.tokresnet_path)

def test_attack(args, iptresnet, test_loader):
    iptresnet.tokenembedder.set_stage('full')
    with torch.no_grad():
        total = correct = adv_correct = 0
        for images, labels in tqdm(test_loader, ncols=90, desc='test_attack', unit='batch'):
            images = images.to(args.device)
            labels = labels.to(args.device)
            pred = iptresnet(images)
            correct += (pred.argmax(dim=1) == labels).sum()
            
            iptresnet.tokenembedder.attackable = True
            adv_images = adv_perturb(images, iptresnet, pred, args.eps, args.attack_iters)
            iptresnet.tokenembedder.attackable = False
            adv_pred = iptresnet(adv_images)

            adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
            total += len(labels)
        
        print(f'test accuracy: {correct/total:.4f}, adv accuracy: {adv_correct/total:.4f}')


