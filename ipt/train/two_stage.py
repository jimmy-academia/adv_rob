import torch
import torch.nn as nn
from tqdm import tqdm

from utils import check

def train_tokenizer(args, iptresnet, train_loader, adv_perturb):
    iptresnet.to(args.device)
    optimizer = torch.optim.Adam(iptresnet.tokenizer.parameters(), lr=0.001)
    pbar = tqdm(range(args.config['train']['toktrain_epochs']), ncols=90, desc='advtr. tokr.')
    anchors = torch.randn(args.vocab_size, args.patch_numel).to(args.device)
    init_kick = True
    for __ in pbar:
        for images, __ in tqdm(train_loader, ncols=70, desc='load image...token', leave=False):
            images = images.to(args.device)
            patches = iptresnet.patcher(images, True)
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
                for __ in range(10):
                    optimizer.zero_grad()
                    sim_loss = nn.CrossEntropyLoss()(iptresnet.tokenizer(patches), label)
                    sim_loss.backward()
                    optimizer.step()

            acc = float((torch.argmax(iptresnet.tokenizer(patches), dim=1) == label).float().mean()*100)

            # use original prediction as ground truth label
            pred = torch.argmax(iptresnet.tokenizer(patches), dim=1)
            adv_patches = adv_perturb(args, patches, iptresnet.tokenizer, pred)
            adv_prob = iptresnet.tokenizer(adv_patches)

            adv_loss = nn.CrossEntropyLoss()(adv_prob, pred)

            if init_kick:
                optimizer.zero_grad()
                adv_loss.backward()
                optimizer.step()

            adv_pred = torch.argmax(adv_prob, dim=1)
            adv_acc = int((adv_pred == pred).sum()/pred.numel()*100)

            if not init_kick:
                loss = (100 - acc)/10 * sim_loss + (100 - adv_acc)/10 * adv_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            new_anchor_sum = torch.zeros_like(anchors)
            count = torch.zeros(args.vocab_size, device=args.device, dtype=torch.long)  # Shape: (vocab_size,)
            new_anchor_sum = new_anchor_sum.scatter_add(0, label.unsqueeze(1).expand(-1, patches.size(1)), patches)
            count = count.scatter_add(0, label, torch.ones_like(label, dtype=torch.long))
            nonzero_counts = count > 0
            delta = args.config['train']['delta'] * new_anchor_sum[nonzero_counts] / count[nonzero_counts].unsqueeze(1)
            anchors[nonzero_counts] = (anchors[nonzero_counts] + delta)/(1+ args.config['train']['delta'])
            delta_sum = float(delta.abs().sum())
            
            accuracy_message = f'sim:{acc:.1f}% adv:{adv_acc}% d:{delta_sum:.2f}, l:{float(sim_loss):.1f},{float(adv_loss):.1f}'
            pbar.set_postfix(r=accuracy_message)

            if acc > 1: 
                init_kick = False

        iptresnet.embedding.weight = nn.Parameter(anchors)
        for param in iptresnet.embedding.parameters():
            param.requires_grad = False
    return iptresnet
    

def train_classifier(args, iptresnet, tok_train_loader):
    iptresnet.train()
    optimizer = torch.optim.Adam(iptresnet.parameters(), lr=0.001)
    train_pbar = tqdm(range(args.config['train']['train_epochs']), ncols=90, desc='train classifier')
    for i in train_pbar:
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

        # ntest = args.config['train']['num_tests']  
        # div = args.config['train']['train_epochs']//ntest if args.config['train']['train_epochs'] > ntest else 1
        # if (i+1) % div == 0:
            # correct, adv_correct, psadv_correct, total = test_attack(args, iptresnet, test_loader, adv_perturb)
            # message = f'train acc: {accuracy:.2f} test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}, ps acc {psadv_correct/total:.4f}...'

def test_attack(args, iptresnet, test_loader, adv_perturb):
    total = correct = adv_correct = psadv_correct = 0
    iptresnet.eval()
    for images, labels in tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False):
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = iptresnet.inference(images)
        correct += (pred.argmax(dim=1) == labels).sum()

        if args.bpda:
            adv_images = adv_perturb(args, images, iptresnet, labels)
        else:
            adv_images = adv_perturb(args, images, iptresnet.inference, labels)

        adv_pred = iptresnet.inference(adv_images)
        psadv_pred = iptresnet(adv_images)
        adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
        psadv_correct += (psadv_pred.argmax(dim=1) == labels).sum()
        total += len(labels)
    return correct, adv_correct, psadv_correct, total


