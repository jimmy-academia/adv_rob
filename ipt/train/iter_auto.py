import torch
import torch.nn as nn
from tqdm import tqdm

from ipt.attacks import patch_square_attack
from ipt.data import tokenize_dataset, get_dataloader
from ipt.train.two_stage import train_classifier, test_attack
from ipt.attacks import square_attack

from utils import check


def autoenc_adv_training(args, iptresnet, decoder, train_loader, test_loader, tau):
    iptresnet.train()
    iptresnet.to(args.device)
    decoder.to(args.device)
    optimizer = torch.optim.Adam(list(iptresnet.tokenizer.parameters()) + list(decoder.parameters()), lr=args.config['train']['lr'])
    # optimizer = torch.optim.Adam(list(iptresnet.tokenizer.parameters()) + list(iptresnet.embedding.parameters()) + list(decoder.parameters()), lr=args.config['train']['lr'])
    anchors = torch.randn(args.vocab_size, args.patch_numel).to(args.device)
    pbar = tqdm(train_loader, ncols=80, desc='auto')
    for iter_, (images, labels) in enumerate(pbar):

        # mse autoencoder training
        images, labels = images.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        mse_loss = torch.nn.MSELoss()(decoder(iptresnet.refit(images, tau)), images)
        mse_loss.backward()

        # update embedding to average of predicted token patches
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

        new_anchor_sum = torch.zeros_like(anchors)
        count = torch.zeros(args.vocab_size, device=args.device, dtype=torch.long)  # Shape: (vocab_size,)
        new_anchor_sum = new_anchor_sum.scatter_add(0, label.unsqueeze(1).expand(-1, patches.size(1)), patches)
        count = count.scatter_add(0, label, torch.ones_like(label, dtype=torch.long))
        nonzero_counts = count > 0
        anchors[nonzero_counts] =  new_anchor_sum[nonzero_counts] / count[nonzero_counts].unsqueeze(1)
        iptresnet.embedding.weight = nn.Parameter(anchors)    

        ## adv traininng
        if iter_ % 2 == 0:
            pred = torch.argmax(iptresnet.tokenize_image(images), dim=2)
            adv_images = patch_square_attack(args, images, iptresnet.tokenize_image, pred)
            adv_prob = iptresnet.tokenize_image(adv_images)
            adv_loss = nn.CrossEntropyLoss()(adv_prob.view(-1, adv_prob.size(-1)), pred.view(-1))
            adv_loss.backward()

            optimizer.step()
            tau *= 2**(1/len(train_loader))

            if ((iter_+1) % (len(train_loader)//20)) == 0:
                adv_acc = (adv_prob.max(2)[1] == pred).all(1)
                adv_mean = (adv_prob.max(2)[1] == pred).sum(1)/pred.size(1)
                tqdm.write(f'{iter_}) adv_acc: {float(adv_acc.sum()/len(adv_acc)):.4f}, adv_mean: {float(adv_mean.sum()/len(adv_mean)):.4f}, tau: {tau:.4f}, mse_loss: {mse_loss.item():.4f}, adv_loss: {adv_loss.item():.4f}') 
            pbar.set_postfix({'mse_loss': mse_loss.item(), 'adv_loss': adv_loss.item()})

        if ((iter_+1) % (len(train_loader)//5))== 0:
            tok_train_set = tokenize_dataset(train_loader, iptresnet, args.device)
            tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)
            train_classifier(args, iptresnet, tok_train_loader)
            correct, adv_correct, psadv_correct, total = test_attack(args, iptresnet, test_loader, square_attack)
            message = f'test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}, ps acc {psadv_correct/total:.4f}...'
            tqdm.write(message)


