import torch
import torch.nn as nn

from utils import check

from tqdm import tqdm
from sklearn.cluster import KMeans

def prepare_tokenembedder(args, tokenembedder, train_loader, test_loader):

    def test_tokenizer(args, tokenembedder, test_loader):
        tokenembedder.set_stage('token')
        correct = total = 0
        for images, __ in train_loader:
            images = images.to(args.device)
            pred = tokenembedder(images)

            tokenembedder.attackable = True
            adv_images = adv_perturb(images, tokenembedder, pred, args.eps, args.attack_iters)
            tokenembedder.attackable = False

            adv_pred = tokenembedder(adv_images)
            correct += (adv_pred == pred).sum()
            total += pred.numel()
        return correct, total

    optimizer = torch.optim.SGD(tokenembedder.tokenizer.parameters(), lr=0.01)

    if args.precluster_method == 'kmeans':
        images, __ = next(iter(train_loader))
        patches = tokenembedder.split_patch(images)
        patches = patches.view(-1, patches.size(-1))
        
        print('kmeans clustering fitting....')
        kmeans = KMeans(n_clusters=args.vocabulary_size, random_state=args.seed).fit(patches)
        kmeans_ids = kmeans.predict(patches)
        print('... done')
        
        patches = patches.to(args.device)
        kmeans_ids = torch.from_numpy(kmeans_ids).to(args.device).long()
        for __ in range(10):
            optimizer.zero_grad()
            output = tokenembedder.tokenizer(patches)
            loss = torch.nn.CrossEntropyLoss()(output, kmeans_ids)
            loss.backward()
            optimizer.step()
        correct, total = test_tokenizer(args, tokenembedder, test_loader)
        print(f'attacked tokenizer accuracy after kmeans: {correct/total:.4f}')

    tokenembedder.set_stage('token')
    pbar = tqdm(range(args.toktrain_epochs), ncols=70, desc='tr...tok')
    for epoch in pbar:
        for images, __ in train_loader:
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

        if (epoch+1) % (args.toktrain_epochs//20) == 0:
            correct, total = test_tokenizer(args, tokenembedder, test_loader)
            print(f'epoch: {epoch}| attacked tokenizer accuracy: {correct/total:.4f}')
            
def perpare_classifier(args, tokenembedder, classifier, train_loader, test_loader):
    pass


def adv_perturb(primary, model, pred, eps, num_iters):
    secondary = primary.clone().detach()
    for __ in range(num_iters):
        variable = secondary.clone().detach().requires_grad_(True)
        output = model(variable)
        # torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, pred)
        loss.backward()
        grad = variable.grad.data
        variable = variable + eps * torch.sign(grad)
        variable = variable.clamp(primary - eps, primary + eps)
        variable = variable.clamp(0, 1) # for images
        secondary = variable.detach()
    return secondary