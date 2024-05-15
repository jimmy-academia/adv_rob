import torch
import torch.nn as nn
from utils import check

def prepare_tokenembedder(args, tokenembedder, train_loader, test_loader):

    def test_tokenizer(args, tokenembedder, test_loader):
        tokenembedder.set_stage('train_token')
        correct = total = 0
        for images, __ in train_loader:
            images = images.to(args.device)
            pred = tokenembedder(images)
            adv_images = adv_perturb(images, tokenembedder, pred, args.eps, args.attack_iters)
            adv_pred = tokenembedder(adv_images)
            correct += (adv_pred == pred).sum()
            total += pred.numel()
        return correct, total

    if args.precluster_method == 'kmeans':
        images, __ = next(iter(train_loader))
        patches = tokenembedder.split_patch(images)
        patches = patches.view(-1, patches.size(-1))
        from sklearn.cluster import KMeans
        print('kmeans clustering fitting....')
        kmeans = KMeans(n_clusters=args.vocabulary_size, random_state=args.seed).fit(patches)
        kmeans_ids = kmeans.predict(patches)
        optimizer = torch.optim.SGD(tokenembedder.tokenizer.parameters(), lr=0.01)
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

    tokenembedder.set_stage('train_token')
    for epoch in range(args.toktrain_epochs):
        for images, __ in train_loader:
            optimizer.zero_grad()
            images = images.to(args.device)
            pred = tokenembedder(images)
            adv_images = adv_perturb(images, tokenembedder, pred, args.eps, args.attack_iters)
            adv_pred = tokenembedder(adv_images)
            loss = nn.CrossEntropyLoss()(adv_pred, pred)
            loss.backward()
            optimizer.step()
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
        torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, pred)
        loss.backward()
        grad = variable.grad.data
        variable = variable + eps * torch.sign(grad)
        variable = variable.clamp(primary - eps, primary + eps)
        variable = variable.clamp(0, 1) # for images
        secondary = variable.detach()
    return secondary