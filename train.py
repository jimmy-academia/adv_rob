import torch
from tqdm import tqdm

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

def train_classifier(args, model, train_loader):
    model.train()
    model.to(args.device)
    cor = tot = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for images, labels in train_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        output = model(images)

        loss = torch.nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
        cor += (output.argmax(dim=1) == labels).sum()
        tot += len(labels)
    accuracy = float(cor/tot)
    return accuracy    

