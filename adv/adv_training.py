import torch
from tqdm import tqdm
from adv import pgd_attack

def adversarial_training(args, model, train_loader):
    model.train()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.advtrain_lr)

    pbar = tqdm(train_loader, ncols=88, desc='adversarial training')
    correct = total = 0
    for images, labels in pbar:
        images, labels = images.to(args.device), labels.to(args.device)
        adv_images = pgd_attack(args, images, model, labels)
        output = model(adv_images)
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += len(output)
        correct += (output.argmax(dim=1) == labels).sum().item()
        pbar.set_postfix({'acc': correct/total})
    return correct/total
