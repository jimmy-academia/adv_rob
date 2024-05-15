import torch
import torch.nn as nn
from data import get_dataset, get_dataloader
from networks import Tokenizer, TokClassifier
from utils import check
from pathlib import Path
from tqdm import tqdm
# Load datasets
train_set, test_set = get_dataset('mnist')
train_loader, test_loader = get_dataloader(train_set, test_set, 128)

patch_size = 2  
T = 64 # T = 1024
Epochs = 1024
tokenizer = Tokenizer(patch_size**2, T)


if not Path('tokenizer.pth').exists():
    optimizer = torch.optim.SGD(tokenizer.parameters(), lr=0.01)
    split_patch = lambda x: x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous().view(x.size(0), -1, patch_size*patch_size*x.size(1))
    pbar = tqdm(range(Epochs), ncols=90, desc='training', unit='epochs')
    for epoch in pbar:
        for images, labels in tqdm(train_loader, ncols=90, desc='epoch', unit='batch', leave=False):
            pixels = split_patch(images)
            for iter in range(1280):
                embedded = tokenizer(pixels)
                softmax = torch.softmax(embedded, dim=1)
                pred = softmax.argmax(dim=1)
                # conf_loss = -softmax.max(dim=1)[0].mean() 

                # Adversarial attack
                pixels_clone = pixels.clone().detach()
                for i in range(1):
                    eps = 0.03
                    perturbed = pixels_clone.clone().detach().requires_grad_(True)
                    emb_perturbed = tokenizer(perturbed)
                    softmax_perturbed = torch.softmax(emb_perturbed, dim=1)
                    loss = nn.CrossEntropyLoss()(softmax_perturbed, pred)
                    loss.backward()
                    grad = perturbed.grad.data
                    perturbed = perturbed + eps * torch.sign(grad)
                    perturbed = perturbed.clamp(pixels - eps, pixels + eps)
                    pixels_clone = perturbed.detach()
                
                adv_embedded = tokenizer(pixels_clone)
                adv_softmax = torch.softmax(adv_embedded, dim=1)
                adv_pred = adv_softmax.argmax(dim=1)
                # adv_stab_loss = nn.MSELoss()(softmax, adv_softmax)
                adv_stab_loss = nn.CrossEntropyLoss()(adv_softmax, pred)
                # total_loss = 0.001* conf_loss + adv_stab_loss
                optimizer.zero_grad()
                adv_stab_loss.backward()
                # total_loss.backward()
                optimizer.step()
                # pbar.set_postfix(r=f'{(adv_pred == pred).sum()}/{pred.numel()}')
                pbar.set_postfix(r=f'{(adv_pred == pred).sum()/len(pred):.2f}/{pred.numel()//len(pred)}')
        if True:
        # if epoch % (Epochs//20) == 0:
            print()
            print(f'epoch {epoch}: {(adv_pred == pred).sum()/len(pred):.2f}/{pred.numel()//len(pred)}')
    tokenizer.to('cpu')
    torch.save(tokenizer.state_dict(), 'tokenizer.pth')


tokenizer.load_state_dict(torch.load('tokenizer.pth'))    
tokenizer = tokenizer.to(0)
# tokenize the dataset first, then train
tok_trainset = []
tok_testset = []

def tokenize_dataset(_set, tokenizer, patch_size=patch_size):
    tok_set = []
    for image, label in tqdm(_set, ncols=90, desc='tokenizing', unit='images', leave=False):
        image = image.to(0)
        h, w = image.shape[1:]
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).contiguous().view(-1, patch_size*patch_size*image.size(0))
        token_probability = tokenizer(patches)
        tok_image = torch.argmax(token_probability, dim=1)  # Assign the largest element in tok_image to tok_image
        tok_image = tok_image.view(int(w/patch_size), int(h/patch_size))
        tok_image = tok_image.to('cpu')
        tok_set.append((tok_image, label))
    return tok_set

tok_train_set = tokenize_dataset(train_set, tokenizer)
tok_test_set = tokenize_dataset(test_set, tokenizer)
tok_train_loader, tok_test_loader = get_dataloader(tok_train_set, tok_test_set)
    
# train model on tokenized data
classifer = TokClassifier(T).to(0)
optimizer = torch.optim.Adam(classifer.parameters(), lr=0.01)
for epoch in tqdm(range(1), ncols=90, desc='training', unit='epochs'):
    for images, labels in tok_train_loader:
        images = images.to(0)
        labels = labels.to(0)
        optimizer.zero_grad()
        output = classifer(images)
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        trainacc = 0
        count = 0
        for images, labels in tok_train_loader:
            images, labels = images.to(0), labels.to(0)
            output = classifer(images)
            trainacc += sum(output.argmax(dim=1).eq(labels).float())
            count += len(labels)
        trainacc = trainacc / count

        testacc = 0
        count = 0
        for images, labels in tok_test_loader:
            images, labels = images.to(0), labels.to(0)
            output = classifer(images)
            testacc += sum(output.argmax(dim=1).eq(labels).float())
            count += len(labels)
        testacc = testacc / count
        print()
        print(f'epoch {epoch}: train acc {trainacc}, test acc {testacc}')
    torch.save(classifer.state_dict(), 'classifier.pth')

