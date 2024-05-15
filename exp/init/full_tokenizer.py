import torch
import torch.nn as nn

from data import get_dataset, get_dataloader
from networks import Tokenizer
from tqdm import tqdm

patch_size = 2  
T = 32

# Load datasets
train_set, test_set = get_dataset('mnist')
train_loader, test_loader = get_dataloader(train_set, test_set, 128)

# Extract pixel values from the dataset
# pixels = []
# for idx, (img, _) in enumerate(train_set):
#     h, w = img.shape[1:]
#     for y in range(0, h, patch_size):
#         for x in range(0, w, patch_size):
#             window = img[:, y:y+patch_size, x:x+patch_size].flatten()
#             pixels.append(window)
#     if idx == 99:
#         break
# pixels = torch.stack(pixels).to(0)

split_patch = lambda x: x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous().view(x.size(0), -1, patch_size*patch_size*x.size(1))

for images, labels in train_loader:
    pixels = split_patch(images)
    break
pixels = pixels.to(0)

# Model and optimizer
tokenizer = Tokenizer(patch_size**2, T).to(0)
optimizer = torch.optim.SGD(tokenizer.parameters(), lr=0.01)

# Training loop
# lamb = 0.1
# pbar = tqdm(range(10240), ncols=90, desc='token training', unit='epochs')
for epoch in range(10240):
    embedded = tokenizer(pixels)
    softmax = torch.softmax(embedded, dim=1)
    pred = softmax.argmax(dim=1)
    conf_loss = -softmax.max(dim=1)[0].mean() 

    # Adversarial attack
    pixels_clone = pixels.clone().detach()
    for i in range(10):
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
    total_loss = 0.01* conf_loss + adv_stab_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # pbar.set_postfix(r = f'{(adv_pred == pred).sum()}/{pred.numel()}')
    if epoch % 100 == 0:
        print(f'epoch {epoch}: {(adv_pred == pred).sum()}/{pred.numel()}')

# Save the tokenizer
# tokenizer.to('cpu')
# torch.save(tokenizer.state_dict(), 'tokenizer.pth')

    