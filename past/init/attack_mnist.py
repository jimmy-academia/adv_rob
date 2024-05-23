import torch
import torch.nn as nn
from networks import Tokenizer, TokClassifier
from data import get_dataloader, get_dataset
from tqdm import tqdm
from utils import check

patch_size = 2  
T = 1024
tokenizer = Tokenizer(patch_size**2, T).to(0)
tokenizer.load_state_dict(torch.load('tokenizer.pth'))

classifer = TokClassifier(T).to(0)
classifer.load_state_dict(torch.load('classifier.pth'))
classifer.tokenizer = tokenizer

train_set, test_set = get_dataset('mnist')
train_loader, test_loader = get_dataloader(train_set, test_set)
attackacc = 0
count = 0
for images, labels in tqdm(test_loader, ncols=90, desc='attacking', unit='images', leave=False):
    images, labels = images.to(0), labels.to(0)

    images_clone = images.clone().detach()
    eps = 0.03
    for i in range(10):
        image_var = images_clone.clone().detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(classifer.inference_image(image_var), labels)
        loss.backward()
        grad = image_var.grad.data
        image_var = image_var + eps * torch.sign(grad)
        image_var = image_var.clamp(images - eps, images + eps)
        images_clone = image_var.detach()

    batch_patches = classifer.split_patch(images)
    token_probability = classifer.tokenizer(batch_patches)
    tokens = torch.argmax(token_probability, dim=2)

    batch_patches = classifer.split_patch(images_clone)
    token_probability_var = classifer.tokenizer(batch_patches)
    tokens_var = torch.argmax(token_probability_var, dim=2)
    
    output = classifer.inference_image(images_clone)
    attackacc += sum(output.argmax(dim=1).eq(labels).float())
    count += len(labels)

    check()
    
attackacc = attackacc / count
print('Attack accuracy:', attackacc.item())
