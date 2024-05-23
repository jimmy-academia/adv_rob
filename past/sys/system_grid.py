import torch
import torch.nn as nn

from data import get_dataset, split_patch
from networks import Tokenizer
from train import adv_perturb
from utils import check, set_seeds

set_seeds(0)

print('conduct systematic comparisons')

train_set, test_set = get_dataset('mnist')

# Extract pixel values from the dataset

for patch_size in [2, 4, 8, 16]:
    init_patches = split_patch(train_set.data[:100].unsqueeze(1), patch_size).to(0)
    for vocabulary_size in [32, 128, 1024]:
        for eps in [0.01, 0.03, 0.1, 0.3]:
            for num_hidden_layer in range(2):
                tokenizer = Tokenizer(patch_size**2, vocabulary_size, num_hidden_layer).to(0)
                optimizer = torch.optim.SGD(tokenizer.parameters(), lr=0.01)

                print('==== working on ====')
                print(f'patch_size: {patch_size}, vocabulary_size: {vocabulary_size}, eps: {eps}, num_hidden_layer: {num_hidden_layer}')
                loss_log = ''
                countdown = False
                num_countdown = 0
                for epoch in range(10240):
                    embedded = tokenizer(init_patches)
                    softmax = torch.softmax(embedded, dim=1)
                    pred = softmax.argmax(dim=1)
                    conf_loss = -softmax.max(dim=1)[0].mean() 

                    adv_patches = adv_perturb(init_patches, tokenizer, pred, eps = eps)
                    adv_embedded = tokenizer(adv_patches)
                    adv_softmax = torch.softmax(adv_embedded, dim=1)
                    adv_pred = adv_softmax.argmax(dim=1)
                    adv_stab_loss = nn.CrossEntropyLoss()(adv_softmax, pred)
                    total_loss = 0.01* conf_loss + adv_stab_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    if epoch % 100 == 0:
                        print(f'epoch {epoch}: {(adv_pred == pred).sum()}/{pred.numel()}')
                    loss_log += f'epoch {epoch}: {(adv_pred == pred).sum()}/{pred.numel()} = {(adv_pred == pred).sum()/pred.numel():.4f}\n'
                    with open(f'logs/p{patch_size}_voc{vocabulary_size}_e{int(eps*100)}_nh{num_hidden_layer}.log', 'w') as f:
                        f.write(loss_log)

                    if (adv_pred == pred).sum() / pred.numel() > 0.9:
                        print('stability greater than 0.9')
                        countdown = True
                    if countdown:
                        num_countdown += 1
                        if num_countdown > 10:
                            break
                # Save the tokenizer
                tokenizer.to('cpu')
                torch.save(tokenizer.state_dict(), f'ckpt/p{patch_size}_voc{vocabulary_size}_e{int(eps*100)}_nh{num_hidden_layer}.pth')

            