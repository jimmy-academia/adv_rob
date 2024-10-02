import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from attacks.default import pgd_attack
from train_env.eval import test_attack

class AdversarialTrainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        self.model.train()
        self.model.to(self.args.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        training_records = defaultdict(list)
        runtime = 0
        for epoch in range(1, 101):
            pbar = tqdm(self.train_loader, ncols=88, desc='adversarial training', leave=False)
            correct = total = 0
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model, labels, attack_iters=7)
                # Forward pass
                output = self.model(adv_images)
                loss = torch.nn.CrossEntropyLoss()(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                runtime += time.time() - start_time
                total += len(output)
                correct += (output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': correct/total, 'loss': loss.cpu().item()})
                
            training_records['epoch'].append(epoch)
            training_records['train_acc'].append(correct/total)
            
            test_correct, adv_correct, test_total = test_attack(self.args, model, self.test_loader, pgd_attack)
            training_records['test_acc'].append(test_correct/test_total)
            training_records['adv_acc'].append(adv_correct/test_total)
            training_records['loss'].append(loss.cpu().item())
            training_records['runtime'].append(runtime)

            print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item():.4f}')
        return training_records

