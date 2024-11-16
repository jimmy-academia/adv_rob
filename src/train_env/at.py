import time
import torch
from tqdm import tqdm

from attacks.default import pgd_attack
from train_env.base_eval import BaseTrainer

class AdversarialTrainer(BaseTrainer):

    def train_setup(self):
        self.model.train()
        self.model.to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        

    def train_one_epoch(self):
        # instantiate/update: self.correct, self.total, self.runtime, self.loss

        self.total = self.correct = 0
        pbar = tqdm(self.train_loader, ncols=88, desc='adversarial training', leave=False)

        for images, labels in pbar:
            start_time = time.time()
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            adv_images = pgd_attack(self.args, images, self.model, labels, attack_iters=7)

            
            output = self.model(adv_images)
            self.loss = torch.nn.CrossEntropyLoss()(output, labels)
            
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
            self.runtime += time.time() - start_time
            self.total += len(output)
            self.correct += (output.argmax(dim=1) == labels).sum().item()
            pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item()})