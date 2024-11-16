# copied from Avg_fineadjustTrainer

import copy
import time
import torch
from tqdm import tqdm

import operator
from functools import reduce

from attacks.default import pgd_attack
from train_env.base_eval import Base_trainer

from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader, Subset

def get_downsample_loader(dataloader, downsample_ratio=0.1):
    dset = dataloader.dataset
    num_samples = int(len(dset) * downsample_ratio)

    indices = torch.randperm(len(dset))[:num_samples]  # Randomly permute the indices
    subset = Subset(dset, indices)
    downsampled_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)

    return downsampled_loader


class ZLQH_dirTrainer(Base_trainer):
    def train(self):
        # assume: model.iptnet vs model.classifier
        self.model.train()
        self.model.to(self.args.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        step_size = 10  # Example step size
        gamma = 0.5     # Reduce LR by half every step_size epochs
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        self.runtime = 0
        best_acc = 0
        for self.epoch in range(1, self.num_epochs+1):
            pbar = tqdm(self.train_loader, ncols=90, desc='ast:pretrain')
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = self.model.iptnet(images)
                loss = torch.nn.MSELoss()(output, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.runtime += time.time() - start_time
                pbar.set_postfix(loss=loss.item())

            pbar = tqdm(self.train_loader, ncols=90, desc='ast:adv pretrain')
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model.iptnet, images, True, attack_iters=7)
                output = self.model.iptnet(adv_images)
                loss = torch.nn.MSELoss()(output, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.runtime += time.time() - start_time
                pbar.set_postfix(loss=loss.item())

            # adversarial similarity training
            pbar = tqdm(self.train_loader, ncols=88, desc='adv/sim training')
        
            self.total = self.correct = 0
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model, labels, False, attack_iters=7)
                output = self.model.iptnet(adv_images)

                mseloss = torch.nn.MSELoss()(output, images)
                optimizer.zero_grad()
                mseloss.backward()
                optimizer.step()

                cl_output = self.model.classifier(output.detach())
                self.loss = torch.nn.CrossEntropyLoss()(cl_output, labels)
                optimizer.zero_grad()
                self.loss.backward()
                optimizer.step()

                self.runtime += time.time() - start_time
                self.total += len(cl_output)
                self.correct += (cl_output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item()})
                if not self.args.direct:
                    self.model.iptnet.reorder_embedding()


            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()

            self.append_training_record()    
            self.periodic_check()

            if best_acc < self.robust_acc:
                best_acc = self.robust_acc

        print(f'Best robust acc: {best_acc:.4f}')
        return self.training_records


