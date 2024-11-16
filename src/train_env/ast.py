# copied from Avg_fineadjustTrainer

import copy
import time
import torch
from tqdm import tqdm

import operator
from functools import reduce

from attacks.default import pgd_attack
from train_env.base_eval import BaseTrainer

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

def get_downsample_loader(dataloader, downsample_ratio=0.1):
    dset = dataloader.dataset
    num_samples = int(len(dset) * downsample_ratio)

    indices = torch.randperm(len(dset))[:num_samples]  # Randomly permute the indices
    subset = Subset(dset, indices)
    downsampled_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)

    return downsampled_loader


class AdversarialSimilarityTrainer(BaseTrainer):

    def train_setup(self):
        # assume: model.iptnet vs model.classifier
        self.model.train()
        self.model.to(self.args.device)
        
        self.optimizer_embed = torch.optim.Adam(self.model.iptnet.embedding.parameters(), lr=1e-4)
        iptnet = self.model.iptnet
        pred_list = [iptnet.zero_predictor, iptnet.linear_predictor, iptnet.high_predictor]
        pred_list = [list(x.parameters()) for x in pred_list]
        self.optimizer_pred = torch.optim.Adam(reduce(operator.add, pred_list))
        self.optimizer_class = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        
        self.scheduler_embed = StepLR(self.optimizer_embed, step_size=self.args.step_size, gamma=self.args.gamma)
        self.scheduler_pred = StepLR(self.optimizer_pred, step_size=self.args.step_size, gamma=self.args.gamma)
        self.scheduler_class = StepLR(self.optimizer_class, step_size=self.args.step_size, gamma=self.args.gamma)

        self.embedding_loader = get_downsample_loader(self.train_loader)


    def train_one_epoch(self):
        # instantiate/update: self.correct, self.total, self.runtime, self.loss
        
        dual_steps = ["predictor", "embedding"]
        for step in dual_steps:
            print(f'step {dual_steps.index(step) + 1}, fix {(set(dual_steps) - {step}).pop()}, train {step}.')
            self.model.iptnet.embedding.requires_grad_(step == "embedding")
            self.model.iptnet.zero_predictor.requires_grad_(step == "predictor")
            self.model.iptnet.linear_predictor.requires_grad_(step == "predictor")
            self.model.iptnet.high_predictor.requires_grad_(step == "predictor")
            optimizer = self.optimizer_embed if step == 'embedding' else self.optimizer_pred
            scheduler = self.scheduler_embed if step == 'embedding' else self.scheduler_pred
            train_loader = self.embedding_loader if step == 'embedding' else self.train_loader

            pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
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

            pbar = tqdm(train_loader, ncols=90, desc='ast:adv pretrain')
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
            pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        

            ## todo, check if below can not joint dual step!

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
                self.optimizer_class.zero_grad()
                self.loss.backward()
                self.optimizer_class.step()

                self.runtime += time.time() - start_time
                self.total += len(cl_output)
                self.correct += (cl_output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item()})
                self.model.iptnet.reorder_embedding()


        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        self.scheduler_class.step()


