# copied from Avg_fineadjustTrainer

import copy
import torch
from tqdm import tqdm

import operator
from functools import reduce
import logging

from attacks import pgd_attack
from train_env.base import BaseTrainer

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
        self.scheduler_embed = StepLR(self.optimizer_embed, step_size=self.args.step_size, gamma=self.args.gamma)
        self.embedding_loader = get_downsample_loader(self.train_loader)

        # e.g. iptnet.predictor_list = [iptnet.zero_predictor, iptnet.linear_predictor, iptnet.quadratic_predictor, iptnet.high_predictor]
        if self.args.joint_train:
            self.optimizer_pred = torch.optim.Adam(self.model.iptnet.parameters())
        else:
            pred_list_params = [list(x.parameters()) for x in self.model.iptnet.predictor_list]
            self.optimizer_pred = torch.optim.Adam(reduce(operator.add, pred_list_params))
        self.optimizer_class = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        self.scheduler_pred = StepLR(self.optimizer_pred, step_size=self.args.step_size, gamma=self.args.gamma)
        self.scheduler_class = StepLR(self.optimizer_class, step_size=self.args.step_size, gamma=self.args.gamma)

    def train_one_epoch(self):
        # instantiate/update: self.correct, self.total, self.loss
        
        dual_steps = ["predictor", "embedding"]
        for step in dual_steps:
            logging.debug(f'step {dual_steps.index(step) + 1}, fix {(set(dual_steps) - {step}).pop()}, train {step}.')

            if (self.args.direct or self.args.joint_train) and step == "embedding":
                continue

            self.model.iptnet.embedding.requires_grad_(step == "embedding" or self.args.joint_train)
            [x.requires_grad_(step == "predictor") for x in self.model.iptnet.predictor_list]

            # self.model.iptnet.zero_predictor.requires_grad_(step == "predictor")
            # self.model.iptnet.linear_predictor.requires_grad_(step == "predictor")
            # self.model.iptnet.high_predictor.requires_grad_(step == "predictor")

            optimizer = self.optimizer_embed if step == 'embedding' else self.optimizer_pred
            scheduler = self.scheduler_embed if step == 'embedding' else self.scheduler_pred
            train_loader = self.embedding_loader if step == 'embedding' else self.train_loader

            pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
            for images, labels in pbar:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = self.model.iptnet(images)
                loss = torch.nn.MSELoss()(output, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

            pbar = tqdm(train_loader, ncols=90, desc='ast:adv pretrain')
            for images, labels in pbar:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model.iptnet, images, sim=True, attack_iters=7)
                output = self.model.iptnet(adv_images)
                loss = torch.nn.MSELoss()(output, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

            # adversarial similarity training
            pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        
            ## todo, check if below can not joint dual step!

            self.total = self.correct = self.loss = 0
            for images, labels in pbar:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model, labels, attack_iters=7)
                output = self.model.iptnet(adv_images)

                mseloss = torch.nn.MSELoss()(output, images)
                optimizer.zero_grad()
                mseloss.backward()
                optimizer.step()

                cl_output = self.model.classifier(output.detach())
                loss = self.criterion(cl_output, labels)
                self.optimizer_class.zero_grad()
                loss.backward()
                self.optimizer_class.step()
                self.loss += loss.detach()

                self.total += len(cl_output)
                self.correct += (cl_output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item()/self.total})
                self.model.iptnet.reorder_embedding()


        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        self.scheduler_class.step()


