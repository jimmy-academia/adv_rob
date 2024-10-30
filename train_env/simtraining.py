import copy
import time
import torch
from tqdm import tqdm

from attacks.default import pgd_attack
from train_env.base_eval import Base_trainer

from torch.optim.lr_scheduler import StepLR

class AdversarialSimilarityTrainer(Base_trainer):
    def train(self):
        # assume: model.iptnet vs model.classifier
        self.model.train()
        self.model.to(self.args.device)
        optimizer = torch.optim.Adam(self.model.iptnet.parameters(), lr=self.args.lr)
        opt_class = torch.optim.Adam(self.model.classifier.parameters(), lr=self.args.lr)
        
        step_size = 10  # Example step size
        gamma = 0.5     # Reduce LR by half every step_size epochs
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_class = StepLR(opt_class, step_size=step_size, gamma=gamma)


        self.runtime = 0
        for self.epoch in range(1, self.num_epochs+1):

            # reconstruction pretrain
            pbar = tqdm(self.train_loader, ncols=90, desc='ast:pretrain')
            for images, labels in pbar:
                
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = self.model.iptnet(images)
                mseloss = torch.nn.MSELoss()(output, images)

                # Add the regularization loss for distinct embedding weights
                weight = self.model.iptnet.embedding.weight
                norm_weight = torch.nn.functional.normalize(weight, p=2, dim=1)
                cosine_similarity = torch.matmul(norm_weight, norm_weight.t())
                identity_mask = torch.eye(cosine_similarity.size(0), device=cosine_similarity.device)
                regloss = (cosine_similarity * (1 - identity_mask)).sum() / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))

                loss = mseloss + regloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.runtime += time.time() - start_time
                pbar.set_postfix(loss=loss.item())
                self.model.iptnet.reorder_embedding()


            # reconstruction adv pretrain
            pbar = tqdm(self.train_loader, ncols=90, desc='ast:adv pretrain')
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model.iptnet, images, True, attack_iters=7)
                output = self.model.iptnet(adv_images)
                mseloss = torch.nn.MSELoss()(output, images)

                # Add the regularization loss for distinct embedding weights
                weight = self.model.iptnet.embedding.weight
                norm_weight = torch.nn.functional.normalize(weight, p=2, dim=1)
                cosine_similarity = torch.matmul(norm_weight, norm_weight.t())
                identity_mask = torch.eye(cosine_similarity.size(0), device=cosine_similarity.device)
                regloss = (cosine_similarity * (1 - identity_mask)).sum() / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))

                loss = mseloss + regloss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.runtime += time.time() - start_time
                pbar.set_postfix(loss=loss.item())
                self.model.iptnet.reorder_embedding()


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
                opt_class.zero_grad()
                self.loss.backward()
                opt_class.step()

                self.runtime += time.time() - start_time
                self.total += len(cl_output)
                self.correct += (cl_output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item()})
                self.model.iptnet.reorder_embedding()


            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()
            scheduler_class.step()

            self.append_training_record()    
            self.periodic_check()


        return self.training_records


