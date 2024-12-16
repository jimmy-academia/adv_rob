import copy
import time
import torch
from tqdm import tqdm

from attacks.default import pgd_attack
from train_env.base import BaseTrainer

from datasets import rotate_batch
from debug import *


class TestTimeTrainer(BaseTrainer):
    '''
    implement Test-Time Training
    '''
    def train_setup(self):
        self.model.train()
        self.model.to(self.args.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [50, 65], gamma=0.1, last_epoch=-1)

    def train_one_epoch(self):

        self.total = self.correct = rotcorrect = 0
        pbar = tqdm(self.train_loader, ncols=88, desc='TTT_training', leave=False)

        for images, labels, rotimages, rotlabels in pbar:
            start_time = time.time()
            images, labels, rotimages, rotlabels = [x.to(self.args.device) for x in [images, labels, rotimages, rotlabels]]

            output = self.model(images)
            self.loss = torch.nn.CrossEntropyLoss()(output, labels)
            
            rotoutput = self.model.sshead(rotimages)
            self.loss += torch.nn.CrossEntropyLoss()(rotoutput, rotlabels)

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
            self.runtime += time.time() - start_time
            self.total += len(output)
            rotcorrect += (output.argmax(dim=1) == labels).sum().item()
            self.correct += (rotoutput.argmax(dim=1) == rotlabels).sum().item()
            pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item(), 'rotac': rotcorrect/self.total})
            
        self.scheduler.step()

    def test_time_training(self, images):

        model_copy = copy.deepcopy(self.model)
        model_copy.to(self.args.device)
        model_copy.eval()
        optimizer = torch.optim.SGD(list(model_copy.extractor.parameters()) + list(model_copy.rotatehead.parameters()), lr=0.001)

        # check confidence
        # with torch.no_grad():
        #     rotoutput = model_copy.sshead(images)
        #     confidence = torch.nn.functional.softmax(rotoutput, dim=1)[:, 0]
        #     mask = confidence < 0.9
        
        for __ in range(self.args.test_time_iter):
            rotimages, rotlabels = rotate_batch(images, 'random')
            rotlabels = rotlabels.to(self.args.device)
            # rotimages, rotlabels = rotate_batch(images[mask], 'random')

            rotoutput = model_copy.sshead(rotimages)
            loss = torch.nn.CrossEntropyLoss()(rotoutput, rotlabels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model_copy

class TestTimeAdvTrainer(TestTimeTrainer):
    '''
    implement Test-Time Training + Adversarial Training
    '''
    def train_one_epoch(self):

        self.total = self.correct = rotcorrect = 0
        pbar = tqdm(self.train_loader, ncols=88, desc='TTAdv_training', leave=False)

        for images, labels in pbar:
            start_time = time.time()
            images, labels = [x.to(self.args.device) for x in [images, labels]]
            adv_images = pgd_attack(self.args, images, self.model, labels, attack_iters=7)
            output = self.model(adv_images)
            self.loss = torch.nn.CrossEntropyLoss()(output, labels)
            
            rotimages, rotlabels = rotate_batch(adv_images, 'random')
            rotlabels = rotlabels.to(self.args.device)
            rotoutput = self.model.sshead(rotimages)
            self.loss += torch.nn.CrossEntropyLoss()(rotoutput, rotlabels)

            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()
            
            self.runtime += time.time() - start_time
            self.total += len(output)
            rotcorrect += (output.argmax(dim=1) == labels).sum().item()
            self.correct += (rotoutput.argmax(dim=1) == rotlabels).sum().item()
            pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item(), 'rotac': rotcorrect/self.total})
            
        scheduler.step()

