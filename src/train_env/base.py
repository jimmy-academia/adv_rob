import os
import time
import torch
import logging
from collections import defaultdict
from attacks import conduct_attack

class BaseTrainer:
    def __init__(self, args, model, train_loader, valid_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        ## logging
        self.epoch = 0
        self.num_epochs = args.num_epochs
        self.training_records = defaultdict(list)
        self.eval_records = defaultdict(list)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def train(self):
        self.train_setup() # for initializing self.model, self.optimizer, self.scheduler... etc.

        self.runtime = 0
        for self.epoch in range(1, self.num_epochs+1):
            self.model.train()
            
            start = time.time()
            self.train_one_epoch()
            self.runtime += time.time() - start

            self.valid_one_epoch()
            self.append_training_record()    
            self.periodic_save()
            self.periodic_check()

    def train_setup(self):
        raise NotImplementedError

    def train_one_epoch(self):
        # instantiate/update: self.correct, self.total, self.loss
        # calculate: self.loss
        raise NotImplementedError

    def valid_one_epoch(self):
        self.model.eval()
        self.val_loss = 0
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                self.val_loss += loss.item()
        self.val_loss /= len(self.valid_loader.dataset)

    def append_training_record(self):
        self.training_records['epoch'].append(self.epoch)
        self.training_records['loss'].append(self.loss.cpu().item()/self.total)
        self.training_records['val_loss'].append(self.val_loss)
        self.training_records['runtime'].append(self.runtime)

        logging.info(f'Epoch [{self.epoch}/{self.num_epochs}], '
          f'Train_acc: {self.correct/self.total:.4f}, Loss: {self.loss.item()/self.total:.4f}; Validation_Loss: {self.val_loss:.4f}')

    def periodic_check(self):
        if self.args.eval_interval > 0: # not -1
            is_eval_interval = self.epoch % self.args.eval_interval == 0
            if is_eval_interval or self.epoch == self.num_epochs:
                self.eval()

    def periodic_save(self):
        suffix = f'.pth.{self.epoch}' if self.epoch != self.num_epochs else '.pth'
        weight_path = self.args.record_path.with_suffix(suffix)
        
        is_save_interval = self.epoch % self.args.save_interval == 0
        if is_save_interval or self.epoch == self.num_epochs:
            torch.save(self.model.state_dict(), weight_path)

    def eval(self):
        self.model.eval()
        test_correct, adv_correct, test_total = conduct_attack(self.args, self.model, self.test_loader)
        self.robust_acc = adv_correct/test_total
        self.eval_records['epoch'].append(self.epoch)
        self.eval_records['test_acc'].append(test_correct/test_total)
        self.eval_records['adv_acc'].append(self.robust_acc)

        logging.info(f'///eval/// Test_acc: {test_correct/test_total:.4f}, Adv_acc: {adv_correct/test_total:.4f}, ')
        
