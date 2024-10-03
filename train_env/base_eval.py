from tqdm import tqdm
from collections import defaultdict
from attacks.default import pgd_attack

class Base_trainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        if type(test_loader) == list:
            self.test_loader, self.rot_test_loader = test_loader ## for TTT
        else:
            self.test_loader = test_loader
        
        ## training
        self.num_epochs = 100
        ## logging
        self.epoch = 0
        self.training_records = defaultdict(list)
        self.eval_records = defaultdict(list)

    def train(self):
        raise NotImplementedError

    def append_training_record(self):
        self.training_records['epoch'].append(self.epoch)
        self.training_records['train_acc'].append(self.correct/self.total)            
        self.training_records['loss'].append(self.loss.cpu().item())
        self.training_records['runtime'].append(self.runtime)

        print(f'Epoch [{self.epoch}/{self.num_epochs}], '
          f'Train_acc: {self.correct/self.total:.4f}, Loss: {self.loss.item():.4f}')

    def eval(self):
        test_correct, adv_correct, test_total = test_attack(self.args, self.model, self.test_loader, pgd_attack)
        self.eval_records['epoch'].append(self.epoch)
        self.eval_records['test_acc'].append(test_correct/test_total)
        self.eval_records['adv_acc'].append(adv_correct/test_total)
        print(f'///eval/// Test_acc: {test_correct/test_total:.4f}, '
          f'Adv_acc: {adv_correct/test_total:.4f}, ')

    def periodic_check(self):
        if self.epoch % self.args.eval_interval == 0 and self.epoch != self.num_epochs:
            self.eval()


def test_attack(args, model, test_loader, adv_perturb):
    total = correct = adv_correct = 0
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        correct += float((pred.argmax(dim=1) == labels).sum())

        adv_images = adv_perturb(args, images, model, labels)
        adv_pred = model(adv_images)
        adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
        total += len(labels)
    return correct, adv_correct, total
