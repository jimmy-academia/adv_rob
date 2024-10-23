import torch
from tqdm import tqdm
from collections import defaultdict
from attacks.default import auto_attack, pgd_attack

class Base_trainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        ## training
        self.num_epochs = 75
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
        self.model.eval()
        _attack = auto_attack if self.args.attack_type == 'aa' else pgd_attack
        test_correct, adv_correct, test_total, tt_correct = self.test_attack(_attack)
        self.eval_records['epoch'].append(self.epoch)
        self.eval_records['test_acc'].append(test_correct/test_total)
        self.eval_records['test_time_acc'].append(tt_correct/test_total)
        self.eval_records['adv_acc'].append(adv_correct/test_total)
        print(f'///eval/// Test_acc: {test_correct/test_total:.4f}, Test_time_acc: {tt_correct/test_total:.4f}, '
          f'Adv_acc: {adv_correct/test_total:.4f}, ')

    def periodic_check(self):
        if self.epoch % self.args.eval_interval == 0 and self.epoch != self.num_epochs:
            self.eval()
        self.model.train()

    def test_attack(self, adv_perturb):
        printed=False
        total = test_correct = tt_correct = adv_correct = 0
        for images, labels in tqdm(self.test_loader, ncols=90, desc='test_attack', unit='batch', leave=False):
            images = images.to(self.args.device); labels = labels.to(self.args.device)
            pred = self.model(images)
            test_correct += float((pred.argmax(dim=1) == labels).sum())

            if self.args.test_time == 'none':
                model_copy = self.model
                tt_correct = test_correct
            else:
                if self.args.test_time == 'standard':
                    model_copy = self.test_time_training(images)
                elif self.args.test_time == 'online':
                    raise NotImplementedError("implement online test time training")
                tt_pred = model_copy(images)
                tt_correct += float((tt_pred.argmax(dim=1) == labels).sum())

            adv_images = adv_perturb(self.args, images, model_copy, labels)
            adv_pred = model_copy(adv_images)
            adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
            total += len(labels)
            
            if not printed:
                correct_ind = pred.argmax(dim=1) == labels
                incorrect_ind = adv_pred.argmax(dim=1) != labels
                indices = torch.nonzero(incorrect_ind*correct_ind).squeeze()[:5]

                from printer import display_images_in_grid
                tmpfilepath = f'ckpt/tmp/{self.epoch}.jpg' 
                testimages = images[indices]
                test_recon = model_copy.iptnet(testimages)
                advimages = adv_images[indices]
                diffimages = advimages - testimages
                adv_recons = model_copy.iptnet(advimages)

                Plot_images = []
                Plot_images.append([img.cpu() for img in testimages])
                Plot_images.append([img.cpu() for img in test_recon])

                Plot_images.append([img.cpu() for img in diffimages])

                Plot_images.append([img.cpu() for img in advimages])
                Plot_images.append([img.cpu() for img in adv_recons])
                display_images_in_grid(tmpfilepath, Plot_images, None, self.args.dataset, 1) 
                printed = True
                
            if self.args.attack_type == 'aa':
                break 

        return test_correct, adv_correct, total, tt_correct
