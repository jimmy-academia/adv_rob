import os
import torch
from tqdm import tqdm
from collections import defaultdict
from attacks.default import auto_attack, pgd_attack

class BaseTrainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        ## training
        self.num_epochs = args.num_epochs
        ## logging
        self.epoch = 0
        self.training_records = defaultdict(list)
        self.eval_records = defaultdict(list)


        self.orig_allvis = None

    def train(self):
        self.train_setup() # for initializing self.model, self.optimizer, self.scheduler... etc.

        self.runtime = 0
        for self.epoch in range(1, self.num_epochs+1):
            self.train_one_epoch()
                
            self.append_training_record()    
            self.periodic_check()

    def train_setup(self):
        raise NotImplementedError

    def train_one_epoch(self):
        # instantiate/update: self.correct, self.total, self.runtime, self.loss
        raise NotImplementedError

    def append_training_record(self):
        self.training_records['epoch'].append(self.epoch)
        self.training_records['train_acc'].append(self.correct/self.total)            
        self.training_records['loss'].append(self.loss.cpu().item())
        self.training_records['runtime'].append(self.runtime)

        print(f'Epoch [{self.epoch}/{self.num_epochs}], '
          f'Train_acc: {self.correct/self.total:.4f}, Loss: {self.loss.item():.4f}')

    def periodic_check(self):
        if self.epoch % self.args.eval_interval == 0 and self.epoch != self.num_epochs:
            self.eval()
        self.model.train()

    def eval(self):
        self.model.eval()
        _attack = auto_attack if self.args.attack_type == 'aa' else pgd_attack
        test_correct, adv_correct, test_total, tt_correct = self.test_attack(_attack)
        self.robust_acc = adv_correct/test_total
        self.eval_records['epoch'].append(self.epoch)
        self.eval_records['test_acc'].append(test_correct/test_total)
        self.eval_records['test_time_acc'].append(tt_correct/test_total)
        self.eval_records['adv_acc'].append(self.robust_acc)

        print(f'///eval/// Test_acc: {test_correct/test_total:.4f}, Test_time_acc: {tt_correct/test_total:.4f}, '
          f'Adv_acc: {adv_correct/test_total:.4f}, ')
        print()
        
    def test_attack(self, adv_perturb):
        printed=False
        total = test_correct = tt_correct = adv_correct = 0
        sample_imgs = []
        sample_adv_imgs = []

        for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)):
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

            if not printed and not self.args.direct:
                correct_ind = pred.argmax(dim=1) == labels
                incorrect_ind = adv_pred.argmax(dim=1) != labels
                batch_indices = torch.nonzero(incorrect_ind*correct_ind).squeeze()[:5]
                
                sample_imgs+= [img for img in images[batch_indices]]
                sample_adv_imgs+= [img for img in adv_images[batch_indices]]
                

                final_cond = batch_idx == len(self.test_loader) - 1 and len(sample_imgs) > 0


                if len(sample_imgs) >= 5 or final_cond:
                    
                    sample_imgs = torch.stack(sample_imgs)
                    sample_adv_imgs = torch.stack(sample_adv_imgs)

                    all_vis = model_copy.iptnet.visualize_embeddings().cpu()
                    if len(all_vis[:5]) < 5:
                        padding = torch.zeros(5-len(all_vis), self.args.channels, self.args.image_size, self.args.image_size)
                        all_vis = torch.cat([all_vis, padding], dim=0)

                    train_imgs, __ = next(iter(self.train_loader))
                    sample_imgs = sample_imgs[:5] 
                    sample_adv_imgs = sample_adv_imgs[:5]  

                    from printer import display_images_in_grid
                    tmpfilepath = f'ckpt/tmp/{self.args.train_env}/{self.epoch}.jpg'
                    os.makedirs(f'ckpt/tmp/{self.args.train_env}', exist_ok=True)
                    # Perform reconstruction
                    train_imgs = train_imgs[:5].to(self.args.device)  #5, 3, 32, 32
                    train_recon = model_copy.iptnet(train_imgs)
                    test_recon = model_copy.iptnet(sample_imgs)
                    diffimages = sample_adv_imgs - sample_imgs
                    adv_recons = model_copy.iptnet(sample_adv_imgs)


                    display_images_in_grid(tmpfilepath, [all_vis, train_imgs, train_recon, sample_imgs, test_recon, diffimages, sample_adv_imgs, adv_recons], None, 1)

                    printed = True 

                
            if self.args.attack_type == 'aa':
                break 

        return test_correct, adv_correct, total, tt_correct
